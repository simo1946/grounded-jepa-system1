import torch

from hybrid_wm.control.latent_policy import ChunkPolicy, hindsight_batch


def test_learns_goal_conditioned_chunks():
    """Toy dynamics z_{j+1} = z_j + W a_j: the policy must recover the chunk
    that leads from z_t to z_goal."""
    torch.manual_seed(0)
    D, H, A, B = 8, 3, 2, 256
    W = torch.randn(D, A)
    pol = ChunkPolicy(D, H, A, hidden=128)
    opt = torch.optim.Adam(pol.parameters(), lr=2e-3)

    def data():
        acts = torch.randn(B, H + 1, A)
        z = [torch.randn(B, D)]
        for j in range(H):
            z.append(z[-1] + acts[:, j] @ W.T)
        return torch.stack(z, 1), acts

    for _ in range(600):
        emb, acts = data()
        z_t, z_g, chunk, mask = hindsight_batch(emb, acts, H)
        mask[:] = False
        mask[:, 0] = True              # first block is identifiable from (z_t, z_{t+1})-style goals
        z_g = emb[:, 1]
        opt.zero_grad()
        pol.loss(z_t, z_g, chunk, mask).backward()
        opt.step()
    emb, acts = data()
    mean, std = pol(emb[:, 0], emb[:, 1])
    err = (mean[:, 0] - acts[:, 0]).pow(2).mean()
    assert err < 0.05 * acts[:, 0].pow(2).mean()
    assert (std[:, 0].mean() < 0.5)                          # confident where it is right


def test_hindsight_mask_and_shapes():
    g = torch.Generator().manual_seed(1)
    emb, acts = torch.randn(64, 6, 4), torch.randn(64, 6, 2)
    z_t, z_g, chunk, mask = hindsight_batch(emb, acts, 5, g)
    assert z_t.shape == (64, 4) and chunk.shape == (64, 5, 2) and mask.shape == (64, 5)
    k = mask.sum(1)
    assert (k >= 1).all() and (k <= 5).all()
    for i in range(64):                                      # goal really is frame k
        assert torch.equal(z_g[i], emb[i, int(k[i])])


def test_sample_first_candidate_is_mean_and_nan_targets_masked():
    pol = ChunkPolicy(4, 2, 3, hidden=16)
    z = torch.randn(5, 4)
    c = pol.sample(z, z, 7)
    mean, _ = pol(z, z)
    assert c.shape == (5, 7, 2, 3)
    torch.testing.assert_close(c[:, 0], mean)
    acts = torch.randn(5, 2, 3)
    acts[0, 0] = float("nan")
    loss = pol.loss(z, z, acts, torch.ones(5, 2, dtype=torch.bool))
    assert torch.isfinite(loss)


def test_policy_pool_select_and_mean():
    import pytest
    from hybrid_wm.control.latent_policy import PolicyPool
    torch.manual_seed(0)
    members = [ChunkPolicy(8, 5, 4) for _ in range(3)]
    z, g = torch.randn(6, 8), torch.randn(6, 8)
    sel = PolicyPool(members, "select")
    c = sel.sample(z, g, 3)
    assert c.shape == (6, 3, 5, 4)
    for i, m in enumerate(members):                                  # candidate i = member i's mean
        assert torch.allclose(c[:, i], m(z, g)[0])
    avg = PolicyPool(members, "mean").sample(z, g, 1)
    assert avg.shape == (6, 1, 5, 4) and torch.allclose(avg[:, 0], c.mean(1))
    with pytest.raises(ValueError):
        sel.sample(z, g, 16)                                         # k must match the pool
    with pytest.raises(ValueError):
        PolicyPool([members[0], ChunkPolicy(8, 4, 4)])               # mismatched horizons
    with pytest.raises(ValueError):
        PolicyPool(members[:1])


def test_flow_policy_represents_two_modes_where_gaussian_averages():
    """Each goal is reachable by action +1 or -1 (50/50). A Gaussian head's mean
    collapses to ~0; flow samples must land near +-1 with both modes present."""
    from hybrid_wm.control.latent_policy import FlowChunkPolicy, policy_from_checkpoint
    torch.manual_seed(0)
    D, H, A, B = 4, 2, 1, 512
    flow, gauss = FlowChunkPolicy(D, H, A, hidden=128, n_steps=20), ChunkPolicy(D, H, A, hidden=128)
    opt = torch.optim.Adam([*flow.parameters(), *gauss.parameters()], lr=2e-3)
    mask = torch.ones(B, H, dtype=torch.bool)
    for _ in range(1500):
        z, g = torch.randn(B, D), torch.randn(B, D)
        sign = torch.where(torch.rand(B, 1, 1) < 0.5, -1.0, 1.0)
        act = sign.expand(B, H, A).clone()
        opt.zero_grad()
        (flow.loss(z, g, act, mask) + gauss.loss(z, g, act, mask)).backward()
        opt.step()
    z, g = torch.randn(64, D), torch.randn(64, D)
    s = flow.sample(z, g, 32, generator=torch.Generator().manual_seed(1))       # [64, 32, H, A]
    near_mode = ((s.abs() - 1).abs() < 0.3).float().mean()
    assert near_mode > 0.8                                                         # samples sit on the modes
    pos = (s[..., 0, 0] > 0).float().mean()
    assert 0.25 < pos < 0.75                                                       # both modes are used
    assert gauss(z, g)[0].abs().mean() < 0.3                                       # the Gaussian mean averages them
    # candidate 0 is deterministic; std_scale 0 -> all candidates equal candidate 0
    s0 = flow.sample(z, g, 4, generator=torch.Generator().manual_seed(2), std_scale=0.0)
    assert torch.allclose(s0[:, 1], s0[:, 0]) and torch.allclose(s0[:, 0], s[:, 0], atol=1e-6)
    # checkpoint round trip
    pk = {"policy_type": "flow", "latent_dim": D, "horizon": H, "action_dim": A, "flow_steps": 20,
          "state_dict": flow.state_dict()}
    flow2 = FlowChunkPolicy(D, H, A, hidden=128, n_steps=20)
    flow2.load_state_dict(pk["state_dict"])
    assert torch.allclose(flow2.sample(z, g, 1), s[:, :1], atol=1e-6)
    import pytest
    with pytest.raises(ValueError):
        policy_from_checkpoint({**pk, "policy_type": "nope"})


def test_residual_policy_starts_at_base_and_applies_per_block():
    from hybrid_wm.control.latent_policy import ResidualChunkPolicy, policy_from_checkpoint
    torch.manual_seed(0)
    base = ChunkPolicy(8, 5, 10)                       # 5 blocks x (2 steps x 5 dims)
    res = ResidualChunkPolicy(base, 8, n_blocks=5, block_dim=10, step_dim=5)
    z, g = torch.randn(4, 8), torch.randn(4, 8)
    s = res.sample(z, g, 3, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(s[:, 0], base(z, g)[0])     # zero-initialised residual = base policy
    corr = torch.arange(25.).repeat(4, 1)
    out = res.apply(base(z, g)[0], corr)
    d = (out - base(z, g)[0]).view(4, 5, 2, 5)
    assert torch.allclose(d[:, :, 0], d[:, :, 1])       # same correction on both steps of a block
    assert torch.allclose(d[0, 1, 0], torch.arange(5., 10.))
    pk = {"policy_type": "residual", "base": {"policy_type": "gauss", "latent_dim": 8, "horizon": 5, "action_dim": 10,
                                              "state_dict": base.state_dict()},
          "residual": {"latent_dim": 8, "n_blocks": 5, "block_dim": 10, "step_dim": 5}, "state_dict": res.state_dict()}
    assert torch.allclose(policy_from_checkpoint(pk).sample(z, g, 1)[:, 0], s[:, 0])
