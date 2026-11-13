import pytest
import torch

from hybrid_wm.grounding import InverseDynamicsHead, StateAlignmentHead


def test_inverse_dynamics_learns_a_known_mapping():
    """If z_{t+1} = z_t + W a_t, the head must recover a_t from (z_t, z_{t+1})."""
    torch.manual_seed(0)
    B, T, D, A = 64, 4, 16, 3
    W = torch.randn(D, A)
    z0 = torch.randn(B, D)
    acts = torch.randn(B, T, A)
    zs = [z0]
    for t in range(T - 1):
        zs.append(zs[-1] + acts[:, t] @ W.T)
    emb = torch.stack(zs, 1)
    head = InverseDynamicsHead(D, A, hidden=64)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    first = head.loss(emb, acts).item()
    for _ in range(400):
        opt.zero_grad()
        loss = head.loss(emb, acts)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05 * first


def test_inverse_dynamics_masks_nan_and_uses_only_transitions():
    torch.manual_seed(0)
    head = InverseDynamicsHead(8, 2, hidden=16)
    emb = torch.randn(2, 4, 8)
    acts = torch.randn(2, 4, 2)
    base = head.loss(emb, acts)
    acts2 = acts.clone()
    acts2[:, -1] = 1e6                          # last action has no next frame -> ignored
    torch.testing.assert_close(head.loss(emb, acts2), base)
    acts3 = acts.clone()
    acts3[0, 1] = float("nan")                  # boundary NaN -> masked, loss stays finite
    assert torch.isfinite(head.loss(emb, acts3))
    all_nan = torch.full_like(acts, float("nan"))
    assert head.loss(emb, all_nan).item() == 0.0


def test_gradients_reach_the_encoder_side():
    head = StateAlignmentHead(8, 3, hidden=16)
    emb = torch.randn(2, 4, 8, requires_grad=True)
    head.loss(emb, torch.randn(2, 4, 3)).backward()
    assert emb.grad is not None and emb.grad.abs().sum() > 0


def test_state_alignment_targets_level_and_delta():
    """With a perfect head the loss is zero only if both s_{t+1} and s_{t+1}-s_t are right."""
    head = StateAlignmentHead(4, 2, hidden=8)
    state, emb = torch.randn(1, 3, 2), torch.randn(1, 3, 4)
    with torch.no_grad():
        s_hat, ds_hat = head(torch.zeros(1, 2, 4), torch.zeros(1, 2, 4))
    assert s_hat.shape == ds_hat.shape == (1, 2, 2)
    level, delta = state[:, 1:], state[:, 1:] - state[:, :-1]
    head.forward = lambda z_t, z_next: (level, delta)                        # a perfect head
    assert head.loss(emb, state).item() == pytest.approx(0.0, abs=1e-12)
    head.forward = lambda z_t, z_next: (level, torch.zeros_like(delta))      # right level, wrong delta
    assert head.loss(emb, state).item() > 0
    head.forward = lambda z_t, z_next: (torch.zeros_like(level), delta)      # wrong level, right delta
    assert head.loss(emb, state).item() > 0
    with pytest.raises(ValueError):
        head.loss(torch.randn(1, 3, 4), torch.randn(1, 2, 2))


def test_action_resolution_loss_rewards_action_sensitive_predictors():
    import torch

    from hybrid_wm.grounding import action_resolution_loss, perturb_actions

    torch.manual_seed(0)
    B, T, A, D = 64, 3, 5, 8
    W_true = torch.randn(A, D)
    ctx, act = torch.randn(B, T, D), torch.randn(B, T, A)
    target = ctx + act @ W_true                                   # outcome depends on the action
    cand = perturb_actions(act, 4, generator=torch.Generator().manual_seed(1))
    assert cand.shape == (B, 5, T, A) and torch.equal(cand[:, 0], act)
    sensitive = ctx.unsqueeze(1) + cand @ W_true                  # exact model
    blind = ctx.unsqueeze(1).expand(B, 5, T, D) + 0.1             # ignores the action
    l_s, acc_s = action_resolution_loss(sensitive, target)
    l_b, acc_b = action_resolution_loss(blind, target)
    assert float(acc_s) == 1.0 and float(acc_b) == 0.0            # ties do not count as correct
    assert float(l_s) < float(l_b)
    assert abs(float(l_b) - float(torch.log1p(torch.tensor(0.1)))) < 1e-4   # blind: hinge at the margin


def test_action_resolution_loss_is_bounded():
    """Perturbed predictions already (1 + gamma) farther get no gradient (no runaway inflation)."""
    import torch

    from hybrid_wm.grounding import action_resolution_loss

    target = torch.zeros(4, 3, 8)
    pred = torch.zeros(4, 5, 3, 8)
    pred[:, 0] += 0.1
    pred[:, 1:] += 1.0                                            # 100x farther in squared distance
    pred.requires_grad_(True)
    loss, acc = action_resolution_loss(pred, target)
    loss.backward()
    assert float(loss) == 0.0 and float(acc) == 1.0 and float(pred.grad.abs().sum()) == 0.0


def test_action_resolution_loss_gradient_reaches_predictor():
    import torch

    from hybrid_wm.grounding import action_resolution_loss, perturb_actions

    W = torch.zeros(5, 8, requires_grad=True)
    ctx, act = torch.randn(16, 3, 8), torch.randn(16, 3, 5)
    target = ctx + act @ torch.randn(5, 8)
    cand = perturb_actions(act, 4)
    loss, _ = action_resolution_loss(ctx.unsqueeze(1) + cand @ W, target)
    loss.backward()
    assert W.grad is not None and float(W.grad.abs().sum()) > 0
