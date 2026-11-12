import torch

from hybrid_wm.control.reachability import ReachabilityCritic, ReachObjective


def _clips(B=64, T=8, D=6, g=None):
    """Random walks: distance in steps is recoverable from the latents."""
    start = torch.randn(B, D, generator=g)
    steps = 0.6 * torch.randn(B, T - 1, D, generator=g)
    return torch.cat([start[:, None], start[:, None] + steps.cumsum(1)], 1)


def test_learns_monotone_temporal_distance():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    crit = ReachabilityCritic(6, max_steps=7, hidden=128)
    opt = torch.optim.Adam(crit.parameters(), lr=2e-3)
    for _ in range(800):
        opt.zero_grad()
        crit.loss(_clips(g=g), generator=g).backward()
        opt.step()
    with torch.no_grad():
        e = _clips(B=256, g=g)
        d1 = crit(e[:, 0], e[:, 1]).mean()
        d4 = crit(e[:, 0], e[:, 4]).mean()
        d7 = crit(e[:, 0], e[:, 7]).mean()
        other = crit(e[:, 0], e.roll(1, 0)[:, 3]).mean()
    assert d1 < d4 < d7 < other


def test_objective_shapes_match_goalmse_contract():
    crit = ReachabilityCritic(4, max_steps=3, hidden=16)
    info = {"predicted_emb": torch.randn(2, 5, 6, 4), "goal_emb": torch.randn(2, 1, 4)}
    cost = ReachObjective(crit)(info)
    assert cost.shape == (2, 5) and torch.isfinite(cost).all()
    assert (cost >= 0).all() and (cost <= crit.values.max()).all()
