import pytest
import torch

from hybrid_wm.integrations.lewm import HybridPredictor


def _pred():
    torch.manual_seed(0)
    p = HybridPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, backend="reference").eval()
    with torch.no_grad():  # leave AdaLN-zero identity init so every path is exercised
        for name, prm in p.named_parameters():
            if ".ada." in name:
                prm.normal_(0, 0.02)
    return p


@torch.no_grad()
def test_contract_shape_and_window():
    p = _pred()
    assert p.num_frames == 3
    for T in (1, 2, 3):                       # LeWM.rollout calls it with growing windows
        assert p(torch.randn(4, T, 192), torch.randn(4, T, 192)).shape == (4, T, 192)


@torch.no_grad()
def test_causal_within_window():
    p = _pred()
    x, c = torch.randn(2, 3, 192), torch.randn(2, 3, 192)
    base = p(x, c)
    x2 = x.clone()
    x2[:, 2] += 1.0
    torch.testing.assert_close(p(x2, c)[:, :2], base[:, :2], rtol=1e-5, atol=1e-5)


def test_parameter_matched_to_lewm_predictor():
    try:
        from stable_worldmodel.wm.lewm.module import Predictor
    except Exception as e:  # pragma: no cover
        pytest.skip(f"stable-worldmodel not installed: {e}")
    ref = Predictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=6, heads=16, mlp_dim=2048,
                    dim_head=64, dropout=0.1)
    n_ref = sum(x.numel() for x in ref.parameters())
    n_ours = sum(x.numel() for x in _pred().parameters())
    assert abs(n_ours - n_ref) / n_ref < 0.01


def test_trains_and_reports_aux():
    p = _pred().train()
    y = p(torch.randn(4, 3, 192), torch.randn(4, 3, 192))
    (y.square().mean() + p.aux_loss).backward()
    assert p.aux_loss.item() > 0
