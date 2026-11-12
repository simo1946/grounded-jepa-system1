"""Contrastive action verifier, after CLM-8B / CoVer-VLA (prereg_critic.md, addendum E10).

score(z_t, z_goal, chunk) = cos(f(z_t, z_goal), g(chunk)) / temperature, trained with a symmetric InfoNCE
loss on expert windows from the latent cache (the executed chunk is the positive, the other chunks in the
batch the negatives; optionally perturbed copies of the positive as hard negatives). Used two ways:
  * as a selector among System 1 candidates (evaluate.py --select verifier), and
  * as a retrieval policy over a cached menu of expert chunks (policy_type 'retrieval').

python contrastive_verifier.py --spec lewm+idm+sa+obj --out verifier_in.pt
python contrastive_verifier.py --spec lewm+idm+sa+obj --hard 2 --out verifier_hard.pt
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_flush_denormal(True)

from cached_policy import H, OfflineWindows  # noqa: E402
from latent_cache import cache_path  # noqa: E402
from models import HOME, Spec  # noqa: E402

SUCCESS_M = 0.04


def _mlp(i: int, o: int, hidden: int = 512) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, hidden), nn.LayerNorm(hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, o))


class ContrastiveVerifier(nn.Module):
    def __init__(self, latent_dim: int, chunk_dim: int, emb: int = 256):
        super().__init__()
        self.chunk_dim = chunk_dim
        self.f = _mlp(2 * latent_dim, emb)
        self.g = _mlp(chunk_dim, emb)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def state(self, z: torch.Tensor, zg: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.f(torch.cat([z, zg], -1)), dim=-1)

    def action(self, chunk: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.g(chunk.flatten(-2)), dim=-1)              # [..., H, A] -> [..., E]

    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=100.0)

    def score(self, z: torch.Tensor, zg: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """Lower is better (the GatedSolver 'scorer' convention): z, zg [B, D]; cand [B, K, H, A] -> [B, K]."""
        s, a = self.state(z, zg), self.action(cand)                        # [B, E], [B, K, E]
        return -(a * s[:, None]).sum(-1)


class RetrievalPolicy(nn.Module):
    """System 1 as retrieval: the menu chunk with the highest verifier score (CLM's cached action menu).
    Same interface as ChunkPolicy.sample; candidate i = the i-th best menu chunk."""

    def __init__(self, verifier: ContrastiveVerifier, menu: torch.Tensor):
        super().__init__()
        self.v = verifier
        self.register_buffer("menu", menu.float())                         # [M, H, A]
        with torch.no_grad():
            self.register_buffer("menu_emb", verifier.action(self.menu))   # [M, E], computed once

    @torch.no_grad()
    def sample(self, z_t, z_goal, k, generator=None, extra=None, **kw):
        s = self.v.state(z_t.float(), z_goal.float()) @ self.menu_emb.T    # [B, M]
        idx = s.topk(k, dim=-1).indices                                      # [B, k]
        return self.menu[idx].to(z_t.dtype)


def retrieval_from_checkpoint(pk: dict) -> RetrievalPolicy:
    v = ContrastiveVerifier(pk["latent_dim"], pk["chunk_dim"])
    v.load_state_dict(pk["state_dict"])
    return RetrievalPolicy(v.eval(), pk["menu"]).eval()


def load_verifier(path) -> ContrastiveVerifier:
    pk = torch.load(path, map_location="cpu", weights_only=False)
    v = ContrastiveVerifier(pk["latent_dim"], pk["chunk_dim"])
    v.load_state_dict(pk["state_dict"])
    return v.eval()


def batch(win: OfflineWindows, n: int, g: torch.Generator):
    b = win.sample(n, g)
    k = torch.randint(1, H + 1, (n,), generator=g)                         # hindsight goal k blocks ahead
    z, zg = b["z"][:, 0], b["z"][torch.arange(n), k]
    return z, zg, b["action"]                                              # action [n, H, fs*A]


def info_nce(v: ContrastiveVerifier, z, zg, act, hard: int, g: torch.Generator) -> torch.Tensor:
    s, a = v.state(z, zg), v.action(act)
    logits = v.scale() * s @ a.T                                           # [n, n]
    lbl = torch.arange(len(z))
    if hard:
        sig = torch.exp(torch.empty(len(z), hard, 1, 1).uniform_(math.log(0.1), math.log(1.0), generator=g))
        neg = act[:, None] + sig * torch.randn((len(z), hard, *act.shape[1:]), generator=g)
        hn = v.scale() * (v.action(neg) * s[:, None]).sum(-1)              # [n, hard]
        row = torch.cat([logits, hn], 1)
    else:
        row = logits
    return 0.5 * (F.cross_entropy(row, lbl) + F.cross_entropy(logits.T, lbl))


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra, rb = a.argsort(1).argsort(1).astype(float), b.argsort(1).argsort(1).astype(float)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return np.where(den > 0, (ra * rb).sum(1) / np.where(den > 0, den, 1), np.nan)


@torch.no_grad()
def offline(v: ContrastiveVerifier, spec_name: str, seed: int) -> dict:
    """Same held-out set and metrics as critic.offline_eval: 1,500 evaluation states x 16 executed System 1
    candidates with real final cube-goal distances; plus top-1 retrieval of the true chunk among 1,024."""
    from critic import load_parts
    res = {}
    zev = torch.load(cache_path(spec_name, seed, "cube_eval"), map_location="cpu", weights_only=False)
    win = OfflineWindows(zev)
    g = torch.Generator().manual_seed(123)
    z, zg, act = batch(win, 1024, g)
    res["top1_true_chunk_of_1024"] = float(((v.state(z, zg) @ v.action(act).T).argmax(1) == torch.arange(1024))
                                           .float().mean())
    try:
        d = load_parts(spec_name, "cube_eval")
    except FileNotFoundError:
        return res
    pred = v.score(zev["z"][d["rows"]], zev["z"][d["goal_rows"]], d["cand"].float()).numpy()
    real = d["dist"].numpy()
    ok = real <= SUCCESS_M
    inf = ok.any(1) & ~ok.all(1)
    sp = spearman_rows(pred, real)
    n = len(real)
    res.update({"states": n, "informative_states": int(inf.sum()), "spearman_mean": float(np.nanmean(sp)),
                "spearman_informative": float(np.nanmean(sp[inf]))})
    for name, idx in (("mean_candidate", np.zeros(n, int)), ("verifier_pick", pred.argmin(1)),
                      ("oracle_pick", real.argmin(1))):
        res[f"success_{name}"] = float(ok[np.arange(n), idx].mean())
        res[f"success_{name}_informative"] = float(ok[np.arange(n), idx][inf].mean())
    return res


def train(spec: str, seed: int = 0, steps: int = 6000, bs: int = 1024, lr: float = 1e-3, hard: int = 0,
          menu_size: int = 20_000, out: str = "verifier_in.pt") -> dict:
    torch.manual_seed(seed)
    sp = Spec.parse(spec)
    cache = torch.load(cache_path(sp.name(), seed, "cube"), map_location="cpu", weights_only=False)
    win = OfflineWindows(cache)
    D, chunk_dim = cache["z"].shape[1], H * win.fs * cache["action"].shape[1]
    v = ContrastiveVerifier(D, chunk_dim)
    opt = torch.optim.AdamW(v.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    t0, log = time.time(), []
    for step in range(1, steps + 1):
        z, zg, act = batch(win, bs, g)
        loss = info_nce(v, z, zg, act, hard, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            log.append({"step": step, "loss": round(loss.item(), 4), "scale": round(v.scale().item(), 1),
                        "min": round((time.time() - t0) / 60, 2)})
            print(json.dumps(log[-1]), flush=True)
    v.eval()
    res = {"spec": sp.name(), "hard_negatives": hard, "steps": steps, "log": log, "offline": offline(v, sp.name(), seed)}
    print(json.dumps(res["offline"]), flush=True)
    gm = torch.Generator().manual_seed(seed + 1)
    rows = win.starts[torch.randperm(len(win.starts), generator=gm)[:menu_size]]
    ar = rows[:, None] + win.act[None]
    menu = cache["action"][ar].reshape(len(rows), H, -1)                  # expert chunks from training episodes
    ck = HOME / "checkpoints" / f"cube/{sp.name()}/seed{seed}"
    pk = {"state_dict": v.state_dict(), "latent_dim": D, "chunk_dim": chunk_dim, "meta": res}
    torch.save(pk, ck / out)
    torch.save({**pk, "policy_type": "retrieval", "menu": menu.half(), "tau_abs": torch.load(
        ck / "policy_gauss6k.pt", map_location="cpu", weights_only=False)["tau_abs"]},
        ck / out.replace(".pt", "_retrieval.pt"))
    (HOME / "results" / f"verifier__{sp.name()}__seed{seed}__{out.removesuffix('.pt')}.json").write_text(
        json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--hard", type=int, default=0, help="perturbed copies of the positive as extra negatives")
    ap.add_argument("--out", default="verifier_in.pt")
    a = ap.parse_args()
    train(a.spec, a.seed, a.steps, hard=a.hard, out=a.out)
