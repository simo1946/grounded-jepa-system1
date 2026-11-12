"""Train System 1 on clean expert labels, optionally plus DAgger states.

Labels (``--labels``):
  sim    (default) clean oracle chunks simulated closed-loop from each window's state
         (gen_oracle_chunks.py / oracle_sim.py); open-loop replay of such chunks solves 50/50
         evaluation pairs (oracle_sim_check.py).
  block  the oracle action at each block's first state, held for the block. INVALID for a chunk
         policy (the oracle itself drops from 48/50 to 8/50 when its action is held per block);
         kept only to reproduce that failed first attempt.

Offline windows come from the latent cache (same rows as the dataset). For a window starting at
frame row r with a hindsight goal k blocks ahead (frame g = r + k*fs), block j's label is the clean
oracle action at the block's first state (row r + j*fs) toward the cube pose at g, repeated over
the block's fs steps. All H blocks are labelled (the oracle also says what to do after the goal
time, e.g. hold still). DAgger buffers (dagger.py) hold states the learner visited, labelled the
same way toward the commanded goal.

python oracle_policy.py --spec lewm+idm+sa --out policy_oracle6k.pt --steps 6000
python oracle_policy.py --spec lewm+idm+sa --out policy_dagger_r1.pt --init policy_oracle6k.pt \
    --steps 2000 --dagger swm_home/checkpoints/.../dagger/round1.pt --mix 0.5
"""
from __future__ import annotations

import argparse
import json
import time

import h5py
import numpy as np
import torch

from cached_policy import H, OfflineWindows, block_displacement_median
from hybrid_wm.control.latent_policy import ChunkPolicy, FlowChunkPolicy, policy_from_checkpoint
from latent_cache import cache_path
from models import HOME, Spec
from oracle import ORACLE_COLS, cube_oracle

torch.set_flush_denormal(True)


def action_scaler(cache: dict, data: str = "cube") -> tuple[np.ndarray, np.ndarray]:
    """(mean, std) with cache action = (raw - mean) / std, recovered exactly from the cache."""
    with h5py.File(HOME / "datasets" / f"{data}.h5", "r") as f:
        raw = f["action"][:]
    norm = cache["action"].numpy().astype(np.float64)
    ok = np.isfinite(raw).all(1) & np.isfinite(norm).all(1)
    raw, norm = raw[ok].astype(np.float64), norm[ok]
    std = raw.std(0) / norm.std(0)
    mean = (raw - norm * std).mean(0)
    err = np.abs((raw - mean) / std - norm).max()
    if err > 1e-3:
        raise RuntimeError(f"cache actions are not an affine z-score of the dataset actions (max err {err:.2e})")
    return mean, std


class OracleWindows:
    """Hindsight windows from the cache, labelled by the clean oracle (normalised action units)."""

    def __init__(self, cache: dict, data: str = "cube"):
        self.win = OfflineWindows(cache)
        self.fs, self.A = cache["frameskip"], cache["action"].shape[1]
        with h5py.File(HOME / "datasets" / f"{data}.h5", "r") as f:
            if len(f["action"]) != len(cache["z"]) or not np.array_equal(f["ep_offset"][:], cache["offsets"].numpy()):
                raise RuntimeError("dataset rows do not match the latent cache")
            self.S = {c: f[c][:] for c in ORACLE_COLS}
        self.mean, self.std = action_scaler(cache, data)
        self.z = cache["z"]

    def label(self, state_rows: np.ndarray, target_rows: np.ndarray) -> torch.Tensor:
        S = self.S
        raw = cube_oracle(S["proprio_effector_pos"][state_rows], S["proprio_effector_yaw"][state_rows],
                          S["proprio_gripper_contact"][state_rows], S["privileged_block_0_pos"][state_rows],
                          S["privileged_block_0_yaw"][state_rows], S["privileged_block_0_pos"][target_rows],
                          S["privileged_block_0_yaw"][target_rows])
        return torch.from_numpy(((raw - self.mean) / self.std).astype(np.float32))

    def sample(self, n: int, g: torch.Generator) -> dict:
        r = self.win.starts[torch.randint(len(self.win.starts), (n,), generator=g)].numpy()
        k = torch.randint(1, H + 1, (n,), generator=g).numpy()
        goal = r + k * self.fs
        states = r[:, None] + np.arange(H)[None] * self.fs                         # [n, H] block starts
        lab = self.label(states.reshape(-1), np.repeat(goal, H)).view(n, H, 1, self.A)
        act = lab.expand(n, H, self.fs, self.A).reshape(n, H, self.fs * self.A)
        return {"z_t": self.z[r], "z_g": self.z[goal], "action": act, "mask": torch.ones(n, H, dtype=torch.bool)}


class SimChunkWindows:
    """Hindsight windows with clean simulated oracle chunks (gen_oracle_chunks.py)."""

    def __init__(self, cache: dict, data: str = "cube"):
        parts = sorted((HOME / "datasets").glob(f"{data}_oracle_chunks_p*.pt"))
        if not parts:
            raise FileNotFoundError("no oracle chunk files; run gen_oracle_chunks.py")
        d = [torch.load(p_, map_location="cpu") for p_ in parts]
        self.r = torch.cat([x["rows"] for x in d]).long()
        self.g = torch.cat([x["goal_rows"] for x in d]).long()
        raw = torch.cat([x["actions_raw"] for x in d]).double().numpy()               # [N, H*fs, A]
        self.mean, self.std = action_scaler(cache, data)
        self.fs, self.A = cache["frameskip"], cache["action"].shape[1]
        norm = ((raw - self.mean) / self.std).astype(np.float32)
        self.act = torch.from_numpy(norm).view(len(self.r), H, self.fs * self.A)
        self.z = cache["z"]

    def sample(self, n: int, g: torch.Generator) -> dict:
        i = torch.randint(len(self.r), (n,), generator=g)
        return {"z_t": self.z[self.r[i]], "z_g": self.z[self.g[i]], "action": self.act[i],
                "mask": torch.ones(n, H, dtype=torch.bool)}


class DaggerBuffer:
    """Learner-visited block states (dagger.py) with oracle labels toward the commanded goal."""

    def __init__(self, episodes: list[dict], mean: np.ndarray, std: np.ndarray, fs: int):
        self.eps = [e for e in episodes if e["z"].shape[0] >= 1]
        self.mean, self.std, self.fs = mean, std, fs
        self.lab, self.chunked = [], all("chunk_raw" in e for e in self.eps)
        for e in self.eps:
            if self.chunked:                                  # simulated oracle chunks [F, H*fs, A] -> [F, H, fs*A]
                c = (e["chunk_raw"] - mean) / std
                self.lab.append(torch.from_numpy(c.astype(np.float32)).view(c.shape[0], H, -1))
                continue
            # label once (block-level, invalid for chunk policies; see module docstring): [F, A] normalised
            S, F = e["oracle_in"], e["z"].shape[0]
            raw = cube_oracle(S["proprio_effector_pos"], S["proprio_effector_yaw"], S["proprio_gripper_contact"],
                              S["privileged_block_0_pos"], S["privileged_block_0_yaw"],
                              np.repeat(e["goal_pos"][None], F, 0), np.repeat(e["goal_yaw"][None], F, 0))
            self.lab.append(torch.from_numpy(((raw - mean) / std).astype(np.float32)))

    def __len__(self):
        return len(self.eps)

    def sample(self, n: int, g: torch.Generator) -> dict:
        D = self.eps[0]["z"].shape[1]
        A = self.lab[0].shape[-1] // self.fs if self.chunked else self.lab[0].shape[1]
        z_t, z_g = torch.zeros(n, D), torch.zeros(n, D)
        act, mask = torch.zeros(n, H, self.fs * A), torch.zeros(n, H, dtype=torch.bool)
        for i, e in enumerate(torch.randint(len(self.eps), (n,), generator=g).tolist()):
            ep, lab = self.eps[e], self.lab[e]
            F = ep["z"].shape[0]
            t = int(torch.randint(F, (1,), generator=g))
            z_t[i], z_g[i] = ep["z"][t], ep["z_goal"]
            if self.chunked:                                  # the full simulated chunk from this state
                act[i], mask[i] = lab[t], True
                continue
            m = min(H, F - t)                                 # blocks whose start state was recorded
            act[i, :m] = lab[t:t + m].unsqueeze(1).expand(m, self.fs, A).reshape(m, -1)   # block label x fs steps
            mask[i, :m] = True
        return {"z_t": z_t, "z_g": z_g, "action": act, "mask": mask}


def train(spec_str: str, seed: int, out: str, steps: int, init: str | None = None, dagger: list[str] | None = None,
          mix: float = 0.5, policy_type: str = "gauss", batch: int = 256, lr: float = 1e-3,
          policy_seed: int = 0, labels: str = "sim") -> dict:
    spec = Spec.parse(spec_str)
    ckpt = HOME / "checkpoints" / f"cube/{spec.name()}/seed{seed}"
    cache = torch.load(cache_path(spec.name(), seed, "cube"), map_location="cpu")
    if labels not in ("sim", "block"):
        raise ValueError("labels must be 'sim' or 'block'")
    ow = SimChunkWindows(cache) if labels == "sim" else OracleWindows(cache)
    buf = DaggerBuffer([e for f in (dagger or []) for e in torch.load(f, map_location="cpu", weights_only=False)["episodes"]],
                       ow.mean, ow.std, ow.fs) if dagger else None
    n_on = int(round(batch * mix)) if buf is not None and len(buf) else 0
    torch.manual_seed(policy_seed)
    act_dim = ow.fs * ow.A
    if init:
        pk0 = torch.load(ckpt / init, map_location="cpu")
        pol = policy_from_checkpoint(pk0)
        policy_type = pk0.get("policy_type", "gauss")
    else:
        pol = (FlowChunkPolicy if policy_type == "flow" else ChunkPolicy)(cache["z"].shape[1], H, act_dim)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(policy_seed)
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        b = ow.sample(batch - n_on, g)
        if n_on:
            d = buf.sample(n_on, g)
            b = {k: torch.cat([b[k], d[k]]) for k in b}
        loss = pol.loss(b["z_t"], b["z_g"], b["action"], b["mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            log.append({"step": step, "loss": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 2)})
            print(json.dumps(log[-1]), flush=True)
    meta = {"state_dict": {k: v.cpu() for k, v in pol.state_dict().items()}, "latent_dim": cache["z"].shape[1],
            "policy_type": policy_type, "flow_steps": getattr(pol, "n_steps", None), "horizon": H,
            "action_dim": act_dim, "extra_dim": 0, "tau_abs": block_displacement_median(cache),
            "labels": f"oracle-{labels}", "trained": {"steps": steps, "init": init, "dagger": dagger or [], "mix": mix,
                                            "dagger_episodes": len(buf) if buf else 0, "batch": batch, "lr": lr,
                                            "policy_seed": policy_seed}, "log": log}
    torch.save(meta, ckpt / out)
    print(json.dumps({"saved": str(ckpt / out), "minutes": round((time.time() - t0) / 60, 2)}), flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--init", default=None)
    ap.add_argument("--dagger", nargs="*", default=None)
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--policy-type", choices=["gauss", "flow"], default="gauss")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--policy-seed", type=int, default=0)
    ap.add_argument("--labels", choices=["sim", "block"], default="sim")
    a = ap.parse_args()
    train(a.spec, a.seed, a.out, a.steps, a.init, a.dagger, a.mix, a.policy_type, a.batch, policy_seed=a.policy_seed,
          labels=a.labels)
