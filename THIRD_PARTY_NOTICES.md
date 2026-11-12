# Third-party code and dependencies

## Code included in this repository

| File | Origin | Licence |
|---|---|---|
| `tests/third_party/mamba3_siso_ref.py` | Verbatim copy of the pure-PyTorch Mamba-3 SISO reference functions from [state-spaces/mamba](https://github.com/state-spaces/mamba), `tests/ops/triton/test_mamba3_siso.py`. Copyright (c) 2025 Dao AI Lab, Goombalab. Used only as a test oracle. | Apache-2.0 (`LICENSES/Apache-2.0.txt`) |
| `src/hybrid_wm/layers/mamba3.py` (reference backend) | A port of the same reference equations (`mamba3_siso_fwd_ref`, `mamba3_siso_step_ref`). | Apache-2.0 for the ported math (`LICENSES/Apache-2.0.txt`) |
| `bench/lewm_suite/gcidm.py` | Independent re-implementation, from the paper's text (Sec. 4.2, App. F), of GC-IDM in "Latent Geometry Beyond Search" (arXiv 2605.08732), written without the authors' code. No code copied. | MIT (this repository) |
| `bench/lewm_suite/tasks.py::pickup_candidates` | Re-implementation of the grasp-centred Cube windows of [H-JEPA](https://github.com/kevinghst/H-JEPA) (`h_jepa/scripts/generate_dataset_eval_trajs.py`). | MIT (original) |

## Dependencies (installed separately, not redistributed)

| Package | Use | Licence |
|---|---|---|
| [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) 0.1.1 | LeWM model, environments, data collection, CEM planner, evaluation | MIT |
| [le-wm](https://github.com/lucas-maes/le-wm) | LeWM evaluation protocol and hyperparameters (`config/eval`), followed, not copied | MIT |
| [GC-IDM authors' code](https://github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models), commit 48c45b1 | The official GC-IDM (`train_idm.py`), cloned separately and called unmodified by `bench/lewm_suite/official_gcidm.py`; not copied or redistributed | CC BY-NC 4.0 |
| [OGBench](https://github.com/seohongpark/ogbench) 1.2.1 | Cube manipulation environment | MIT |
| stable-pretraining 0.1.8 | ViT encoder construction | MIT |
| MuJoCo 3.15 | physics simulator | Apache-2.0 |
| PyTorch, gymnasium, NumPy, SciPy, h5py | numerics | BSD / MIT |
| mamba-ssm (optional, GPU) | official Mamba-3 kernels | Apache-2.0 |

All other code in this repository is by the author and released under the MIT licence (`LICENSE`).
