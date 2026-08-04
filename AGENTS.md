# AGENTS.md — onboarding for AI coding agents

This file tells an autonomous coding agent how to work in this repository. Read
it fully before editing. Human-facing overview is in `README.md`.

## What this repo is
RGB aerial cetacean (whale/dolphin/porpoise) detection for SurveyLabs' AEROSUB
project. Full research codebase: data curation, training, evaluation, scale-aware
inference (adaptive SAHI), a ROS 2 node, and the LaTeX paper. It was staged from
the working tree at `/home/dell/cetacean-detection-final`.

## Ground truth for results
`docs/MODEL_REGISTRY.md` is authoritative for every model, dataset split and
metric. Do not restate numbers from memory — read the registry. The paper
(`paper/main.tex`) must stay consistent with it.

## Environment (must do first)
- Python **3.10**. Interpreter: `./ml-venv/bin/python` (recreate with
  `python3.10 -m venv ml-venv && ./ml-venv/bin/pip install -r requirements.txt
  -r requirements-dev.txt`). Torch is CUDA 13.0 (`--index-url .../whl/cu130`).
- **Always** `export PYTHONPATH=$(pwd)` before running any script — the
  `cetacean` package is NOT pip-installed and scripts import it.
- GPU is `cuda:0`.
- Label Studio has its OWN isolated venv `labelstudio-venv`
  (`requirements-labelstudio.txt`); never install it into `ml-venv`.
- Shell: run `set +H` before commands containing `!` (history expansion bites
  weights/globs).

## Known gotchas (verified)
- **scipy is broken in `ml-venv`** (NumPy 2 ABI mismatch). Never `import scipy`;
  use OpenCV / NumPy / pure Python. Existing scripts already avoid it.
- The paper builds with **tectonic**, not pdflatex: `cd paper && ./build.sh`.
  The tectonic binary is not committed; install it or run `tectonic main.tex`.
  Verify a build: `pdfinfo main.pdf | grep Pages` and
  `pdftotext main.pdf - | grep -c '??'` must be 0 (no undefined refs).
- Several scripts and `cetacean/inference/adaptive_sahi.py` contain **absolute
  `/home/dell/cetacean-detection-final/...` default paths** (e.g.
  `DEFAULT_WEIGHTS`). They are overridable via CLI flags (`--weights`, etc.).
  Either pass explicit paths or clone to the same location.
- `data/` (57 GB) and `runs/` are NOT in the repo. Scripts that read
  `data/dataset/...` or `runs/train/...` need those materialised first (rebuild
  the dataset + retrain), OR use the shipped `models/*.pt` for inference-only work.

## Model tags <-> paper names
Code/runs/registry use experiment tags; the paper uses M1–M6:
`B=M1(11n base)`, `E0_s=M2(11s, best)`, `E1=M3(+surface)`, `E2=M4(−porpoise)`,
`E4_0=M5(neg0%)`, `E4_20=M6(neg20%)`. `E3` was dropped (never in paper).
Weights are shipped in `models/<tag>.pt`; training also writes
`runs/train/<tag>/weights/best.pt`.

## Key commands
```bash
export PYTHONPATH=$(pwd)
PY=./ml-venv/bin/python
$PY scripts/build_dataset.py                 # (re)build data/dataset from normalised sources
bash scripts/train_queue.sh                  # train the six variants
$PY scripts/eval_screening_froc.py --help    # low-GSD screening FROC (recall@FPPI)
$PY scripts/eval_altitude_sweep.py --help    # adaptive vs fixed SAHI (synthetic sweep)
$PY scripts/eval_apem_anchor.py --help       # real 151 MP high-GSD anchor
$PY -m cetacean.inference.adaptive_sahi --help
```

## Conventions
- Single detection class: `0 = cetacean`. Seed `1337`. SAHI tile 640 / overlap
  0.2 / model imgsz 1024 (locked defaults).
- Evaluation matching: IoU >= 0.3 for screening/FROC; standard mAP for exact
  detection. Screening caches (`runs/screening/.../cache_*_n704.json`) let you
  rescore/replot instantly without re-predicting.
- Figures for the paper live in `paper/figures/`; regenerate plots from cached
  CSVs (e.g. `runs/screening/final_t1024/froc_points.csv`) rather than
  re-running inference.

## Open workstreams (where to pick up)
1. **Close the domain gap** — run more pseudo-labelling rounds on hard survey
   frames, retrain, re-evaluate the screening FROC (expect the ~0.6 ceiling to rise).
2. **Deployment hardening** — Jetson Orin + TensorRT export of the selected
   model (`models/E0_s.pt`); see the companion `cetacean-detector-ros2` repo.
3. **Geometry-based size estimation** — invert GSD to turn a detection's pixel
   length into a physical length on nadir frames.
4. **View-conditional augmentation** — separate schedules for nadir vs oblique.

Do NOT: reintroduce IR/multispectral/thermal framing into the paper (removed by
request); rename `models/`/`runs/` tags in code (paper-only rename to M1–M6).
