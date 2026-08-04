# Aerial Cetacean Detection — Developer Repository

RGB perception for **SurveyLabs Ltd.** under the HORIZON Europe project
**AEROSUB** (Grant Agreement 101189723). This repo is the full research and
development codebase: the data pipeline, the trained detectors, the evaluation
harnesses, the scale-aware inference tooling, the ROS 2 node source, and the
LaTeX paper.

> If you just want to *run the detector on a robot/aircraft*, use the separate
> deployment repo **cetacean-detector-ros2** instead. This repo is for
> understanding, reproducing and extending the work.

---

## 1. What this project is, in plain language

We detect whales, dolphins and porpoises in aerial images of open water so that
offshore-wind wildlife surveys can be done faster and, eventually, on-board an
autonomous aircraft.

The single most important idea in the whole project is the **GSD axis**
(ground-sample distance = how many metres of sea each pixel covers, i.e. how big
an animal *looks* in the image). One detector has to work across a huge range of
apparent sizes, and the two ends of that range are genuinely **different jobs**:

| End of the axis | What you see | The task | How we score it |
|---|---|---|---|
| **High GSD** (low altitude / big sensor) | animal is hundreds of pixels | **exact detection** — find and count each animal | precision / mAP |
| **Low GSD** (high altitude / coarse sensor) | animal is a few faint pixels | **wide-area screening** — flag frames a human should review | recall at a fixed false-alarm budget (FROC) |

Two levers control performance:

1. **Data composition** — *what* the detector is trained on. A controlled
   six-way study shows the porpoise data is essential, extra "surface"
   (boat-deck) data hurts, fewer hard negatives help recall, and a bigger
   backbone (YOLO11s) lifts low-GSD screening recall ~30%.
2. **Scale-aware inference (adaptive SAHI)** — instead of one fixed crop size,
   we size the crop from the flight geometry (GSD) so an animal always lands at
   the size the model was trained on. On a real 151-megapixel survey frame this
   turns "plain inference finds nothing" into "all 3 dolphins found, 0 false
   positives, 17× fewer tiles."

The honest limit: on a completely unseen survey (different sensor/provider) the
low-GSD screening recall plateaus around **0.6 at 2 false positives/image** — a
*domain gap*, not a scale problem. The fix is more in-domain labelled data,
collected via the human-in-the-loop **pseudo-labelling loop** (Label Studio) and
used to retrain.

The full write-up is in [`paper/main.tex`](paper/main.tex).

---

## 2. Repository layout

```
cetacean/inference/adaptive_sahi.py   # the scale-aware inference core (also used by the ROS node)
scripts/                              # the whole pipeline: build data, train, evaluate, pseudo-label
configs/                              # tracker / misc configs
docs/MODEL_REGISTRY.md                # authoritative results table for every model + eval
docs/raw_datasets.md                  # notes on the raw source datasets
models/                              # the six trained weights (B, E0_s, E1, E2, E4_0, E4_20)
paper/                               # LaTeX source of the report (build with tectonic; see below)
ros2_ws/src/cetacean_detector/       # ROS 2 node source (canonical copy lives in the deployment repo)
plan/plan.md                         # the original project plan / thesis
data/                                # NOT in git — 57 GB; see data/README.md to obtain
runs/                                # NOT in git — training/eval outputs; regenerate
```

Big things deliberately excluded from git: `data/` (57 GB of imagery),
`runs/` (training runs + eval caches), and the Python virtual environments.
See [`data/README.md`](data/README.md) and Section 4 below.

---

## 3. Environment setup

Python **3.10**. Two virtual environments are used on the working machine:

```bash
# 1) ML / inference / training env  (GPU)
python3.10 -m venv ml-venv
./ml-venv/bin/pip install -r requirements.txt        # torch cu130, ultralytics, sahi, opencv, numpy2 ...
./ml-venv/bin/pip install -r requirements-dev.txt

# 2) Label Studio env (ISOLATED — do not mix with ml-venv)
python3.10 -m venv labelstudio-venv
./labelstudio-venv/bin/pip install -r requirements-labelstudio.txt
```

- Torch is installed from the CUDA 13.0 wheel index (`--index-url .../whl/cu130`);
  a single NVIDIA GPU (`cuda:0`) is assumed.
- **Almost every script must be run with the repo root on `PYTHONPATH`** because
  the `cetacean` package is not pip-installed:

  ```bash
  export PYTHONPATH=$(pwd)
  ./ml-venv/bin/python scripts/eval_screening_froc.py --help
  ```

- **Known gotcha:** `scipy` is broken in `ml-venv` (NumPy 2 ABI mismatch). Do
  not `import scipy`; use OpenCV / pure-Python instead (the scripts already do).

---

## 4. Data

The imagery is **not** in this repo (57 GB). Every training source is public;
`data/README.md` lists each one with its download link (also in Table 1 of the
paper). The two held-out survey "anchors" (HiDef, APEM) are proprietary vendor
frames available on request.

After download, `scripts/build_dataset.py` normalises everything into a single
deduplicated, EXIF-corrected, single-class (`0 = cetacean`) YOLO dataset of
**27,290 images / 55,163 boxes** under `data/dataset/`.

---

## 5. Typical workflows

All commands assume `export PYTHONPATH=$(pwd)` and the `ml-venv` interpreter.

```bash
# Build / rebuild the training dataset from normalised sources
./ml-venv/bin/python scripts/build_dataset.py

# Train the six controlled variants (see scripts/train_queue.sh)
bash scripts/train_queue.sh

# High-GSD exact-detection eval (precision / mAP on the held-out source)
./ml-venv/bin/python scripts/eval_mammals_sahi.py --help    # SAHI-tiled eval helpers

# Low-GSD screening eval (FROC, recall@FPPI) — the headline screening number
./ml-venv/bin/python scripts/eval_screening_froc.py \
    --models B=runs/train/B/weights/best.pt E0_s=runs/train/E0_s/weights/best.pt ... \
    --images data/dataset/test_mammals/images --labels data/dataset/test_mammals/labels \
    --out runs/screening/final_t1024 --tile 1024

# Scale-aware inference experiments
./ml-venv/bin/python scripts/make_altitude_benchmark.py     # build the synthetic sweep
./ml-venv/bin/python scripts/eval_altitude_sweep.py --help  # adaptive vs fixed SAHI
./ml-venv/bin/python scripts/eval_apem_anchor.py --help     # the real 151 MP frame

# Pseudo-labelling loop (grow the dataset)
./labelstudio-venv/bin/label-studio start --port 8080       # 1) label/correct
./ml-venv/bin/python scripts/predict_to_labelstudio.py      # 2) pre-annotate
./ml-venv/bin/python scripts/ls_export_to_normalized.py     # 3) fold back in
```

Adaptive SAHI on any image (the reusable core):

```bash
./ml-venv/bin/python -m cetacean.inference.adaptive_sahi --help   # or import in Python
```

---

## 6. Models and results

Six controlled variants (one factor changed at a time from baseline `B`). Full
numbers are in [`docs/MODEL_REGISTRY.md`](docs/MODEL_REGISTRY.md).

| Weight file (`models/`) | Backbone | Change from baseline | Paper name |
|---|---|---|---|
| `B.pt`     | YOLO11n | baseline (porpoises in, 10% neg) | **M1** |
| `E0_s.pt`  | YOLO11s | ↑ capacity — **selected/best model** | **M2** |
| `E1.pt`    | YOLO11n | + surface data (hurts) | **M3** |
| `E2.pt`    | YOLO11n | − porpoise source (collapses) | **M4** |
| `E4_0.pt`  | YOLO11n | hard-neg dose 0% | **M5** |
| `E4_20.pt` | YOLO11n | hard-neg dose 20% | **M6** |

> **Naming note:** the code, run directories and this registry use the original
> experiment tags (`B, E0_s, ...`). The paper renames them `M1–M6` for
> readability. The mapping above is the only thing you need to connect the two.

Headline results:
- **Screening (low GSD):** best model `E0_s`/M2 reaches **0.535 recall @ 1 FPPI**
  and **0.611 @ 2 FPPI** on the out-of-domain HiDef survey (704 frames).
- **Exact detection (high GSD):** adaptive SAHI recovers **3/3 dolphins, 0 FP,
  17× fewer tiles** on the real 151 MP APEM frame, where plain inference sees
  nothing.

---

## 7. The paper

```bash
cd paper && ./build.sh        # uses tectonic, NOT pdflatex → main.pdf
```

The bundled `tectonic` binary is **not** committed (it's large). Install
tectonic (`cargo install tectonic` or a release binary) and point `build.sh` at
it, or run `tectonic main.tex` directly.

---

## 8. The ROS 2 node

`ros2_ws/src/cetacean_detector/` is a copy of the deployment node for reference.
The **canonical, self-contained** version for building on the aircraft lives in
the companion repo **cetacean-detector-ros2** (it vendors
`cetacean/inference/adaptive_sahi.py` so it needs nothing from this repo).

---

## 9. Where to pick up the work

See [`AGENTS.md`](AGENTS.md) for a machine-readable onboarding (build/test
commands, conventions, gotchas) and [`plan/plan.md`](plan/plan.md) for the
original plan. Open workstreams are listed at the end of `AGENTS.md`.
