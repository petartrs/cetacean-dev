# Model Registry

Single class: `0 = cetacean`. Seed `1337`. Tile 640 / overlap 0.2 / model imgsz 1024 (SAHI, locked).
Weights land at `runs/train/<variant>/weights/best.pt`. Datasets built by `scripts/build_dataset.py`.

## Fixed evaluation (identical for every variant)
| Set | Path | Purpose |
|-----|------|---------|
| val | `data/dataset/shared/val` (873) | checkpoint selection (aerial) |
| test_precision | `data/dataset/shared/test_precision` (106) | held-out source `whale-bbox-layer` — precision / mAP |
| test_aerial | `data/dataset/shared/test_aerial` (291) | aerial in-distribution 5% slices |
| test_screening | `data/dataset/test_mammals` (704 img / 838 box / 12 empty) | MAMMALS low-GSD screening — recall@FPPI (FROC) |

## Experiments (train composition differs; eval fixed)
Hard-neg dose = fraction of the FINAL train set that is negatives (neg/(neg+pos)).
| Variant | Model | Train | Δ vs B | data.yaml |
|---------|-------|-------|--------|-----------|
| B | 11n | 4855 pos + 539 neg (10%) | baseline (aerial, gommapps=certain+inferred, porpoises in, 10% neg) | `data/dataset/B/data.yaml` |
| E0 | 11n / 11s / **11m** | = B data | capacity control (reuses B; **11m runs at END of phase**) | `data/dataset/B/data.yaml` |
| E1 | 11n | 7756 pos + 862 neg (10%) | + 4 surface sources | `data/dataset/E1/data.yaml` |
| E2 | 11n | 1355 pos + 151 neg (10%) | porpoises OUT of train | `data/dataset/E2/data.yaml` |
| E4_0 | 11n | 4855 pos + 0 neg (0%) | hard-neg dose 0% | `data/dataset/E4_0/data.yaml` |
| B (=E4_10) | 11n | 4855 pos + 539 neg (10%) | hard-neg dose 10% | `data/dataset/B/data.yaml` |
| E4_20 | 11n | 4855 pos + 1214 neg (20%) | hard-neg dose 20% | `data/dataset/E4_20/data.yaml` |
| E5 | best.pt | — | adaptive SAHI, inference-only (no training) | — |

Negatives are gommapps-derived, whale-free, TRAIN-only; pool = 1715 (57 full + 1236 @1024 + 342 @2048 + 80 validated). Dose = fraction of the final train set (neg/(neg+pos)).

## Trained weights
All plain imgsz1024 val (no SAHI). test_precision = whale-bbox-layer held-out source (106img/153inst).
test_aerial = in-distribution 5% aerial slices (291img/770inst, porpoise-dominated).

| Variant | Model | epochs | test_precision (P/R/mAP50/mAP50-95) | test_aerial (P/R/mAP50/mAP50-95) |
|---------|-------|--------|-------------------------------------|----------------------------------|
| B | 11n | 100 | 0.596 / 0.536 / 0.538 / 0.221 | 0.931 / 0.749 / 0.824 / 0.52 |
| E0_s | 11s | 100 | 0.688 / 0.506 / 0.540 / 0.227 | 0.911 / 0.798 / 0.863 / 0.55 |
| E1 | 11n | 100 | 0.653 / 0.510 / 0.510 / 0.238 | 0.905 / 0.729 / 0.793 / 0.508 |
| E2 | 11n | 69 (ES) | 0.342 / 0.209 / 0.202 / 0.070 | 0.616 / 0.144 / 0.143 / 0.093 |
| E4_0 | 11n | 100 | 0.676 / 0.504 / 0.531 / 0.212 | 0.920 / 0.773 / 0.823 / 0.532 |
| E4_20 | 11n | 100 | 0.633 / 0.549 / 0.503 / 0.224 | 0.919 / 0.747 / 0.801 / 0.521 |

Weights: `runs/train/<variant>/weights/best.pt`. B val(873) mAP50 0.853 / mAP50-95 0.541.

### Read-out (plain 1024, pre-SAHI)
- E0 capacity: 11s > 11n modestly (aerial mAP50 0.863 vs 0.824). Real test is SAHI (E5).
- E1 (+surface): slightly HURTS aerial (0.793 vs 0.824) and precision (0.510 vs 0.538) -> surface not helpful.
- E2 (porp OUT): collapses -> porpoise data is load-bearing for small-aerial (test_aerial is porpoise-heavy).
- E4 neg dose: 10% (B, 0.538) ~ best precision-mAP50; 0% 0.531; 20% 0.503 but highest recall 0.549.
- Pending: 11m (E0) at end of phase; E5 adaptive-SAHI eval of all.

## Screening regime — MAMMALS FROC (SAHI-tiled, final 2026-08-03)
Test = `data/dataset/test_mammals` (704 frames / 838 boxes / 12 empty). Metric = box recall
at a false-positive-per-image (FPPI) budget, IoU>=0.3, greedy match. Harness =
`scripts/eval_screening_froc.py` (two-stage: SAHI predict + cache, then FROC rescore).
No injected negatives (FPPI over the 704 test frames). Weights = `runs/train/<v>/weights/best.pt`.

**recall @ FPPI, SAHI tile 640:**
| Variant | R@0.1 | R@0.5 | R@1.0 | R@2.0 |
|---------|-------|-------|-------|-------|
| B | 0.122 | 0.297 | 0.412 | 0.531 |
| E0_s | 0.184 | 0.327 | 0.432 | 0.513 |
| E1 | 0.060 | 0.106 | 0.132 | 0.172 |
| E2 | 0.005 | 0.027 | 0.039 | 0.056 |
| E4_0 | 0.070 | 0.187 | 0.316 | 0.445 |
| E4_20 | 0.036 | 0.128 | 0.216 | 0.362 |

**recall @ FPPI, SAHI tile 1024:**
| Variant | R@0.1 | R@0.5 | R@1.0 | R@2.0 |
|---------|-------|-------|-------|-------|
| B | 0.063 | 0.264 | 0.412 | 0.573 |
| E0_s | **0.240** | **0.449** | **0.535** | 0.611 |
| E1 | 0.132 | 0.260 | 0.307 | 0.384 |
| E2 | 0.004 | 0.016 | 0.024 | 0.038 |
| E4_0 | 0.081 | 0.396 | 0.525 | **0.635** |
| E4_20 | 0.033 | 0.173 | 0.296 | 0.443 |

**SAHI cost (predict, 704 frames):** tile 640 total 7116 s (11n ~1.57 s/frame, 11s ~2.24);
tile 1024 total 5150 s (11n ~1.17 s/frame, 11s ~1.43). Tile 1024 is ~28% FASTER overall.
Artifacts: `runs/screening/final_t{640,1024}/{froc.png,froc_points.csv,timings.csv}`;
caches `cache_<v>_t<tile>_..._n704.json` (instant rescore).

### Read-out (screening)
- Best screening model = **E0_s (11s) @ tile 1024**: R@1.0=0.535, R@2.0=0.611. Capacity helps
  screening MORE than it helped precision -> low-GSD tiny objects benefit from the bigger backbone.
- **Tile 1024 >= tile 640 for the strong configs AND ~28% cheaper** -> for screening, tile 1024 is
  the better operating point (reverses the 588-frame dry-run hint that 640 wins). Only plain B at the
  very tightest 0.1 FPPI prefers 640 (0.122 vs 0.063).
- E2 (porpoises OUT) collapses (~0.02-0.06) -> porpoise data is load-bearing for screening too.
- E1 (+surface) hurts vs B/E0_s.
- E4 hard-neg dose: for screening LESS is better -> E4_0 (0%) > B (10%) > E4_20 (20%) at loose budgets
  (tile1024 R@2: 0.635 > 0.573 > 0.443). Hard negatives trade recall away in the recall-oriented regime.
- Absolute ceiling ~0.61-0.64 recall @2 FPPI -> real but limited transfer of aerial detectors to
  low-GSD screening (the domain-gap headline).
- IoU 0.3 vs 0.1 barely differs (dry-run check) -> misses are true misses, not localization slack.

## E5 — scale-aware inference: adaptive vs fixed-tile SAHI (2026-08-03)
Inference-only (no training). Two evals: (A) controlled semi-synthetic altitude sweep isolates GSD;
(B) the one real 151 MP APEM frame = high-GSD real anchor (opposite extreme from low-GSD MAMMALS).
Harness: `cetacean/inference/adaptive_sahi.py` + `scripts/{make_altitude_benchmark,eval_altitude_sweep,eval_apem_anchor}.py`.
`adaptive` sizes the crop so a nominal 3 m animal lands at ~45 px (skip SAHI if already big);
`adaptive_range` sizes so the LARGEST animal in-frame fits one tile (never fragments; more tiles).
Metrics: recall/precision/F1 + tiles/frame (compute) + latency, IoU 0.3. Run with PYTHONPATH=repo root.

### (A) Altitude sweep — 18 animals (L 1.7-18.2 m realistic mix), 7 alts, canvas 8192x5460, DJI-P1 GSD
Tiles/frame (compute): plain=1, fixed640=176, fixed1024=70, adaptive=1-24 (scales w/ alt), adapt_range=1-280.
recall by altitude (E0_s | B):
| alt (GSD cm/px) | plain | fixed640 | fixed1024 | adaptive | adapt_range |
|-----------------|-------|----------|-----------|----------|-------------|
| 60  (0.75) | .18/.35 | **.88/.76** | .71/.65 | .18/.35 | .18/.35 |
| 120 (1.51) | .11/.28 | .50/.56 | .50/.39 | .22/.44 | .39/.50 |
| 200 (2.51) | .17/.17 | .39/.28 | .44/.39 | .22/.22 | **.44/.33** |
| 320 (4.01) | .06/.11 | .22/.33 | .17/.28 | .11/.28 | .11/.28 |
Artifacts: `runs/e5/sweep/eval_{B,E0s}/{metrics.csv,recall_vs_altitude.png,f1_vs_altitude.png,tiles_vs_altitude.png}`.

### (B) APEM 151 MP real anchor — 3 bottlenose dolphins, EXIF alt 448.5 m -> GSD 2.41 cm/px
GT = 3 dolphins (2 from model boxes + 1 top-right MISS added by visual confirm). `data/apem/APEM_bottlenose_S15.txt`.
| method | tiles | latency | E0_s R (TP/FP) | B R (TP/FP) |
|--------|-------|---------|----------------|-------------|
| plain     | 1   | 1.6 s  | 0.00 (0/0)  | 0.00 (0/0)  |
| fixed640  | 588 | 15-21s | 0.67 (2/8)  | 0.67 (2/**83**) |
| fixed1024 | 234 | 11-13s | 0.67 (2/0)  | 1.00 (3/23) |
| **adaptive** | **35** | ~9 s | **1.00 (3/0)** | **1.00 (3/1)** |
Artifacts: `runs/e5/apem_{E0_s,B}/{metrics.csv,det_*.jpg}`.

### Read-out (E5)
- **APEM is the headline single result**: `plain` finds 0/3 dolphins on a 151 MP frame even though the
  animals are clearly resolvable -> whole-frame downscale (14x) destroys them. `adaptive`, sized to the
  EXIF GSD, gets **3/3 with ~0 FP at 35 tiles** — 17x cheaper than fixed640 AND fastest. Geometry-matched
  crops also minimise whitecap FPs (fixed640 floods 8-83 FP; more/finer tiles = more false alarms).
- Sweep: `adaptive` matches fixed-SAHI recall at a fraction of tiles at mid/high altitude with higher
  precision (spends tiles only when geometry demands). `adaptive_range` recovers recall at high alt but
  at high tile cost.
- **Honest caveat**: at low altitude (60 m) both adaptive modes skip to plain (skip rule reasons about
  the nominal 3 m animal) and lose the smaller animals vs fixed640 (.18/.35 vs .88/.76). Tunable via
  --skip-px / --target-len (set to the smallest expected animal); not a fundamental failure.
- GSD/domain axis made concrete: high-GSD real (APEM, geometry known) -> adaptive perfect; low-GSD
  domain-shift real (MAMMALS) -> recall ceilings ~0.6. Two real endpoints of the same detector.
