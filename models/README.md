# Trained model weights

Six Ultralytics YOLO11 detectors from the controlled six-way study. Single class
`0 = cetacean`, seed `1337`, input 1024 px. Full metrics: `../docs/MODEL_REGISTRY.md`.

| File | Backbone | Change from baseline | Paper name | Role |
|---|---|---|---|---|
| `B.pt`     | YOLO11n | baseline (porpoises in, 10% neg) | M1 | reference |
| `E0_s.pt`  | YOLO11s | ↑ capacity (same data as B) | M2 | **selected / deployed model** |
| `E1.pt`    | YOLO11n | + surface data | M3 | ablation (surface hurts) |
| `E2.pt`    | YOLO11n | − porpoise source | M4 | ablation (porpoises essential) |
| `E4_0.pt`  | YOLO11n | hard-negative dose 0% | M5 | ablation |
| `E4_20.pt` | YOLO11n | hard-negative dose 20% | M6 | ablation |

## Headline numbers (see registry for full tables)
- **`E0_s.pt` (M2)** is the deployed model: screening recall **0.535 @ 1 FPPI**,
  **0.611 @ 2 FPPI** on the out-of-domain HiDef survey (SAHI tile 1024).
- On the real 151 MP APEM frame, adaptive SAHI with `E0_s.pt` gets **3/3
  dolphins, 0 FP, 35 tiles**.

## Usage
```python
from ultralytics import YOLO
model = YOLO("models/E0_s.pt")          # plain inference
# scale-aware (recommended for survey frames):
#   python -m cetacean.inference.adaptive_sahi --weights models/E0_s.pt --image <frame>
```

These `.pt` files are committed to the repo (~46 MB total). The corresponding
training runs (`runs/train/<tag>/`) are not committed.
