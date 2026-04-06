# Interpretable RGB-D Mixture-of-Experts for Embedded Robotic Perception

A deterministic RGB-D Mixture-of-Experts (MoE) perception system that fuses a TensorRT-accelerated YOLOv8 segmentation expert with a geometry-driven depth expert through an interpretable gating mechanism. Designed for robust perception under challenging conditions on embedded robotic platforms.

---

## Overview

This system combines two complementary perception experts:

- **RGB Expert**: YOLOv8 instance segmentation (TensorRT)
- **Depth Expert**: geometry-based confidence maps using plane fitting
- **Bi-Directional Context Gate**: deterministic per-instance fusion

The system improves robustness in scenarios where RGB-only perception fails (darkness, glare, motion blur, clutter, occlusion) while maintaining interpretable decision signals.

---

## Repository Structure

```text
.
├── src/
│   ├── rgb_exp/                 # RGB expert (YOLOv8 TensorRT)
│   ├── depth_exp/               # Depth expert (geometry + confidence)
│   └── gate/                    # Fusion + pipeline
│       ├── run_moe.py
│       ├── gate.py
│       ├── broker.py
│       ├── adapter.py
│       └── watcher.py
│
├── docs/
│   ├── rgb_depth_collab.png
│   ├── bag5_darkness.png
│   └── sys_arch.pdf
│
├── README.md
├── LICENSE
└── requirements.txt
```

---

## System Architecture

See:
- `docs/sys_arch.pdf` → full system diagram  
- `docs/rgb_depth_collab.png` → RGB-depth fusion example  
- `docs/bag5_darkness.png` → failure/recovery case  

---

## Core Components

### RGB Expert (`src/rgb_exp/`)
- YOLOv8 segmentation (TensorRT)
- Outputs masks, classes, confidences

### Depth Expert (`src/depth_exp/`)
- RealSense depth processing
- RANSAC plane fitting
- Depth confidence maps

### Broker + Adapter (`src/gate/`)
- Timestamp synchronization
- RGB/depth alignment and remapping

### Gate (`src/gate/gate.py`)
- Deterministic fusion logic
- Per-instance routing:
  - RGB-led
  - Depth-assisted
  - Depth-proposed

---

## Requirements

- Python 3.10+
- CUDA + TensorRT
- Intel RealSense SDK

### Python Packages

- numpy
- opencv-python
- torch
- ultralytics
- cupy
- open3d
- pyrealsense2

### Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Live Inference (RealSense)

```bash
MOE_RS_ALIGN=0 MOE_FAST_PLANE_EVERY=1 MOE_SCALE_PLANE=0.50 MOE_RANSAC_ITERS=50 MOE_RANSAC_POINTS=500 MOE_CONF_EMA_ALPHA=0.0 MOE_VIZ_SHOW_PROPS=1 MOE_PUB_DEPTH=1 MOE_DEPTH_SPLIT_PROPS=1 MOE_TOPK=8 MOE_MAX_DET=3 MOE_FAST_MAX_INST=3 MOE_PROP_DECIMATE=3 MOE_PROP_BUDGET_MS=10.0 MOE_POLISH_MASKS=1 MOE_DEPTH_MAX_AGE_MS=140 MOE_FAST_NMS=1 MOE_BATCH_MASKS=1 MOE_VECTOR_STATS=1 MOE_JETSON_MODE=1 MOE_YOLO_SKIP=2 MOE_GATE_VIZ=1 MOE_YOLO_DECODE_RES=128 MOE_CONF_DOWNSAMPLE=192 MOE_DS=128 MOE_GATE_LOG_DETAIL=1 MOE_VIZ_SCALE=0.5 MOE_PROF=1 MOE_DEBUG=1 python3 src/gate/run_moe.py   --depth-module depth_exp.depth_expert   --yolo-module rgb_exp.yolo_trt_stream   --engine src/rgb_exp/best.engine   --hold-last   --tol-frac 0.9   --tol-ceiling 0.120   --tol-floor 0.010
```

---

### Bag Replay

```bash
MOE_RS_ALIGN=0 MOE_FAST_PLANE_EVERY=1 MOE_SCALE_PLANE=0.50 MOE_RANSAC_ITERS=50 MOE_RANSAC_POINTS=500 MOE_CONF_EMA_ALPHA=0.0 MOE_VIZ_SHOW_PROPS=1 MOE_PUB_DEPTH=1 MOE_DEPTH_SPLIT_PROPS=1 MOE_TOPK=8 MOE_MAX_DET=8 MOE_FAST_MAX_INST=8 MOE_PROP_DECIMATE=3 MOE_PROP_BUDGET_MS=20.0 MOE_YOLO_DECODE_RES=128 MOE_POLISH_MASKS=1 MOE_DEPTH_MAX_AGE_MS=140 MOE_YOLO_MAX_AGE_MS=140 MOE_FAST_NMS=1 MOE_BATCH_MASKS=1 MOE_VECTOR_STATS=1 MOE_JETSON_MODE=1 MOE_YOLO_SKIP=2 MOE_GATE_VIZ=0 MOE_CONF_DOWNSAMPLE=192 MOE_DS=128 MOE_GATE_LOG_DETAIL=1 MOE_VIZ_SCALE=0.5 MOE_PROF=1 MOE_DEBUG=1 MOE_REPLAY_BAG=/path/to/your/bag.bag python3 src/gate/run_moe.py   --depth-module depth_exp.depth_expert   --yolo-module rgb_exp.yolo_trt_stream   --engine src/rgb_exp/best.engine   --hold-last   --tol-frac 0.9   --tol-ceiling 0.120   --tol-floor 0.010
```

---

## Important Notes

- TensorRT `.engine` files are not included  
- `.bag` files are not included  
- Update all local paths accordingly  
- Optimized for Jetson devices, but works on desktop GPUs  

---

## Outputs

- Instance segmentation masks  
- Class predictions  
- Bounding boxes  
- Fusion mode (RGB-led / depth-assisted / depth-proposed)  
- Depth confidence maps  
- Plane estimation signals  
- Visualization overlays (optional)  
- Profiling logs  

---

## License

MIT License
