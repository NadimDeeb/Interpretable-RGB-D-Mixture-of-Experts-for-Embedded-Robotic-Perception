# Environment Variable Guide

This guide documents the main runtime environment variables exposed by the MoE perception stack. It focuses on variables that materially affect throughput, synchronization, fusion behavior, recovery aggressiveness, replay behavior, or visualization. Purely internal constants and stale exploratory flags are intentionally omitted.

## How to use

Set variables inline before launching the system:

```bash
MOE_RS_ALIGN=0 MOE_SCALE_PLANE=0.50 MOE_MAX_DET=3 MOE_YOLO_SKIP=2 MOE_DS=128 \
MOE_PROP_DECIMATE=3 MOE_PROP_BUDGET_MS=10.0 MOE_FAST_NMS=1 MOE_BATCH_MASKS=1 \
MOE_VECTOR_STATS=1 MOE_GATE_VIZ=1 MOE_GATE_LOG_DETAIL=1 \
python3 gate/run_moe.py --depth-module depth_exp.depth_expert --yolo-module rgb_exp.yolo_trt_stream --engine rgb_exp/best.engine --hold-last
```

## Recommended repository layout

Place this file at the repository root as `MOE_ENV_GUIDE.md`, then add a short section in `README.md` linking to it. This keeps the main README readable while still exposing the full tuning surface.

---

## 1. Global debug and replay controls

| Variable | Type | Default | Typical values | Effect |
|---|---|---:|---|---|
| `MOE_DEBUG` | bool | `0` | `0`, `1` | Enables verbose debug prints across modules. Useful for diagnosis, but adds console overhead. |
| `MOE_PROF` | bool | `0` | `0`, `1` | Enables runtime profiling prints such as YOLO timing, broker timing, and gate timing. Useful for benchmarking support logs. |
| `MOE_REPLAY_BAG` | path/string | empty | absolute path to `.bag` | Enables RealSense bag replay mode instead of live capture. Also changes some runtime behavior such as YOLO-ready signaling and queue handling. |

**Recommendation:** keep `MOE_DEBUG=0` for timing measurements and only enable `MOE_PROF=1` when collecting profiler logs.

---

## 2. YOLO expert controls

| Variable | Type | Default | Typical values | Effect |
|---|---|---:|---|---|
| `MOE_CONF_THR` | float | `0.45` | `0.35`–`0.55` | YOLO confidence threshold. Lower values increase recall but can increase false positives and fusion load. |
| `MOE_IOU_THR` | float | `0.5` | `0.45`–`0.60` | YOLO NMS IoU threshold. Higher values retain more overlapping detections. |
| `MOE_MAX_DET` | int | `8` | `3`, `5`, `8` | Maximum detections returned by YOLO per frame. Lower values reduce downstream gate cost. |
| `MOE_YOLO_SKIP` | int | `0` | `0`, `1`, `2` | Skips full YOLO inference on intermediate frames and re-emits the last valid detections. Major performance lever. |

**Main tuning role:** `MOE_YOLO_SKIP` and `MOE_MAX_DET` directly affect total system load. `MOE_CONF_THR` changes how permissive the RGB expert is before fusion. |

---

## 3. Depth expert controls

| Variable | Type | Default | Typical values | Effect |
|---|---|---:|---|---|
| `MOE_RS_ALIGN` | bool/int | `1` | `0`, `1` | Enables RealSense alignment. Turning it off favors your custom reprojection path and usually improves speed. |
| `MOE_MAX_RANSAC_POINTS` | int | `1000` | `500`–`1000` | Caps the number of candidate points used by plane fitting. Lower values reduce plane-fit cost. |
| `MOE_DEPTH_BUDGET_MS` | float | `22.0` | `10`–`25` | Target work budget for the depth side. Useful when trying to keep depth processing bounded. |
| `MOE_PLANE_ITERS` | int | `120` | `50`–`120` | Initial/default RANSAC iteration count for support-plane fitting. Lower values are faster but can reduce plane stability. |
| `MOE_PLANE_MIN_INLIER` | float | `0.05` | `0.03`–`0.10` | Minimum plane inlier ratio accepted as valid for downstream use. |
| `MOE_SCALE_PLANE` | float | `0.15` | `0.15`–`0.50` | Relative scale used for the small plane-fitting workspace. Larger values preserve more detail but cost more. |
| `MOE_CONF_DOWNSAMPLE` | int | `192` | `128`, `160`, `192` | Downsampled grid resolution used in confidence-map processing. Larger values preserve finer geometry but add cost. |
| `MOE_CONF_EMA` | float | `0.35` | `0.0`–`0.5` | Temporal EMA on the depth confidence map. Larger values smooth more strongly. |
| `MOE_CONF_FP16` | bool | `0` | `0`, `1` | Stores confidence maps in FP16 instead of FP32 to reduce bandwidth and memory pressure. |
| `MOE_CONF_HOLD_FRAMES` | int | `3` | `2`–`5` | Reuses the last valid depth confidence map for a short window when the current map is degenerate. Helps avoid flicker. |
| `MOE_PLANE_MAX_AGE_MS` | float | `200.0` | `140`–`200` | Maximum age of a cached plane fit before it is considered stale. |
| `MOE_PLANE_BG_EVERY` | int | `1` | `1`, `2`, `3` | Plane refit cadence in frames for the async plane worker. |

**Main tuning role:** `MOE_RS_ALIGN`, `MOE_PLANE_ITERS`, `MOE_SCALE_PLANE`, and `MOE_CONF_DOWNSAMPLE` are the most important depth-side performance knobs. `MOE_CONF_EMA`, `MOE_CONF_HOLD_FRAMES`, and `MOE_PLANE_MIN_INLIER` primarily affect stability and recovery behavior.

---

## 4. Broker and synchronization controls

| Variable | Type | Default | Typical values | Effect |
|---|---|---:|---|---|
| `MOE_DEPTH_MAX_AGE_MS` | float | `150` | `100`–`150` | Maximum acceptable age for depth outputs during pairing. Lower values make synchronization stricter. |
| `MOE_YOLO_MAX_AGE_MS` | float | `150` | `100`–`150` | Maximum acceptable age for YOLO outputs during pairing. |
| `MOE_BROKER_RGB` | bool | `1` | `0`, `1` | Controls whether the broker tries to maintain RGB frames for gate visualization / inspection. |

**CLI flags related to synchronization:** `--hold-last`, `--tol-frac`, `--tol-floor`, and `--tol-ceiling` in `run_moe.py` control temporal pairing behavior and should be documented alongside the environment variables in the README examples.

---

## 5. Gate fusion and proposal controls

| Variable | Type | Default | Typical values | Effect |
|---|---|---:|---|---|
| `MOE_FUSE_THR` | float | `0.40` | `0.35`–`0.50` | Default fusion threshold used by the gate. Lower values make fusion more permissive. |
| `MOE_FUSE_THR_MAP` | JSON string | empty | e.g. `'{"0":0.45,"*":0.40}'` | Optional per-class threshold override map. |
| `MOE_DS` | int | `128` | `128`, `160`, `192` | Gate downsample grid size. One of the most important speed-versus-detail controls for proposals and mask reasoning. |
| `MOE_PROP_DECIMATE` | int | `6` | `3`, `4`, `6` | Runs depth proposals only once every N frames when YOLO is healthy. Lower values increase recovery responsiveness but raise cost. |
| `MOE_PROP_BUDGET_MS` | float | `12.0` | `10.0`, `12.0`, `20.0` | Soft time budget for depth proposal generation within the gate. |
| `MOE_RGB_STRONG` | float | `0.60` | `0.55`–`0.70` | RGB confidence level treated as strong by the gate. |
| `MOE_RGB_ONLY_IF_THIN` | bool/int | `1` | `0`, `1` | Restricts certain RGB-only behavior to thin objects. |
| `MOE_WEAK_DEPTH_MEAN` | float | `0.28` | `0.20`–`0.35` | Mean depth support threshold used in weak-depth decisions. |
| `MOE_WEAK_DEPTH_PIX` | float | `0.22` | `0.15`–`0.30` | Pixel-level weak-depth threshold used by the gate. |
| `MOE_RESCUE_CAP_DS` | int | `150` | `100`–`200` | Cap used during rescue logic on the downsampled grid. |
| `MOE_YOLO_EDGE_THR` | float | `0.30` | `0.25`–`0.40` | Threshold used when the gate uses YOLO mask probabilities to constrain final mask extent. |
| `MOE_SOFT_GATE_ALPHA` | float | `0.7` | `0.5`–`0.8` | Mixing factor for soft gate behavior. |
| `MOE_CONF_MIN_MEAN` | float | `0.005` | `0.003`–`0.01` | Minimum mean confidence before a depth map is considered degenerate at gate level. |
| `MOE_CONF_MIN_MAX` | float | `0.02` | `0.01`–`0.05` | Minimum peak confidence before a depth map is considered degenerate at gate level. |
| `MOE_GATE_HOLD_FRAMES` | int | `10` | `5`–`10` | Number of frames the gate may hold the last non-empty output to suppress flicker. |
| `MOE_INST_HOLD_FRAMES` | int | `3` | `2`–`5` | Short per-instance hold window when YOLO still sees an object but fusion drops it. |
| `MOE_INST_HOLD_IOU` | float | `0.5` | `0.3`–`0.6` | IoU threshold for matching held instances across frames. |

**Main tuning role:** `MOE_DS`, `MOE_PROP_DECIMATE`, `MOE_PROP_BUDGET_MS`, and `MOE_FUSE_THR` are the core gate tuning knobs for speed and recovery aggressiveness.

---

## 6. Temporal repair, persistence, and hysteresis controls

| Variable | Type | Default | Typical values | Effect |
|---|---|---:|---|---|
| `MOE_REPAIR_LATCH` | bool | `0` | `0`, `1` | Enables repair latch logic for short-term mask persistence. |
| `MOE_REPAIR_LATCH_TTL` | int | `12` | `8`–`12` | Lifetime of repair latch entries in frames. |
| `MOE_REPAIR_LATCH_MINPIX` | int | `60` | `40`–`100` | Minimum pixel count for retained repair regions. |
| `MOE_REPAIR_LATCH_IOU` | float | `0.15` | `0.10`–`0.25` | IoU used to match repair-latch entries. |
| `MOE_STICKY_MIN_DEPTH` | float | `0.15` | `0.10`–`0.20` | Minimum depth-assisted fraction before sticky depth expansion is retained. |
| `MOE_STICKY_MAX_EXPAND` | float | `1.5` | `1.2`–`1.8` | Maximum allowed area growth when reusing previous depth-expanded mask shapes. |
| `MOE_APPROVED_TTL` | int | `12` | `8`–`12` | Lifetime of approved depth additions in the gate’s memory. |

**Main tuning role:** these parameters affect temporal stability and are most useful when reducing flicker or keeping depth-assisted masks from collapsing too aggressively.

---

## 7. Proposal-to-instance merge controls

| Variable | Type | Default | Typical values | Effect |
|---|---|---:|---|---|
| `MOE_PROP_MERGE` | bool | `1` | `0`, `1` | Enables merging depth proposals into existing instances instead of creating separate `cls=-1` outputs. |
| `MOE_PROP_MERGE_T_TOUCH` | float | `0.05` | `0.03`–`0.10` | Minimum touch-based association score. |
| `MOE_PROP_MERGE_T_OVERLAP` | float | `0.08` | `0.05`–`0.12` | Minimum overlap-based association score. |
| `MOE_PROP_MERGE_T_IOU` | float | `0.02` | `0.01`–`0.05` | Minimum IoU-based association score. |
| `MOE_PROP_MERGE_T_ACCEPT` | float | `0.12` | `0.10`–`0.20` | Acceptance threshold for merging a proposal into an instance. |
| `MOE_PROP_MERGE_MARGIN` | float | `0.05` | `0.03`–`0.10` | Required best-vs-second-best margin to avoid ambiguous merges. |
| `MOE_PROP_MERGE_MAX_EXPAND` | float | `1.50` | `1.2`–`1.8` | Limits how much a proposal may expand an instance. |
| `MOE_PROP_MERGE_W_OVERLAP` | float | `0.45` | `0.2`–`0.5` | Weight of overlap score in merge decision. |
| `MOE_PROP_MERGE_W_TOUCH` | float | `0.45` | `0.2`–`0.5` | Weight of touch score in merge decision. |
| `MOE_PROP_MERGE_W_IOU` | float | `0.10` | `0.05`–`0.2` | Weight of IoU score in merge decision. |
| `MOE_PROP_MERGE_TOUCH_K` | int | `5` | `3`, `5`, `7` | Morphology kernel size used in the touch test. |
| `MOE_PROP_MERGE_DBG` | bool | `0` | `0`, `1` | Enables verbose diagnostics for proposal merge rejection or acceptance. |

**Main tuning role:** these are advanced controls. Most users can keep the defaults unless proposal blobs are being split too aggressively into unknown objects or merged too readily into known instances.

---

## 8. Performance optimization switches

| Variable | Type | Default | Typical values | Effect |
|---|---|---:|---|---|
| `MOE_FAST_NMS` | bool | `1` | `0`, `1` | Enables faster NMS path in the gate. |
| `MOE_BATCH_MASKS` | bool | `1` | `0`, `1` | Enables batch mask processing in the gate. |
| `MOE_VECTOR_STATS` | bool | `1` | `0`, `1` | Enables vectorized mask statistics in the gate. |
| `MOE_JETSON_MODE` | bool | `0` | `0`, `1` | Applies Jetson-oriented runtime policy for proposals and internal execution. |
| `MOE_INST_WORKERS` | int | `4` | `2`–`4` | Number of parallel workers for per-instance fusion. |

**Main tuning role:** keep these enabled unless you are debugging an optimization issue or trying to compare against a slower reference path.

---

## 9. Visualization and log controls

| Variable | Type | Default | Typical values | Effect |
|---|---|---:|---|---|
| `MOE_GATE_VIZ` | bool | enabled unless set to `0` | `0`, `1` | Enables live gate visualization window. Disable for clean benchmarking. |
| `MOE_GATE_SAVE_DIR` | path/string | empty | output directory | Saves rendered gate frames to disk when visualization is active. |
| `MOE_GATE_LOG_DETAIL` | bool | `0` | `0`, `1` | Enables detailed per-instance gate log lines such as `p_rgb`, `p_fused`, `mean_depth_ds`, and support fractions. |
| `MOE_VIZ_SHOW_PROPS` | bool | `1` | `0`, `1` | Shows or hides depth-only proposals in visualization overlays. |

---

## Minimal recommended profiles

### Qualitative inspection

```bash
MOE_RS_ALIGN=0 \
MOE_SCALE_PLANE=0.50 \
MOE_PLANE_ITERS=50 \
MOE_MAX_DET=3 \
MOE_YOLO_SKIP=2 \
MOE_DS=128 \
MOE_PROP_DECIMATE=3 \
MOE_PROP_BUDGET_MS=10.0 \
MOE_DEPTH_MAX_AGE_MS=140 \
MOE_FAST_NMS=1 MOE_BATCH_MASKS=1 MOE_VECTOR_STATS=1 MOE_JETSON_MODE=1 \
MOE_GATE_VIZ=1 MOE_GATE_LOG_DETAIL=1 MOE_VIZ_SHOW_PROPS=1 \
MOE_PROF=1 MOE_DEBUG=1 \
python3 gate/run_moe.py --depth-module depth_exp.depth_expert --yolo-module rgb_exp.yolo_trt_stream --engine rgb_exp/best.engine --hold-last --tol-frac 0.9 --tol-ceiling 0.120 --tol-floor 0.010
```

### Benchmarking

```bash
MOE_RS_ALIGN=0 \
MOE_SCALE_PLANE=0.50 \
MOE_PLANE_ITERS=50 \
MOE_MAX_DET=3 \
MOE_YOLO_SKIP=2 \
MOE_DS=128 \
MOE_PROP_DECIMATE=3 \
MOE_PROP_BUDGET_MS=10.0 \
MOE_DEPTH_MAX_AGE_MS=140 \
MOE_REPLAY_BAG=/path/to/bag4_clutter.bag \
MOE_FAST_NMS=1 MOE_BATCH_MASKS=1 MOE_VECTOR_STATS=1 MOE_JETSON_MODE=1 \
MOE_GATE_VIZ=0 MOE_GATE_LOG_DETAIL=0 MOE_PROF=1 MOE_DEBUG=0 \
python3 gate/run_moe.py --depth-module depth_exp.depth_expert --yolo-module rgb_exp.yolo_trt_stream --engine rgb_exp/best.engine --hold-last --tol-frac 0.9 --tol-ceiling 0.120 --tol-floor 0.010
```

## README snippet

```md
## Runtime configuration

The MoE pipeline exposes a small set of environment variables for synchronization, depth confidence processing, proposal generation, and visualization. The most important tuning knobs are:

- `MOE_RS_ALIGN`: toggle RealSense alignment vs custom reprojection
- `MOE_PLANE_ITERS`: support-plane fitting cost / stability tradeoff
- `MOE_MAX_DET`: maximum YOLO detections per frame
- `MOE_YOLO_SKIP`: RGB inference decimation
- `MOE_DS`: gate downsample resolution
- `MOE_PROP_DECIMATE`: proposal cadence when YOLO is healthy
- `MOE_PROP_BUDGET_MS`: soft budget for proposal generation
- `MOE_FUSE_THR`: default gate fusion threshold
- `MOE_DEPTH_MAX_AGE_MS`, `MOE_YOLO_MAX_AGE_MS`: synchronization age limits
- `MOE_GATE_VIZ`, `MOE_GATE_LOG_DETAIL`: visualization and log verbosity

A full variable reference, including types, defaults, and recommended ranges, is provided in [`MOE_ENV_GUIDE.md`](./MOE_ENV_GUIDE.md).
```

## Variables intentionally not emphasized here

Some flags from older command notes do not appear in the current uploaded implementation as active `os.getenv(...)` controls, including names such as `MOE_FAST_PLANE_EVERY`, `MOE_RANSAC_POINTS`, `MOE_CONF_EMA_ALPHA`, `MOE_PUB_DEPTH`, `MOE_TOPK`, `MOE_FAST_MAX_INST`, `MOE_POLISH_MASKS`, `MOE_YOLO_DECODE_RES`, and `MOE_DEPTH_SPLIT_PROPS`. Those should not be documented as first-class current controls unless they are reintroduced into the live codebase.
