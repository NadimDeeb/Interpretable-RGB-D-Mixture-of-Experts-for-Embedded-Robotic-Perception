#!/usr/bin/env python3
"""
Bi-Directional Context Gate (BCG) for MoE (YOLOv8-Seg + Depth Expert)
----------------------------------------------------------------------
• Lets either expert 'lead' per instance based on reliability.
• Adds depth-led proposals to recover thin/missed tools.
• Lightweight: ~<1 ms/frame on Jetson class devices.

FIXED: More aggressive depth-led detection when YOLO fails.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import os, time, json
import numpy as np
import cv2
from collections import deque
import concurrent.futures

# Keep types in sync with broker
from broker import YoloInst, GateInput, GateHealth
from adapter import compute_letterbox, depth_to_yolo_img

MOE_DEBUG = os.getenv("MOE_DEBUG", "0") not in ("0", "", "false", "False")

# ----------------- Tunables (ADJUSTED for better depth-led) -----------------
FUSE_THR_DEFAULT   = 0.40      # LOWERED from 0.50 - accept lower confidence fusions
DEPTH_THR_PROPOSE  = 0.22      # LOWERED from 0.40 - more aggressive depth proposals
DEPTH_MIN_MEAN     = 0.08      # LOWERED from 0.45 - accept lower mean confidence
YOLO_MASK_BIN_THR  = 0.50

MOE_PROF = int(os.getenv("MOE_PROF", "0"))

# Area filters
MIN_AREA           = 32
MAX_AREA           = 640*640 // 2

# Morphology
CLOSE_K            = 3
OPEN_K             = 0

# Reliability weights
OVERLAP_PENALTY    = 0.0
VAR_DAMPING_EPS    = 1e-6

# Thinness heuristic
THIN_TAU           = 0.24
THIN_BONUS         = 0.15
THIN_PX_AT_DS = 2

# Depth proposals vs YOLO overlap gating
MAX_IOU_MERGE      = 0.50
MAX_OVERLAP_FRAC   = 0.35

# Visualization
VIZ_EVERY_DEFAULT  = 3

# Instance suppression / proposals
NMS_IOU_THR        = 0.60
MAX_PROPOSALS      = 3
PROPOSAL_IOU_SKIP  = 0.55
PROPOSAL_OVERLAP_FRAC = 0.45

# Per-class thresholds
CLASS_THR: Dict[int, float] = {}
DEFAULT_FUSE_THR = float(os.getenv("MOE_FUSE_THR", "0.40"))  # LOWERED from 0.50

# --- Performance toggles ---
DS_DEFAULT = int(os.getenv("MOE_DS", "128"))   # 128 by default
PROP_DECIMATE_N = int(os.getenv("MOE_PROP_DECIMATE", "6"))  # run proposals 1/N frames when YOLO healthy (was 10 default)
PROP_BUDGET_MS  = float(os.getenv("MOE_PROP_BUDGET_MS", "12.0"))  # tighter default (was 8 default)

_thr_env = os.getenv("MOE_FUSE_THR_MAP", "")
THR_MAP_FALLBACK = {}
if _thr_env:
    try:
        THR_MAP_FALLBACK = json.loads(_thr_env)
    except Exception:
        THR_MAP_FALLBACK = {}

# ----------------- Output dataclasses -----------------
@dataclass
class FusedInst:
    mask: np.ndarray
    cls: int
    score_rgb: float
    score_fused: float
    area: int
    bbox_xyxy: Tuple[int,int,int,int]
    depth_support: float
    mode: str
    mean_prob_ds: float = 0.0
    mean_depth_ds: float = 0.0
    assist_rgb_frac: float = 0.0
    assist_depth_frac: float = 0.0

@dataclass
class GateOutput:
    t: float
    fused: List[FusedInst]
    health: GateHealth
    fuse_thr_info: Dict[int, float]


# ----------------- Gate implementation -----------------
class Gate:
    def __init__(self,
                 fuse_thr: float = FUSE_THR_DEFAULT,
                 class_thr: Optional[Dict[int, float]] = None,
                 close_k: int = CLOSE_K,
                 open_k: int = OPEN_K,
                 min_area: int = MIN_AREA,
                 visualize: Optional[bool] = None,
                 window: str = "MoE BCG",
                 save_dir: Optional[str] = None):
        
        self._log_detail = os.getenv("MOE_GATE_LOG_DETAIL", "0") not in ("0","","false","False")
        
        # Fusion policy knobs
        self._RGB_STRONG = float(os.getenv("MOE_RGB_STRONG", "0.60"))
        self._RGB_ONLY_IF_THIN = int(os.getenv("MOE_RGB_ONLY_IF_THIN", "1"))
        self._WEAK_DEPTH_MEAN = float(os.getenv("MOE_WEAK_DEPTH_MEAN", "0.28"))
        self._WEAK_DEPTH_PIX  = float(os.getenv("MOE_WEAK_DEPTH_PIX",  "0.22"))
        self._RESCUE_CAP_DS   = int(os.getenv("MOE_RESCUE_CAP_DS", "150"))
        
        self.fuse_thr = float(fuse_thr)
        self.class_thr = class_thr or {}
        self.min_area = int(min_area)
        self.min_area_ds = max(1, self.min_area // 16)

        # Viz controls
        env_viz = os.environ.get("MOE_GATE_VIZ", "").strip()
        self.visualize = bool(visualize) if visualize is not None else (env_viz != "0")
        self.win = str(window)
        self.save_dir = save_dir or os.environ.get("MOE_GATE_SAVE_DIR", "").strip() or None
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.viz_every = 1

        self._frame_idx = 0
        self._win_ready = False
        self._tbuf = deque(maxlen=60)
        self._last_fps = 0.0

        # How strict we are when using YOLO to gate the final 640x640 mask.
        # This can be lower than the DS binarization threshold so depth can
        # "fill in" low-probability regions near the object.
        self._YOLO_EDGE_THR = float(os.getenv("MOE_YOLO_EDGE_THR", "0.30"))

        # YOLO health (EMA)
        self._ema_alpha = 0.20
        self._yolo_area_ema = 0.0
        self._yolo_conf_ema = 0.0
        self._dead_streak = 0

        self._alpha = float(os.getenv("MOE_SOFT_GATE_ALPHA", "0.7"))

        # Simple temporal hold for flicker suppression
        self._last_out: Optional[GateOutput] = None
        self._last_out_age: int = 0
        self._max_hold_frames: int = int(os.getenv("MOE_GATE_HOLD_FRAMES", "10"))

        # Depth-conf robustness (temporal hold + relaxed validity)
        self._conf_min_mean: float = float(os.getenv("MOE_CONF_MIN_MEAN", "0.005"))
        self._conf_min_max: float  = float(os.getenv("MOE_CONF_MIN_MAX", "0.02"))
        self._conf_hold_frames: int = int(os.getenv("MOE_CONF_HOLD_FRAMES", "3"))
        self._conf_bad_streak: int = 0
        self._last_conf: Optional[np.ndarray] = None
        self._last_conf_stats: Tuple[float, float] = (0.0, 0.0)

        # Per-instance short-term hysteresis (to avoid flicker)
        self._inst_cache: List[FusedInst] = []
        self._inst_cache_age: List[int] = []

        # ---------------- Repair latch (per-instance) ----------------
        self._REPAIR_LATCH_ENABLE = os.getenv("MOE_REPAIR_LATCH", "0") not in ("0", "", "false", "False")
        self._REPAIR_LATCH_TTL    = int(os.getenv("MOE_REPAIR_LATCH_TTL", "12"))   # frames
        self._REPAIR_LATCH_MINPIX = int(os.getenv("MOE_REPAIR_LATCH_MINPIX", "60"))
        self._REPAIR_LATCH_IOU    = float(os.getenv("MOE_REPAIR_LATCH_IOU", "0.15"))

        # Each entry: {cls:int, bbox:(x1,y1,x2,y2), mask_u8:(640,640) uint8, ttl:int}
        self._repair_cache = []

        # How long to hold a dropped instance (in frames) as long as YOLO still sees it
        self._inst_hold_max: int = int(os.getenv("MOE_INST_HOLD_FRAMES", "3"))
        # IOU needed to consider "same instance" between frames (YOLO box vs fused box)
        self._inst_hold_iou: float = float(os.getenv("MOE_INST_HOLD_IOU", "0.5"))

        # Persistent depth additions memory (per class id)
        self._depth_memory = {}  # key: (cls, cx//32, cy//32) -> uint8 vote count map (0-255)
        self._approved_additions: dict = {}  # key: mem_key -> {'mask': np.ndarray, 'ttl': int}
        self._clean_additions_buf = np.zeros((640, 640), dtype=bool)
        self._mem_clean_buf = np.zeros((640, 640), dtype=bool)

        # Sticky depth-assisted mask hysteresis:
        # if last frame had a larger depth-assisted fused mask for the same tool,
        # don't let YOLO immediately carve it back down.
        self._sticky_min_depth_frac: float = float(
            os.getenv("MOE_STICKY_MIN_DEPTH", "0.15")
        )
        # Max area growth allowed when we reuse previous depth-expanded shape
        self._sticky_max_expand: float = float(
            os.getenv("MOE_STICKY_MAX_EXPAND", "1.5")
        )

        # Letterbox used to map YOLO masks into the 640x640 canvas
        # Here we assume YOLO operates in a 640x640 square space.
        self._lb = compute_letterbox((640, 480), (640, 640))

        # Kernels at full resolution
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)) if close_k >= 2 else None
        self._k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)) if open_k  >= 2 else None
        self._k7      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        self._k3      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        # Downsampled size (faster proposals on DS grid; adjustable via MOE_DS / MOE_JETSON_MODE)
        # Keep this equal to DS_DEFAULT so we preserve mask fidelity (thin handles, screwdrivers, etc.)
        self.DS_W = DS_DEFAULT
        self.DS_H = DS_DEFAULT

        # Per-instance proposal knobs (so we don't re-read env every frame)
        self._PROP_DECIMATE_N = PROP_DECIMATE_N
        self._PROP_BUDGET_MS  = PROP_BUDGET_MS
        self._max_proposals   = MAX_PROPOSALS

        # Selective optimization based on hardware (Jetson mode)
        if os.getenv("MOE_JETSON_MODE", "0") != "0":
            # On Jetson, keep DS resolution for fidelity but make proposals a bit cheaper,
            # NOT almost disabled.
            # - ensure we don't run proposals too often
            # - but allow a reasonable per-frame budget and up to 3 proposals
            self._PROP_DECIMATE_N = self._PROP_DECIMATE_N
            self._PROP_BUDGET_MS  = self._PROP_BUDGET_MS
            self._max_proposals   = self._max_proposals

        # DS-scale morphology kernels
        self._k_ds3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        self._k_ds7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

        # Reusable DS buffers (avoid per-frame allocs)
        self._ds_conf      = np.zeros((self.DS_H, self.DS_W), np.float32)
        self._ds_conf_thr  = np.zeros((self.DS_H, self.DS_W), np.uint8)
        self._ds_union     = np.zeros((self.DS_H, self.DS_W), np.uint8)
        self._ds_covered   = np.zeros((self.DS_H, self.DS_W), np.uint8)
        self._ds_prop_mask = np.zeros((self.DS_H, self.DS_W), np.uint8)
        self._ds_tmp_u8    = np.zeros((self.DS_H, self.DS_W), np.uint8)
        self._ds_tmp_f32   = np.zeros((self.DS_H, self.DS_W), np.float32)
        self._tmp_ds       = np.zeros((self.DS_H, self.DS_W), np.float32)

        # Reusable full-res buffers
        self._acc_mask640  = np.zeros((640, 640), np.float32)
        self._tmp_mask640  = np.zeros((640, 640), np.float32)
        self._dil_buf      = np.empty((640, 640), dtype=np.uint8)
        self._ring_buf     = np.empty((640, 640), dtype=np.uint8)
        self._core_buf     = np.empty((640, 640), dtype=np.uint8)
        self._support_buf  = np.empty((640, 640), dtype=np.uint8)
        self._depth_add_buf = np.zeros((640, 640), dtype=bool)
        self._comp_full_buf  = np.zeros((640, 640), dtype=bool)   # reused per CC, avoids per-frame malloc

        # Cache env-read so it isn't called inside the hot loop every frame
        self._APPROVED_TTL = int(os.getenv("MOE_APPROVED_TTL", "12"))

        # Single-pass kernel equivalent to k7 dilated 10 times (~35px fringe).
        # One morphology pass is ~10x faster than 10 passes of a small kernel.
        self._k_fringe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))

        # Precompute common thresholds and sizes
        self._ds_area           = self.DS_W * self.DS_H
        self._min_area_ds_float = float(self.min_area_ds)
        self._ds_shape          = (self.DS_H, self.DS_W)

        # Precompute border masks once
        self._border_mask_ds = np.ones(self._ds_shape, np.uint8)
        self._border_mask_ds[:4, :]  = 0
        self._border_mask_ds[-4:, :] = 0
        self._border_mask_ds[:, :4]  = 0
        self._border_mask_ds[:, -4:] = 0

        self._frame_idx = 0

        # Precomputed constants
        self._THIN_PX_AT_DS_FLOAT = float(THIN_PX_AT_DS)
        self._RESCUE_CAP_DS_FLOAT = float(self._RESCUE_CAP_DS)

        # Configuration-driven optimizations
        self._USE_FAST_NMS         = os.getenv("MOE_FAST_NMS", "1") != "0"
        self._BATCH_MASK_PROCESSING = os.getenv("MOE_BATCH_MASKS", "1") != "0"
        self._USE_VECTORIZED_STATS  = os.getenv("MOE_VECTOR_STATS", "1") != "0"

        # ---------------- Proposal → Instance merge policy (NEW) ----------------
        # When a depth proposal blob is likely part of an existing tool instance,
        # merge it into that instance mask instead of creating a separate cls=-1 blob.
        self._PROP_MERGE_ENABLE = os.getenv("MOE_PROP_MERGE", "1") not in ("0", "", "false", "False")

        # Association thresholds (tunable)
        self._PROP_MERGE_T_TOUCH   = float(os.getenv("MOE_PROP_MERGE_T_TOUCH",   "0.05"))
        self._PROP_MERGE_T_OVERLAP = float(os.getenv("MOE_PROP_MERGE_T_OVERLAP", "0.08"))
        self._PROP_MERGE_T_IOU     = float(os.getenv("MOE_PROP_MERGE_T_IOU",     "0.02"))
        self._PROP_MERGE_T_ACCEPT  = float(os.getenv("MOE_PROP_MERGE_T_ACCEPT",  "0.12"))
        self._PROP_MERGE_MARGIN    = float(os.getenv("MOE_PROP_MERGE_MARGIN",    "0.05"))  # best - second_best
        self._PROP_MERGE_MAX_EXPAND = float(os.getenv("MOE_PROP_MERGE_MAX_EXPAND", "1.50"))  # proposal_area <= 0.60*inst_area

        # Score weights for association
        self._PROP_MERGE_W_OVERLAP = float(os.getenv("MOE_PROP_MERGE_W_OVERLAP", "0.45"))
        self._PROP_MERGE_W_TOUCH   = float(os.getenv("MOE_PROP_MERGE_W_TOUCH",   "0.45"))
        self._PROP_MERGE_W_IOU     = float(os.getenv("MOE_PROP_MERGE_W_IOU",     "0.10"))

        # Morphology kernel for "touch" (adjacency) test — larger neighborhood reduces “almost-touch” misses
        k_touch_sz = int(os.getenv("MOE_PROP_MERGE_TOUCH_K", "5"))
        k_touch_sz = max(3, k_touch_sz | 1)  # force odd >=3
        self._k_touch = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_touch_sz, k_touch_sz))

        # Optional debug for why merges fail
        self._PROP_MERGE_DBG = os.getenv("MOE_PROP_MERGE_DBG", "0") not in ("0", "", "false", "False")

        # Debug counters for proposals (printed under MOE_DEBUG)
        self._prop_dbg_total = 0
        self._prop_dbg_merged = 0
        self._prop_dbg_new = 0

        _N_WORKERS = int(os.getenv("MOE_INST_WORKERS", "4"))
        self._inst_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=_N_WORKERS, thread_name_prefix="gate_inst"
        )
        self._inst_bufs = [self._make_inst_bufs() for _ in range(_N_WORKERS)] 

    # -------------- Public entry --------------
    def process_cb(self, gi: GateInput):
        out = self.process(gi)

        # --- Temporal hold to suppress flicker ---
        if len(out.fused) == 0:
            # If we have no instances this frame but YOLO still has detections,
            # reuse the last non-empty output for a few frames.
            if (
                self._last_out is not None
                and self._last_out_age < self._max_hold_frames
                and len(gi.yolo_instances) > 0
            ):
                # Reuse last fused instances but with the current timestamp/health
                reused = GateOutput(
                    t=out.t,
                    fused=self._last_out.fused,
                    health=out.health,
                    fuse_thr_info=self._last_out.fuse_thr_info,
                )
                out = reused
                self._last_out_age += 1
            else:
                self._last_out = None
                self._last_out_age = 0
        else:
            # Normal update: store the latest non-empty output
            self._last_out = out
            self._last_out_age = 0
        # --- end temporal hold ---

        modes = {"rgb": sum(1 for f in out.fused if f.mode == "rgb-led"),
                 "dep": sum(1 for f in out.fused if f.mode == "depth-led")}
        if MOE_DEBUG:
            print(f"[fuse] t={out.t:.3f} hw_ts={gi.hw_ts:.3f} | n={len(out.fused)} (rgb={modes['rgb']}, dep={modes['dep']}) "
                f"| plane_ok={out.health.plane_ok} r={out.health.inlier_ratio:.2f} | thr={out.fuse_thr_info}")
        
        if MOE_DEBUG and self._log_detail:
            for k,f in enumerate(out.fused):
                print("[fuse/inst] "
                    f"frame={self._frame_idx} #{k} cls={f.cls} mode={f.mode} "
                    f"#{k} cls={f.cls} mode={f.mode} "
                    f"p_rgb={f.score_rgb:.2f} p_fused={f.score_fused:.2f} "
                    f"mean_prob_ds={f.mean_prob_ds:.2f} mean_depth_ds={f.mean_depth_ds:.2f} "
                    f"assist_rgb={f.assist_rgb_frac:.2f} assist_depth={f.assist_depth_frac:.2f} "
                    f"area={f.area} box={f.bbox_xyxy}")
            
            if self._REPAIR_LATCH_ENABLE and len(self._repair_cache) > 32:
                self._repair_cache = self._repair_cache[-32:]

        self._frame_idx += 1
        if self.visualize:
            try:
                if not self._win_ready:
                    cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
                    self._win_ready = True

                viz_every = getattr(self, "_viz_every", 1)
                do_render = (self._frame_idx % max(1, viz_every) == 0)

                if do_render:
                    view = self._render_view(gi, out)
                    scale = getattr(self, "_viz_scale", 1.0)
                    if scale != 1.0:
                        h, w = view.shape[:2]
                        view = cv2.resize(view, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    cv2.imshow(self.win, view)
                    if self.save_dir:
                        p = os.path.join(self.save_dir, f"gate_{self._frame_idx:06d}.jpg")
                        cv2.imwrite(p, view)

                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    raise KeyboardInterrupt

            except Exception as e:
                if MOE_DEBUG:
                    print(f"[gate/viz] skipped frame: {e}")

    # -------------- Core processing --------------
    def process(self, gi: GateInput) -> GateOutput:
        t0_all = time.perf_counter()

        # Depth confidence sanity check – if it's bad, we’ll fall back to RGB-only,
        # but with a short temporal hold so one bad frame does not kill depth.
        depth_conf = gi.conf_yolo.astype(np.float32, copy=False)

        # Basic stats
        mean_conf = float(np.nanmean(depth_conf))
        max_conf  = float(np.nanmax(depth_conf))

        # A frame is considered *truly* bad only if:
        #   - it contains NaNs, OR
        #   - both max and mean are extremely small (near-flat zero map).
        depth_bad_now = (
            np.isnan(depth_conf).any()
            or (max_conf < self._conf_min_max and mean_conf < self._conf_min_mean)
        )

        depth_bad = depth_bad_now
        conf_used = depth_conf

        if depth_bad_now:
            # If we recently had a good depth map, reuse it for a few frames
            # instead of instantly dropping depth; this avoids "all blue" flicker.
            if self._last_conf is not None and self._conf_bad_streak < self._conf_hold_frames:
                depth_bad = False
                conf_used = self._last_conf
                self._conf_bad_streak += 1
                if MOE_DEBUG and MOE_PROF:
                    print("[gate/hold] Reusing last valid depth_conf for this frame")
            else:
                self._conf_bad_streak += 1
                if MOE_DEBUG and MOE_PROF:
                    print("[gate/skip] Invalid depth confidence – using RGB-only this frame")
        else:
            # Good depth -> reset streak and remember this conf for possible reuse
            self._conf_bad_streak = 0
            if self._last_conf is None or self._last_conf.shape != depth_conf.shape:
                self._last_conf = np.empty_like(depth_conf)
            np.copyto(self._last_conf, depth_conf)
            self._last_conf_stats = (mean_conf, max_conf)

        # DEBUG OUTPUT - shows what we're receiving
        if MOE_DEBUG:
            print(f"[gate/debug] plane_ok={gi.health.plane_ok} inlier_ratio={gi.health.inlier_ratio:.3f}")
            print(f"[gate/debug] depth_conf stats: mean={mean_conf:.3f} max={max_conf:.3f}")
            print(f"[gate/debug] yolo_instances={len(gi.yolo_instances)}")

        alpha = self._alpha

        # If depth is bad, treat conf as zeros so fusion becomes RGB-only
        if depth_bad:
            conf = np.zeros_like(depth_conf, dtype=np.float32)
        else:
            conf = conf_used

        # Pre-compute HSV once for the whole frame — used by appearance filter
        # Downsampled to 128x128: histograms need colour stats, not spatial detail
        if gi.bgr_yolo is not None:
            _bgr_ds = cv2.resize(gi.bgr_yolo, (128, 128), interpolation=cv2.INTER_AREA)
            self._hsv_cache = cv2.cvtColor(_bgr_ds, cv2.COLOR_BGR2HSV)
            self._hsv_ds_scale = 256.0 / 640.0
        else:
            self._hsv_cache = None
            self._hsv_ds_scale = 1.0

        assert conf.shape == (640, 640), f"depth conf must be 640x640, got {conf.shape}"

        # --- Build DS once (reuse buffers) ---
        cv2.resize(conf, (self.DS_W, self.DS_H), dst=self._ds_conf, interpolation=cv2.INTER_AREA)
        np.greater(self._ds_conf, DEPTH_THR_PROPOSE, out=self._ds_conf_thr)  # bool in u8
        self._ds_conf_thr = self._ds_conf_thr.astype(np.uint8, copy=False)

        # knock out DS borders to prevent shimmer
        # Use precomputed border mask
        cv2.bitwise_and(self._ds_conf_thr, self._border_mask_ds, dst=self._ds_conf_thr)

        # clear per-frame unions
        self._ds_union.fill(0)
        self._ds_covered.fill(0)

        # Pre-compute once per frame (used inside the rgb-led instance loop)
        depth_edge_thr = getattr(self, "_DEPTH_EDGE_THR", 0.90)
        # Blur at DS scale (~100× fewer pixels), upsample back
        conf_smooth     = cv2.GaussianBlur(conf, (3, 3), 0)
        depth_strong    = (conf_smooth >= depth_edge_thr)
        depth_strong_hi = (conf_smooth >= 0.90)

        thr_info: Dict[int, float] = {}
        fused: List[FusedInst] = []

        # Build YOLO masks at DS
        if self._BATCH_MASK_PROCESSING:
            probs, bins, self._ds_union = self._process_yolo_masks_batch(gi)
        else:
            # Original fallback
            probs: List[np.ndarray] = []
            bins:  List[np.ndarray] = []
            for inst in gi.yolo_instances:
                m = inst.mask_prob.astype(np.float32)
                if m.max() > 1.0:
                    m = np.clip(inst.mask_prob.astype(np.float32), 0.0, 1.0)
                m = np.clip(m, 0.0, 1.0, out=m)

                # Direct resize into DS grid
                cv2.resize(m, (self.DS_W, self.DS_H), dst=self._ds_tmp_f32, interpolation=cv2.INTER_AREA)
                probs.append(self._ds_tmp_f32.copy())
                np.greater_equal(self._ds_tmp_f32, YOLO_MASK_BIN_THR, out=self._ds_tmp_u8)
                bins.append(self._ds_tmp_u8.copy())
                cv2.bitwise_or(self._ds_union, self._ds_tmp_u8, dst=self._ds_union)
        
        # YOLO health tracking
        union_px_ds = int(self._ds_union.sum())
        union_frac  = union_px_ds / float(self.DS_W * self.DS_H)
        mean_score  = (float(np.mean([inst.score for inst in gi.yolo_instances]))
                       if gi.yolo_instances else 0.0)

        a = self._ema_alpha
        self._yolo_area_ema = (1.0 - a) * self._yolo_area_ema + a * union_frac
        self._yolo_conf_ema = (1.0 - a) * self._yolo_conf_ema + a * mean_score

        yolo_reliable = (self._yolo_area_ema > 0.020) and (self._yolo_conf_ema > 0.40)
        # Adaptive cap: keep depth cheap when YOLO is healthy
        yolo_weak = (self._yolo_area_ema < 0.015) or (self._yolo_conf_ema < 0.35)
        adaptive_max_props = min(self._max_proposals,
                                 3 if (yolo_weak or len(gi.yolo_instances) == 0) else 2)
        if yolo_reliable:
            self._dead_streak = 0
        else:
            self._dead_streak += 1
        
        t1_prep = time.perf_counter()
        if MOE_PROF:
            print(f"[gate/timer] prep={(t1_prep - t0_all)*1000:.1f}ms")

        # Per-instance fusion
        if self._USE_VECTORIZED_STATS:
            mask_stats = self._compute_mask_stats_batch(bins, self._ds_conf)
        else:
            mask_stats = []
            for m_bin in bins:
                area_ds = int(m_bin.sum())
                if area_ds == 0:
                    mask_stats.append((0, 0, 0, 0.0, False))
                    continue
                ys_ds, xs_ds = np.where(m_bin > 0)
                h_ds = int(ys_ds.max() - ys_ds.min() + 1) if ys_ds.size else 0
                w_ds = int(xs_ds.max() - xs_ds.min() + 1) if xs_ds.size else 0
                mask_pixels = self._ds_conf[m_bin > 0]
                mean_conf = float(mask_pixels.mean()) if mask_pixels.size > 0 else 0.0
                is_thin = (min(h_ds, w_ds) <= self._THIN_PX_AT_DS_FLOAT)
                mask_stats.append((area_ds, h_ds, w_ds, mean_conf, is_thin))

        depth_strong_hi = (conf_smooth >= 0.90)  # frame-level, hoist out of instance loop

        # --- Parallel per-instance fusion ---
        # Each instance is processed concurrently using its own buffer set.
        # Read-only shared state (ds_conf, ds_union, hsv_cache, etc.) is safe.
        # All writes are returned as a result dict and merged sequentially below.
        futures = []
        for idx, inst in enumerate(gi.yolo_instances):
            if idx >= len(probs) or idx >= len(mask_stats):
                continue
            cls_id = inst.cls if isinstance(inst.cls, (int, np.integer)) else -1
            thr_map = (self.class_thr or {})
            thr = float(thr_map.get(int(cls_id),
                        thr_map.get(str(cls_id),
                            thr_map.get('*',
                                THR_MAP_FALLBACK.get(str(cls_id),
                                    THR_MAP_FALLBACK.get('*', DEFAULT_FUSE_THR))))))
            buf_idx = idx % len(self._inst_bufs)
            fut = self._inst_pool.submit(
                self._fuse_one,
                inst, probs[idx], bins[idx], mask_stats[idx],
                conf, conf_smooth, depth_strong, depth_strong_hi,
                yolo_reliable, float(gi.health.inlier_ratio), alpha, thr, int(cls_id),
                self._inst_bufs[buf_idx],
            )
            futures.append(fut)

        # Collect results and apply all state mutations sequentially
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            # Threshold info
            if r["thr_cls_id"] is not None:
                thr_info[r["thr_cls_id"][0]] = r["thr_cls_id"][1]
            # Fused instance
            if r["fused_inst"] is not None:
                fused.append(r["fused_inst"])
            # depth_memory write-back
            if r["mem_key"] is not None and r["mem_update"] is not None:
                self._depth_memory[r["mem_key"]] = r["mem_update"]
            # approved_additions write-back
            if r["approved_key"] is not None:
                if r["approved_update"] == "DELETE":
                    self._approved_additions.pop(r["approved_key"], None)
                elif r["approved_update"] is not None:
                    self._approved_additions[r["approved_key"]] = r["approved_update"]
            # repair_cache write-back (apply step: decrement/pop)
            if r["repair_apply"] is not None:
                action, rc_cls, rc_bbox = r["repair_apply"]
                mi = self._match_repair_entry(rc_cls, rc_bbox)
                if mi >= 0:
                    if action == "decrement":
                        self._repair_cache[mi]["ttl"] -= 1
                    else:  # "pop"
                        self._repair_cache.pop(mi)
            # repair_cache write-back (refresh step: add/update)
            if r["repair_new_entry"] is not None:
                e = r["repair_new_entry"]
                mi = self._match_repair_entry(e["cls"], e["bbox"])
                if mi >= 0:
                    self._repair_cache[mi]["mask_u8"] = (self._repair_cache[mi]["mask_u8"] | e["mask_u8"])
                    self._repair_cache[mi]["bbox"] = e["bbox"]
                    self._repair_cache[mi]["ttl"] = e["ttl"]
                else:
                    self._repair_cache.append(e)

        t2_loop = time.perf_counter()
        if MOE_PROF:
            print(f"[gate/timer] loop={(t2_loop - t1_prep)*1000:.1f}ms")

        # ---------------- Depth-only proposals (MODIFIED) ----------------
        self._ds_covered.fill(0)
        for f in fused:
            if f.area >= self.min_area:
                cv2.resize((f.mask > 0).astype(np.uint8),
                        (self.DS_W, self.DS_H),
                        dst=self._ds_tmp_u8,
                        interpolation=cv2.INTER_NEAREST)
                self._ds_covered |= self._ds_tmp_u8

        # SIMPLIFIED/AGGRESSIVE proposal conditions
        mu = float(self._ds_conf.mean())
        # --- depth signal checks on the DS grid ---
        tau_seed = 0.22  # or max(0.22, DEPTH_THR_PROPOSE)
        ds = self._ds_conf

        peak = float(ds.max())
        fg = (ds >= tau_seed)
        fg_px = int(fg.sum())
        fg_frac = fg_px / float(self.DS_W * self.DS_H)

        # Local-signal notion (works even when plane_ok=True and conf is sparse)
        has_signal_local = (peak >= 0.55) or (fg_frac >= 0.0025)
        self._ds_prop_mask = self._ds_prop_mask.astype(np.uint8, copy=False)
        self._ds_prop_mask[...] = fg.astype(np.uint8)

        # In-place uncovered region: ds_prop_mask & (~ds_covered) without Python "~" on u8
        cv2.bitwise_not(self._ds_covered, dst=self._ds_tmp_u8)
        cv2.bitwise_and(self._ds_prop_mask, self._ds_tmp_u8, dst=self._ds_prop_mask)
        uncovered_ds = self._ds_prop_mask
        MIN_UNCOVERED = 60

        yolo_dead_now = (union_px_ds < 50) or (self._dead_streak >= 2)

        # If YOLO is empty, allow an even cheaper/sooner rescue
        yolo_empty = (len(gi.yolo_instances) == 0)
        force_depth_now = yolo_empty and ((peak >= 0.40) or (fg_frac >= 0.0015))

        if force_depth_now:
            RUN_PROPOSALS = (self._max_proposals > 0)   # ignore decimation/uncovered checks
        else:
            # gate_good policy:
            # - respect decimation, but
            # - allow depth proposals whenever:
            #   * YOLO is weak (yolo_weak) OR
            #   * YOLO is dead (yolo_dead_now) OR
            #   * there is enough uncovered confident depth (bypass decimation).
            uncovered_sum = int(uncovered_ds.sum())
            has_strong_uncovered = uncovered_sum >= (MIN_UNCOVERED * 3)  # ~180px DS = clear missed object
            if not yolo_weak and not has_strong_uncovered and (self._frame_idx % PROP_DECIMATE_N) != 0:
                RUN_PROPOSALS = False
            else:
                RUN_PROPOSALS = (
                    (self._max_proposals > 0)
                    and has_signal_local
                    and (
                        len(fused) == 0 or
                        int(uncovered_ds.sum()) >= MIN_UNCOVERED or
                        yolo_dead_now
                    )
                )

        # ---- Per-frame time budget guard for proposals ----
        PROP_BUDGET_MS = self._PROP_BUDGET_MS
        if (time.perf_counter() - t2_loop) * 1000.0 > PROP_BUDGET_MS:
            RUN_PROPOSALS = False

        max_props_now = (adaptive_max_props if RUN_PROPOSALS else 0)
        if max_props_now == 0:
            if self._USE_FAST_NMS:
                fused = self._nms_by_iou_fast(fused, iou_thr=NMS_IOU_THR)
            else:
                fused = self._nms_by_iou(fused, iou_thr=NMS_IOU_THR)
            return GateOutput(t=gi.t, fused=fused, health=gi.health, fuse_thr_info=thr_info)

        # Candidates only from uncovered confident depth
        depth_bin_ds = uncovered_ds.astype(np.uint8, copy=False)

        contours, _ = cv2.findContours(
            depth_bin_ds,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            if self._USE_FAST_NMS:
                fused = self._nms_by_iou_fast(fused, iou_thr=NMS_IOU_THR)
            else:
                fused = self._nms_by_iou(fused, iou_thr=NMS_IOU_THR)
            return GateOutput(t=gi.t, fused=fused, health=gi.health, fuse_thr_info=thr_info)

        candidates = []

        for cnt in contours:
            area_ds = int(cv2.contourArea(cnt))
            # Thin/flat contours (e.g. a wrench seen edge-on) have contourArea≈0
            # even though they occupy real pixels. Fall back to bounding-rect area.
            if area_ds < self.min_area_ds:
                _bx, _by, _bw, _bh = cv2.boundingRect(cnt)
                area_ds = max(area_ds, _bw * _bh)
            if area_ds < self.min_area_ds:
                continue

            # Reuse existing DS buffer
            self._ds_prop_mask.fill(0)
            cv2.drawContours(self._ds_prop_mask, [cnt], -1, 1, thickness=-1)

            pixels = self._ds_conf[self._ds_prop_mask > 0]
            if pixels.size == 0:
                continue

            mean_conf = float(pixels.mean())
            mean_keep = max(0.25, min(0.40, mu + 0.02))
            if mean_conf < mean_keep:
                continue

            candidates.append((cnt, area_ds, mean_conf))
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Time budget to avoid spikes
        t_prop_start = time.perf_counter()
        BUDGET_MS = PROP_BUDGET_MS

        kept = 0
        # Upsample DS covered mask to full 640x640 canvas
        covered_full = cv2.resize(self._ds_covered, (640, 640),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

        for cnt, area_ds, mean_conf in candidates:
            if kept >= max_props_now:
                break
            if (time.perf_counter() - t_prop_start) * 1000.0 > BUDGET_MS:
                break

            self._ds_prop_mask.fill(0)
            cv2.drawContours(self._ds_prop_mask, [cnt], -1, 1, thickness=-1)
            blob_ds = self._ds_prop_mask

            # ring-contrast veto in DS (looser, like gate_prev)
            # For thin/elongated objects the ring is large relative to the core;
            # use a tighter threshold so we don't incorrectly veto a thin wrench.
            _bx2, _by2, _bw2, _bh2 = cv2.boundingRect(cnt)
            _is_thin_cnt = (_bh2 <= 3 or _bw2 <= 3) or (min(_bh2, _bw2) / max(_bw2, _bh2, 1) < 0.12)
            _ring_thr = 0.03 if _is_thin_cnt else 0.06
            if self._ring_contrast_delta_ds(self._ds_conf, blob_ds) < _ring_thr:
                continue

            blob = cv2.resize(blob_ds, (640, 640),
                            interpolation=cv2.INTER_NEAREST)

            # Skip if mostly overlapping existing fused (dup/over-expansion guard)
            cv2.bitwise_and(blob, covered_full.astype(np.uint8), dst=self._tmp_mask640)
            overlap = float(self._tmp_mask640.sum()) / max(1.0, float(blob.sum()))
            if overlap > PROPOSAL_OVERLAP_FRAC:
                continue

            # Compute proposal bbox FIRST (needed for bbox prefilter below)
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1 = int(x * 640 / self.DS_W), int(y * 640 / self.DS_H)
            x2 = int((x + w - 1) * 640 / self.DS_W)
            y2 = int((y + h - 1) * 640 / self.DS_H)
            x1 = max(0, min(639, x1)); x2 = max(0, min(639, x2))
            y1 = max(0, min(639, y1)); y2 = max(0, min(639, y2))
            prop_bbox = (x1, y1, x2, y2)

            dup = False
            for f in fused:
                if f.area < self.min_area:
                    continue
                if not self._bbox_iou_prefilter(prop_bbox, f.bbox_xyxy, 0.05):
                    continue
                if self._mask_iou(blob, f.mask) >= PROPOSAL_IOU_SKIP:
                    dup = True
                    break

            if dup:
                continue

            depth_support = float(conf[blob > 0].mean()) if blob.any() else 0.0
            dc = max(0.0, min(1.0, depth_support))
            score_fused = (1.0 - alpha) * dc  # depth-only score

            mean_depth_ds = float(self._ds_conf[blob_ds > 0].mean()) if blob_ds.any() else 0.0
            mean_prob_ds  = 0.0

            # -------- NEW: Merge proposal into existing instance if plausible --------
            self._prop_dbg_total += 1

            prop_u8 = blob.astype(np.uint8, copy=False)
            prop_bbox = (x1, y1, x2, y2)

            merged = False
            if self._PROP_MERGE_ENABLE and fused:
                best_i, best_s, diag = self._associate_proposal_to_instance(prop_u8, prop_bbox, fused)
                if best_i >= 0:
                    # Merge into the matched instance (repair)
                    self._merge_proposal_into_instance(fused[best_i], prop_u8, conf, alpha, diag=diag)
                    self._prop_dbg_merged += 1
                    merged = True

            if not merged:
                # Keep old behavior only when no strong association:
                fused.append(FusedInst(
                    mask=prop_u8, cls=-1,
                    score_rgb=0.0, score_fused=score_fused,
                    area=int(prop_u8.sum()), bbox_xyxy=prop_bbox,
                    depth_support=depth_support, mode="depth-led",
                    mean_prob_ds=mean_prob_ds, mean_depth_ds=mean_depth_ds,
                    assist_rgb_frac=0.0, assist_depth_frac=1.0
                ))
                self._prop_dbg_new += 1
                kept += 1
            else:
                # We consumed a proposal budget but did not add a new instance
                kept += 1

        t3_prop = time.perf_counter()
        if MOE_PROF:
            print(f"[gate/timer] proposals={(t3_prop - t2_loop)*1000:.1f}ms")

        if MOE_DEBUG and (self._prop_dbg_total > 0) and (self._frame_idx % 30 == 0):
            print(f"[prop/stats] total={self._prop_dbg_total} merged={self._prop_dbg_merged} new={self._prop_dbg_new}")
            # Reset every print interval to keep numbers readable
            self._prop_dbg_total = 0
            self._prop_dbg_merged = 0
            self._prop_dbg_new = 0

        # Final de-duplication across all sources
        if self._USE_FAST_NMS:
            fused = self._nms_by_iou_fast(fused, iou_thr=NMS_IOU_THR)
        else:
            fused = self._nms_by_iou(fused, iou_thr=NMS_IOU_THR)

        # Make depth-assisted mask expansions "sticky" for a few frames
        fused = self._apply_depth_sticky(fused)

        # Per-instance short-term hold to prevent single-tool flicker
        fused, age_override = self._apply_instance_hold(gi, fused)
        
        fused = self._stabilize_depth_only(fused)

        # Final flicker guard: if everything got dropped but YOLO still has instances,
        # synthesize RGB-only masks instead of returning an empty frame.
        if not fused and gi.yolo_instances:
            fused = self._fallback_rgb_only(gi, thr_info)
            # Fallback is RGB-only; reset cache
            self._update_instance_cache(fused, {})
        else:
            # Normal case: update per-instance cache from fused list
            self._update_instance_cache(fused, age_override)

        t4_end = time.perf_counter()
        if MOE_PROF:
            print(f"[gate/timer] nms+return={(t4_end - t3_prop)*1000:.1f}ms, total={(t4_end - t0_all)*1000:.1f}ms\n")

        return GateOutput(t=gi.t, fused=fused, health=gi.health, fuse_thr_info=thr_info)

    # ----------------- Per-instance parallel worker -----------------
    def _fuse_one(self, inst, m_prob, m_bin, mask_stat,
                  conf, conf_smooth, depth_strong, depth_strong_hi,
                  yolo_reliable, inlier_ratio, alpha, thr, cls_id, bufs):
        """
        Fuse one YOLO instance with depth. Designed to run concurrently on a
        ThreadPoolExecutor worker — all writable state is confined to `bufs`
        (per-worker pre-allocated arrays) or returned as a result dict for the
        caller to apply sequentially. Read-only access to shared numpy arrays
        (ds_conf, ds_union, etc.) is GIL-safe.

        Returns a dict:
            fused_inst      : FusedInst | None
            thr_cls_id      : (cls_id, thr) | None
            mem_key         : tuple | None
            mem_update      : ndarray | None
            approved_key    : tuple | None
            approved_update : dict | "DELETE" | None
            repair_apply    : ("decrement"|"pop", cls, bbox) | None
            repair_new_entry: dict | None
        """
        result = {
            "fused_inst":       None,
            "thr_cls_id":       (cls_id, thr),
            "mem_key":          None,
            "mem_update":       None,
            "approved_key":     None,
            "approved_update":  None,
            "repair_apply":     None,
            "repair_new_entry": None,
        }

        area_ds, h_ds, w_ds, mean_conf, is_thin = mask_stat

        if area_ds < self.min_area_ds:
            return result

        # Fast-path for thin objects
        if is_thin and mean_conf < 0.30 and float(inst.score) >= 0.35:
            fi = self._handle_thin_object_fast(inst, m_bin, self._ds_conf)
            result["fused_inst"] = fi
            return result

        # Local stats
        yolo_masked = m_prob[m_bin > 0]
        depth_vals  = self._ds_conf[m_bin > 0]
        if depth_vals.size == 0:
            return result

        S_rgb   = float(inst.score) * float(yolo_masked.mean())
        S_depth = float(depth_vals.mean())
        V_depth = float(depth_vals.var())

        # Overlap ratio
        if self._ds_union.any():
            inter_with_all = float((m_bin & self._ds_union).sum())
            overlap_ratio  = max(0.0, inter_with_all - float(m_bin.sum())) / max(1.0, float(m_bin.sum()))
        else:
            overlap_ratio = 0.0

        R_rgb   = S_rgb * (1.0 - OVERLAP_PENALTY * overlap_ratio)
        thin_bonus = THIN_BONUS if is_thin else 0.0
        R_depth = max(0.0, S_depth + thin_bonus) * inlier_ratio / (1.0 + V_depth + VAR_DAMPING_EPS)

        rgb_led = (R_rgb > R_depth)

        if rgb_led:
            # ----------------------------------------------------------------
            # RGB-led path
            # ----------------------------------------------------------------
            t_hi, t_lo = 0.60, 0.35
            core_ds    = (m_prob >= t_hi).astype(np.uint8)
            support_ds = (m_prob >= t_lo).astype(np.uint8)

            is_thin_local   = is_thin
            mean_conf_local = mean_conf
            rgb_strong      = float(inst.score) >= self._RGB_STRONG
            weak_depth_case = mean_conf_local < self._WEAK_DEPTH_MEAN

            if self._RGB_ONLY_IF_THIN and (is_thin_local or weak_depth_case or rgb_strong):
                m_fused_ds        = support_ds
                assist_rgb_frac   = 1.0
                assist_depth_frac = 0.0
                mean_prob_ds  = float(m_prob[m_fused_ds > 0].mean()) if m_fused_ds.any() else 0.0
                mean_depth_ds = float(self._ds_conf[m_fused_ds > 0].mean()) if m_fused_ds.any() else 0.0
            else:
                ys, xs = np.where(support_ds > 0)
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                x1 = max(0, x1 - 2); y1 = max(0, y1 - 2)
                x2 = min(self.DS_W - 1, x2 + 2)
                y2 = min(self.DS_H - 1, y2 + 2)

                sup_roi  = support_ds[y1:y2+1, x1:x2+1]
                core_roi = core_ds[y1:y2+1, x1:x2+1]
                dep_ok   = (self._ds_conf_thr[y1:y2+1, x1:x2+1] > 0).astype(np.uint8)

                dil_core = cv2.dilate(core_roi, self._k_ds3, iterations=2)
                prob_roi = m_prob[y1:y2+1, x1:x2+1]
                yolo_truly_absent = (prob_roi < 0.05)
                add_dep   = (dep_ok & dil_core) & (~sup_roi) & yolo_truly_absent
                fused_roi = (sup_roi | add_dep).astype(np.uint8)

                m_fused_ds = np.zeros_like(support_ds)
                m_fused_ds[y1:y2+1, x1:x2+1] = fused_roi

                fused_core = (core_roi > 0) & (fused_roi > 0)
                fused_dep  = ((add_dep > 0) & (fused_roi > 0))
                assist_rgb_frac   = float(fused_core.sum()) / max(1.0, float(fused_roi.sum()))
                assist_depth_frac = float(fused_dep.sum())  / max(1.0, float(fused_roi.sum()))
                mean_prob_ds  = float(m_prob[m_fused_ds > 0].mean()) if m_fused_ds.any() else 0.0
                mean_depth_ds = float(self._ds_conf[m_fused_ds > 0].mean()) if m_fused_ds.any() else 0.0

            # Tip preservation
            if is_thin_local and mean_conf_local < self._WEAK_DEPTH_MEAN and float(inst.score) >= 0.35:
                halo   = cv2.dilate(m_bin, self._k_ds3, iterations=1)
                weak   = (self._ds_conf < self._WEAK_DEPTH_PIX).astype(np.uint8)
                rescue = (halo & weak) & (~m_fused_ds)
                if rescue.sum() <= self._RESCUE_CAP_DS:
                    m_fused_ds = (m_fused_ds | rescue).astype(np.uint8)

            # Ring contrast check
            if self._ring_contrast_delta_ds(self._ds_conf, m_fused_ds) < 0.08:
                return result

            prior_ds = (m_fused_ds > 0)
            if not prior_ds.any():
                return result

            m_full = inst.mask_prob
            if m_full is None or m_full.size == 0:
                return result

            m_full = m_full.astype(np.float32, copy=False)
            if m_full.max() > 1.0:
                m_full *= (1.0 / 255.0)

            tmp640 = bufs["tmp_mask640"]
            if m_full.shape == (640, 640):
                m640 = m_full
            elif m_full.ndim == 2 and m_full.shape[0] == m_full.shape[1]:
                # Square mask (e.g. 160x160) — YOLO grid space, resize directly.
                cv2.resize(m_full, (640, 640), dst=tmp640, interpolation=cv2.INTER_LINEAR)
                m640 = tmp640
            else:
                m640 = depth_to_yolo_img(m_full, self._lb, is_mask=True, out=tmp640)

            yolo_bool = (m640 >= 0.5)
            if not yolo_bool.any():
                return result

            _bx, _by, _bw, _bh = cv2.boundingRect(yolo_bool.view(np.uint8))
            bx1, by1 = _bx, _by
            bx2, by2 = _bx + _bw - 1, _by + _bh - 1
            bbox_yolo = (bx1, by1, bx2, by2)

            # Persistent depth memory
            pad = 24
            rx1 = max(0, bx1 - pad); ry1 = max(0, by1 - pad)
            rx2 = min(639, bx2 + pad); ry2 = min(639, by2 + pad)

            _yolo_roi_u8 = yolo_bool[ry1:ry2+1, rx1:rx2+1].astype(np.uint8)
            _yolo_nearby = cv2.dilate(_yolo_roi_u8, self._k7, iterations=10) > 0
            _depth_add_roi = depth_strong[ry1:ry2+1, rx1:rx2+1] & (~yolo_bool[ry1:ry2+1, rx1:rx2+1]) & _yolo_nearby

            depth_add = bufs["depth_add_buf"]
            depth_add.fill(False)
            depth_add[ry1:ry2+1, rx1:rx2+1] = _depth_add_roi

            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            mem_key = (int(inst.cls), cx // 32, cy // 32)

            # Read current memory (GIL-safe dict read)
            mem = self._depth_memory.get(mem_key)
            if mem is None:
                mem = np.zeros_like(depth_add, dtype=bool)
            else:
                mem = mem.copy()  # local copy so we don't alias shared state

            mem = mem & depth_strong
            roi_sl = (slice(ry1, ry2+1), slice(rx1, rx2+1))
            mem[roi_sl] = mem[roi_sl] | depth_add[roi_sl]
            # Schedule write-back via result dict
            result["mem_key"]    = mem_key
            result["mem_update"] = mem

            mem_clean = bufs["mem_clean_buf"]
            mem_clean.fill(False)
            roi_mem = mem[roi_sl]
            if roi_mem.any():
                eroded = cv2.erode(roi_mem.astype(np.uint8), self._k3, iterations=1)
                mem_clean[roi_sl] = eroded > 0

            m640_roi  = m640[roi_sl]
            ds_hi_roi = depth_strong_hi[roi_sl]
            da_roi    = (mem_clean[roi_sl]
                         & ((m640_roi < 0.05) | ((m640_roi >= 0.05) & (m640_roi < 0.15) & ds_hi_roi)))

            depth_additions = bufs["depth_add_buf"]   # reuse
            depth_additions.fill(False)
            depth_additions[roi_sl] = da_roi
            m_fused_bool = yolo_bool | depth_additions

            # Appearance-consistency filter
            new_approved_entry = None
            delete_approved    = False
            if depth_additions.any() and self._hsv_cache is not None:
                da_roi_u8 = depth_additions[roi_sl].astype(np.uint8)
                n_cc, cc_labels_roi = cv2.connectedComponents(da_roi_u8, connectivity=8)

                clean_additions = bufs["clean_additions"]
                clean_additions.fill(False)

                # Read approved entry snapshot (GIL-safe)
                entry = self._approved_additions.get(mem_key)
                if entry is not None:
                    ex1, ey1, ex2, ey2 = entry['bbox']
                    drift = max(abs(ex1 - bx1), abs(ey1 - by1),
                                abs(ex2 - bx2), abs(ey2 - by2))
                    if drift > 20:
                        entry = None
                        delete_approved = True

                comp_full_buf = bufs["comp_full_buf"]
                for lbl in range(1, n_cc):
                    component = (cc_labels_roi == lbl)
                    comp_full_buf.fill(False)
                    comp_full_buf[roi_sl] = component
                    comp_full = comp_full_buf

                    if self._appearance_consistent(None, yolo_bool, comp_full):
                        clean_additions[roi_sl] |= component
                        if entry is None:
                            new_approved_entry = {
                                'mask': comp_full.copy(),
                                'bbox': bbox_yolo,
                                'ttl':  self._APPROVED_TTL,
                            }
                        else:
                            # Build updated entry locally
                            new_approved_entry = {
                                'mask': (entry['mask'] | comp_full).copy(),
                                'bbox': bbox_yolo,
                                'ttl':  self._APPROVED_TTL,
                            }
                            entry = new_approved_entry  # keep accumulating
                    else:
                        if entry is not None and entry['ttl'] > 0:
                            held = entry['mask'] & depth_strong
                            clean_additions |= held
                            new_approved_entry = {
                                'mask': held.copy(),
                                'bbox': bbox_yolo,
                                'ttl':  entry['ttl'] - 1,
                            }
                            entry = new_approved_entry
                        elif entry is not None and entry['ttl'] <= 0:
                            delete_approved = True
                            entry = None

                m_fused_bool = yolo_bool | clean_additions

            # Schedule approved_additions write-back
            result["approved_key"] = mem_key
            if delete_approved and new_approved_entry is None:
                result["approved_update"] = "DELETE"
            elif new_approved_entry is not None:
                result["approved_update"] = new_approved_entry

            if not m_fused_bool.any():
                return result

            # Repair latch: apply cached repair (read-only access to self._repair_cache)
            if self._REPAIR_LATCH_ENABLE:
                mi = self._match_repair_entry(inst.cls, bbox_yolo)
                if mi >= 0:
                    e = self._repair_cache[mi]
                    if e["ttl"] > 0:
                        m_fused_bool = (m_fused_bool | (e["mask_u8"] > 0))
                        result["repair_apply"] = ("decrement", int(inst.cls), bbox_yolo)
                    else:
                        result["repair_apply"] = ("pop", int(inst.cls), bbox_yolo)

            # Repair latch: refresh from current depth
            if self._REPAIR_LATCH_ENABLE:
                if int(depth_add.sum()) >= self._REPAIR_LATCH_MINPIX:
                    result["repair_new_entry"] = {
                        "cls":     int(inst.cls),
                        "bbox":    bbox_yolo,
                        "mask_u8": depth_add.astype(np.uint8).copy(),
                        "ttl":     self._REPAIR_LATCH_TTL,
                    }

            m_fused = m_fused_bool.astype(np.uint8)
            if m_fused.sum() < self.min_area:
                return result

            x1, y1, w, h = cv2.boundingRect(m_fused_bool.astype(np.uint8))
            x2 = x1 + w - 1
            y2 = y1 + h - 1
            roi      = m_fused_bool[y1:y2+1, x1:x2+1]
            roi_conf = conf[y1:y2+1, x1:x2+1]
            depth_support = float(roi_conf[roi].mean()) if roi.any() else 0.0

            score_fused = alpha * float(inst.score) + (1.0 - alpha) * max(0.0, min(1.0, depth_support))
            if score_fused < thr:
                return result

            result["fused_inst"] = FusedInst(
                mask=m_fused, cls=int(inst.cls),
                score_rgb=float(inst.score), score_fused=score_fused,
                area=int(m_fused.sum()), bbox_xyxy=(x1, y1, x2, y2),
                depth_support=depth_support, mode="rgb-led",
                mean_prob_ds=mean_prob_ds, mean_depth_ds=mean_depth_ds,
                assist_rgb_frac=assist_rgb_frac, assist_depth_frac=assist_depth_frac,
            )
            return result

        else:
            # ----------------------------------------------------------------
            # Depth-led path
            # ----------------------------------------------------------------
            dil_ds   = cv2.dilate(m_bin, self._k_ds3, iterations=1)
            local_ds = (self._ds_conf_thr & dil_ds)

            if local_ds.sum() < self.min_area_ds:
                ys, xs = np.where(m_bin > 0)
                if ys.size == 0:
                    return result
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                roi_prob = m_prob[y1:y2+1, x1:x2+1]
                roi_conf = self._ds_conf[y1:y2+1, x1:x2+1]
                roi_bin  = (roi_prob * roi_conf) >= 0.45
                m_fused_ds = np.zeros_like(m_bin)
                m_fused_ds[y1:y2+1, x1:x2+1] = roi_bin.astype(np.uint8)
            else:
                m_fused_ds = self._largest_cc(local_ds)

            if yolo_reliable and (m_bin.sum() > 0) and (float(inst.score) >= 0.40):
                yolo_vic   = cv2.dilate(m_bin, self._k_ds3, iterations=2)
                m_fused_ds = (m_fused_ds & yolo_vic).astype(np.uint8)

            assist_rgb_frac   = float(((m_fused_ds > 0) & (m_bin > 0)).sum()) / max(1.0, float(m_fused_ds.sum()))
            assist_depth_frac = 1.0 - assist_rgb_frac
            mean_prob_ds  = float(m_prob[m_fused_ds > 0].mean()) if m_fused_ds.any() else 0.0
            mean_depth_ds = float(self._ds_conf[m_fused_ds > 0].mean()) if m_fused_ds.any() else 0.0

            if self._ring_contrast_delta_ds(self._ds_conf, m_fused_ds) < 0.08:
                return result

            m_fused = cv2.resize(m_fused_ds, (640, 640), interpolation=cv2.INTER_NEAREST)
            if m_fused.sum() < self.min_area:
                return result

            x1, y1, w, h = cv2.boundingRect((m_fused > 0).astype(np.uint8))
            x2 = x1 + w - 1
            y2 = y1 + h - 1
            roi_mask     = m_fused[y1:y2+1, x1:x2+1] > 0
            roi_conf     = conf[y1:y2+1, x1:x2+1]
            depth_support = float(roi_conf[roi_mask].mean()) if roi_mask.any() else 0.0
            dc = max(0.0, min(1.0, float(depth_support)))
            score_fused = alpha * float(inst.score) + (1.0 - alpha) * dc

            if score_fused < thr:
                return result

            result["fused_inst"] = FusedInst(
                mask=m_fused, cls=int(inst.cls),
                score_rgb=float(inst.score), score_fused=score_fused,
                area=int(m_fused.sum()), bbox_xyxy=(x1, y1, x2, y2),
                depth_support=depth_support, mode="depth-led",
                mean_prob_ds=mean_prob_ds, mean_depth_ds=mean_depth_ds,
                assist_rgb_frac=assist_rgb_frac, assist_depth_frac=assist_depth_frac,
            )
            return result

    # ----------------- Helpers -----------------
    def _stabilize_depth_only(self, fused: List[FusedInst]) -> List[FusedInst]:
        """
        Stabilize depth-only proposals (cls = -1) so they don't flicker
        when depth confidence fluctuates slightly.

        FIX: Only replay cached items that have no match in the current frame.
        The original code added both the current detection AND the cached copy,
        causing every depth proposal to appear twice (or more if cache built up).
        """
        if not fused:
            return fused

        cache = getattr(self, "_depth_only_cache", [])
        age   = getattr(self, "_depth_only_age", [])

        new_cache = []
        new_age   = []
        result    = []

        # Step 1: add all current frame instances, build new cache
        for f in fused:
            result.append(f)
            if f.cls == -1:
                new_cache.append(f)
                new_age.append(0)

        # Step 2: replay cached depth proposals ONLY if they have no match
        # in the current frame (prevents duplication)
        for prev, a in zip(cache, age):
            if a >= 3:
                continue
            # Check if current frame already has a matching proposal
            matched = any(
                self._bbox_iou_xyxy(prev.bbox_xyxy, f.bbox_xyxy) > 0.4
                for f in fused if f.cls == -1
            )
            if not matched:
                # Genuinely missing this frame — replay from cache
                new_cache.append(prev)
                new_age.append(a + 1)
                result.append(prev)

        self._depth_only_cache = new_cache
        self._depth_only_age = new_age
        return result

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        a_bool = a > 0
        b_bool = b > 0
        inter = np.count_nonzero(a_bool & b_bool)
        if inter <= 0:
            return 0.0
        union = np.count_nonzero(a_bool | b_bool)
        return float(inter) / max(union, 1.0)
    
    def _bbox_from_mask(self, m01: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        ys, xs = np.where(m01 > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        return (x1, y1, x2, y2)

    def _bbox_iou_prefilter(self, a: Tuple[int,int,int,int], b: Tuple[int,int,int,int], min_iou: float) -> bool:
        # Cheap reject before mask ops
        return self._bbox_iou_xyxy(a, b) >= float(min_iou)

    def _appearance_consistent(self, bgr: np.ndarray,
                            fg_mask: np.ndarray,
                            candidate_mask: np.ndarray,
                            threshold: float = 0.55) -> bool:
        # Always use the full-frame HSV pre-computed in process().
        # bgr is accepted for emergencies only (e.g. unit tests without _hsv_cache).
        hsv = self._hsv_cache
        if hsv is None and bgr is not None:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        if hsv is None:
            return True

        # Downsample masks to match downsampled HSV cache
        scale = getattr(self, '_hsv_ds_scale', 1.0)
        if scale != 1.0:
            h_ds, w_ds = hsv.shape[0], hsv.shape[1]
            fg_mask_use   = cv2.resize(fg_mask.astype(np.uint8),        (w_ds, h_ds), interpolation=cv2.INTER_NEAREST).astype(bool)
            cand_mask_use = cv2.resize(candidate_mask.astype(np.uint8), (w_ds, h_ds), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            fg_mask_use   = fg_mask.astype(bool)
            cand_mask_use = candidate_mask.astype(bool)

        if hsv.shape[:2] != cand_mask_use.shape:
            return True

        if fg_mask_use.sum() < 10 or cand_mask_use.sum() < 10:
            return True

        def hist_hue(mask):
            h = cv2.calcHist([hsv], [0], mask.astype(np.uint8), [32], [0, 180])
            cv2.normalize(h, h, norm_type=cv2.NORM_L1)
            return h

        dist = cv2.compareHist(hist_hue(fg_mask_use), hist_hue(cand_mask_use),
                            cv2.HISTCMP_BHATTACHARYYA)
        return dist < threshold

    def _associate_proposal_to_instance(self, prop_u8: np.ndarray, prop_bbox, fused: List[FusedInst]):
        px1, py1, px2, py2 = prop_bbox
        prop_area = int(prop_u8.sum())
        if prop_area <= 0:
            return -1, 0.0, {"reason": "empty"}

        best_i = -1
        best_s = -1.0
        second_s = -1.0
        best_diag = {}

        # Precompute once
        prop_bool = (prop_u8 > 0)

        # BBox proximity margin (in pixels)
        # (lets a “missing head chunk” merge even if bbox IoU is ~0)
        margin_px = int(max(2, self._PROP_MERGE_MARGIN * 640))

        for i, inst in enumerate(fused):
            if getattr(inst, "cls", -1) < 0:
                continue  # never merge into a proposal blob

            ix1, iy1, ix2, iy2 = inst.bbox_xyxy

            # Fast reject if far apart (bbox expanded by margin)
            if (px2 < ix1 - margin_px) or (px1 > ix2 + margin_px) or (py2 < iy1 - margin_px) or (py1 > iy2 + margin_px):
                continue

            inst_area = int(inst.area)
            if inst_area <= 0:
                continue

            # Prevent absurd expansions, but allow real repairs
            if prop_area > int(self._PROP_MERGE_MAX_EXPAND * inst_area):
                continue

            inst_bool = (inst.mask > 0)

            inter = int((prop_bool & inst_bool).sum())
            overlap_frac = inter / float(prop_area)

            # ----- ROI-based dilation (FAST) -----
            # Compute tight ROI covering both proposal and instance bbox
            ix1, iy1, ix2, iy2 = inst.bbox_xyxy
            rx1 = max(0, min(px1, ix1) - 4)
            ry1 = max(0, min(py1, iy1) - 4)
            rx2 = min(prop_u8.shape[1]-1, max(px2, ix2) + 4)
            ry2 = min(prop_u8.shape[0]-1, max(py2, iy2) + 4)

            # Slice ROI
            prop_roi = prop_bool[ry1:ry2+1, rx1:rx2+1]
            inst_roi = inst_bool[ry1:ry2+1, rx1:rx2+1]

            # Dilate only inside ROI
            inst_u8_roi = inst_roi.astype(np.uint8)
            dil_roi = cv2.dilate(inst_u8_roi, self._k_touch, iterations=1)

            touch = int((prop_roi & (dil_roi > 0)).sum())
            touch_frac = touch / float(prop_area)

            # BBox IoU (very weak signal for chunk repairs)
            iou_bbox = self._bbox_iou_xyxy((px1, py1, px2, py2), (ix1, iy1, ix2, iy2))

            # Hard minimums (now relaxed)
            if (touch_frac < self._PROP_MERGE_T_TOUCH) and (overlap_frac < self._PROP_MERGE_T_OVERLAP) and (iou_bbox < self._PROP_MERGE_T_IOU):
                continue

            s = (
                self._PROP_MERGE_W_OVERLAP * overlap_frac +
                self._PROP_MERGE_W_TOUCH   * touch_frac +
                self._PROP_MERGE_W_IOU     * iou_bbox
            )

            if s > best_s:
                second_s = best_s
                best_s = s
                best_i = i
                best_diag = {
                    "overlap_frac": overlap_frac,
                    "touch_frac": touch_frac,
                    "iou_bbox": iou_bbox,
                    "prop_area": prop_area,
                    "inst_area": inst_area,
                    "margin_px": margin_px,
                }
            elif s > second_s:
                second_s = s

        # Ambiguity check: only accept if clearly best
        if best_i < 0:
            return -1, 0.0, {"reason": "no_candidate"}

        if best_s < self._PROP_MERGE_T_ACCEPT:
            if getattr(self, "_PROP_MERGE_DBG", False) and MOE_DEBUG:
                print(f"[prop/merge] reject: best_s={best_s:.3f} diag={best_diag}")
            return -1, best_s, {"reason": "below_accept", **best_diag}

        if (best_s - second_s) < self._PROP_MERGE_MARGIN:
            if getattr(self, "_PROP_MERGE_DBG", False) and MOE_DEBUG:
                print(f"[prop/merge] reject: ambiguous best={best_s:.3f} second={second_s:.3f} diag={best_diag}")
            return -1, best_s, {"reason": "ambiguous", "second_s": second_s, **best_diag}

        if getattr(self, "_PROP_MERGE_DBG", False) and MOE_DEBUG:
            print(f"[prop/merge] accept: s={best_s:.3f} -> inst#{best_i} diag={best_diag}")

        return best_i, best_s, best_diag

    def _bbox_iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
        area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
        return float(inter) / float(area_a + area_b - inter + 1e-6)

    def _match_repair_entry(self, cls_id, bbox):
        # Return index of best matching cache entry, or -1
        best_i = -1
        best_iou = 0.0
        for i, e in enumerate(self._repair_cache):
            if e["cls"] != cls_id:
                continue
            iou = self._bbox_iou(e["bbox"], bbox)
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i >= 0 and best_iou >= self._REPAIR_LATCH_IOU:
            return best_i
        return -1

    def _merge_proposal_into_instance(
        self,
        fi: 'FusedInst',
        prop_u8: np.ndarray,
        conf: np.ndarray,
        alpha: float,
        diag: Optional[Dict[str, float]] = None,
        ) -> None:

        inst_bool = (fi.mask > 0)
        prop_bool = (prop_u8 > 0)

        union_bool = (inst_bool | prop_bool)
        union_area = int(np.count_nonzero(union_bool))
        if union_area <= 0:
            return

        # --- FAST BBOX UPDATE ---
        inst_bbox = fi.bbox_xyxy
        prop_bbox = self._bbox_from_mask(prop_u8)

        if prop_bbox is not None:
            x1 = min(inst_bbox[0], prop_bbox[0])
            y1 = min(inst_bbox[1], prop_bbox[1])
            x2 = max(inst_bbox[2], prop_bbox[2])
            y2 = max(inst_bbox[3], prop_bbox[3])
            fi.bbox_xyxy = (x1, y1, x2, y2)
        else:
            x1, y1, x2, y2 = inst_bbox

        # --- ROI-SCOPED DEPTH SUPPORT ---
        roi_union = union_bool[y1:y2+1, x1:x2+1]
        roi_conf  = conf[y1:y2+1, x1:x2+1]

        depth_support = float(roi_conf[roi_union].mean()) if roi_union.any() else float(fi.depth_support)

        # --- Commit mask & stats ---
        fi.mask = union_bool.astype(np.uint8)
        fi.area = union_area
        fi.depth_support = depth_support

        dc = max(0.0, min(1.0, depth_support))
        fi.score_fused = alpha * float(fi.score_rgb) + (1.0 - alpha) * dc

        # Depth assist indicator
        added = int(np.count_nonzero(prop_bool & (~inst_bool)))
        added_frac = float(added) / float(max(1, union_area))
        if hasattr(fi, "assist_depth_frac"):
            fi.assist_depth_frac = max(float(getattr(fi, "assist_depth_frac", 0.0)), added_frac)

        # Optional lightweight debug annotation
        if diag is not None and self._log_detail and MOE_DEBUG:
            print(f"[prop/merge] cls={fi.cls} added={added} ({added_frac:.2f}) "
                  f"overlap={diag.get('overlap_frac',0.0):.2f} touch={diag.get('touch_frac',0.0):.2f} iou={diag.get('iou',0.0):.2f}")

    @staticmethod
    def _nms_by_iou(fused_list: List['FusedInst'], iou_thr: float) -> List['FusedInst']:
        if len(fused_list) <= 1:
            return fused_list
        order = sorted(range(len(fused_list)), key=lambda i: fused_list[i].score_fused, reverse=True)
        keep: List[int] = []
        taken = np.zeros(len(fused_list), np.uint8)
        for i in order:
            if taken[i]:
                continue
            keep.append(i)
            mi = fused_list[i].mask
            for j in order:
                if j == i or taken[j]:
                    continue
                mj = fused_list[j].mask
                if Gate._mask_iou(mi, mj) >= iou_thr:
                    taken[j] = 1
        return [fused_list[k] for k in keep]

    def _make_inst_bufs(self):
        """One independent buffer set per parallel worker."""
        return {
            "tmp_mask640":       np.zeros((640, 640), np.float32),
            "depth_add_buf":     np.zeros((640, 640), dtype=bool),
            "comp_full_buf":     np.zeros((640, 640), dtype=bool),
            "clean_additions":   np.zeros((640, 640), dtype=bool),
            "mem_clean_buf":     np.zeros((640, 640), dtype=bool),
            "dil_buf":           np.empty((640, 640), dtype=np.uint8),
            "ring_buf":          np.empty((640, 640), dtype=np.uint8),
        }

    @staticmethod
    def _bbox_iou_xyxy(a: Tuple[int, int, int, int],
                       b: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1 + 1)
        ih = max(0, iy2 - iy1 + 1)
        if iw <= 0 or ih <= 0:
            return 0.0

        inter = float(iw * ih)
        area_a = float(max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1))
        area_b = float(max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1))
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return inter / denom

    @staticmethod
    def _mask_thinness(mask01: np.ndarray) -> float:
        m = (mask01 > 0).astype(np.uint8)
        if m.sum() == 0:
            return 1.0
        grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        per = float(grad.sum())
        if per <= 0:
            return 1.0
        area = float(m.sum())
        return np.clip(area / (per + 1e-6), 0.0, 1.0)

    @staticmethod
    def _largest_cc(mask01: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask01.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask01.astype(np.uint8)
        largest = max(contours, key=cv2.contourArea)
        out = np.zeros_like(mask01, dtype=np.uint8)
        cv2.drawContours(out, [largest], -1, 1, thickness=-1)
        return out

    def _fallback_rgb_only(self, gi: GateInput, thr_info: Dict[int, float]) -> List[FusedInst]:
        """
        Simple RGB-only fallback to avoid flicker:
        if fusion kills everything but YOLO still has detections,
        we build plain YOLO-led masks without depth gating.
        """
        fused: List[FusedInst] = []
        default_thr = self.fuse_thr

        for inst in gi.yolo_instances:
            cls = int(inst.cls)
            # pick threshold: class-specific > fuse_thr > 0
            if self.class_thr:
                cls_thr = self.class_thr.get(cls, default_thr)
            else:
                cls_thr = default_thr
            thr = thr_info.get(cls, cls_thr)

            score = float(inst.score)
            if score < thr:
                continue

            m_prob = inst.mask_prob.astype(np.float32)
            if m_prob.max() > 1.0:
                m_prob *= (1.0 / 255.0)
            m_prob = np.clip(m_prob, 0.0, 1.0, out=m_prob)
            m_bin = (m_prob >= 0.5).astype(np.uint8)
            area = int(m_bin.sum())
            if area < self.min_area:
                continue

            ys, xs = np.where(m_bin > 0)
            if xs.size == 0 or ys.size == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

            fused.append(FusedInst(
                mask=m_bin,
                cls=cls,
                score_rgb=score,
                score_fused=score,  # RGB-only score
                area=area,
                bbox_xyxy=(x1, y1, x2, y2),
                depth_support=0.0,
                mode="rgb-led",
                mean_prob_ds=score,   # cheap placeholder
                mean_depth_ds=0.0,
                assist_rgb_frac=1.0,
                assist_depth_frac=0.0,
            ))

        return fused
    
    def _apply_depth_sticky(self, fused: List[FusedInst]) -> List[FusedInst]:
        """
        Prevent depth-assisted fused masks from shrinking too quickly.

        For each current fused instance, if there is a recent cached instance
        with:
          - same class,
          - high bbox IoU,
          - larger area,
          - significant depth assistance,
        we expand the current mask to the union of current+previous masks,
        with a cap on area growth.

        This lets depth proposals "lock in" missing tool parts for a few frames
        instead of being immediately overwritten by YOLO.
        """
        if not fused or not self._inst_cache or self._inst_hold_max <= 0:
            return fused

        for fi in fused:
            # Only apply to real tools (YOLO-led or RGB+depth). Skip pure depth blobs (cls == -1).
            if getattr(fi, "cls", -1) < 0:
                continue

            best_prev = None
            best_age = None
            best_iou = 0.0

            # Find best matching cached instance for same class
            for prev, age in zip(self._inst_cache, self._inst_cache_age):
                if age >= self._inst_hold_max:
                    continue
                if getattr(prev, "cls", -1) != fi.cls:
                    continue

                iou = self._bbox_iou_xyxy(prev.bbox_xyxy, fi.bbox_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_prev = prev
                    best_age = age

            if best_prev is None:
                continue
            if best_iou < self._inst_hold_iou:
                # Not the same object / not stable enough
                continue

            # Require that the previous fused instance was actually depth-assisted
            if getattr(best_prev, "assist_depth_frac", 0.0) < self._sticky_min_depth_frac:
                continue

            # Only allow expansion (previous must be larger)
            if best_prev.area <= fi.area:
                continue

            prev_mask = (best_prev.mask > 0)
            cur_mask  = (fi.mask > 0)

            # Union of current + previous shapes
            union = (prev_mask | cur_mask)
            union_area = int(union.sum())

            if union_area <= fi.area:
                continue  # nothing to gain

            # Cap how much we allow the mask to grow relative to current
            if union_area > int(self._sticky_max_expand * max(1, fi.area)):
                continue

            ys, xs = np.where(union)
            if xs.size == 0 or ys.size == 0:
                continue

            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

            # Commit the expansion
            fi.mask = union.astype(np.uint8)
            fi.area = union_area
            fi.bbox_xyxy = (x1, y1, x2, y2)
            # Keep / increase the notion that depth helped
            if hasattr(fi, "assist_depth_frac"):
                fi.assist_depth_frac = max(
                    float(fi.assist_depth_frac),
                    float(best_prev.assist_depth_frac),
                )

        return fused

    def _apply_instance_hold(self,
                             gi: GateInput,
                             fused: List[FusedInst],
    ) -> Tuple[List[FusedInst], Dict[int, int]]:
        """
        If YOLO still sees a tool but the gate has dropped its fused instance
        on this frame, reuse the last fused instance for a few frames to avoid
        flicker. Returns (possibly-extended fused list, age_override map).
        """
        # If disabled or nothing cached, nothing to do
        if self._inst_hold_max <= 0 or not self._inst_cache:
            return fused, {}

        # Build current YOLO boxes (cls, box)
        yolo_boxes: List[Tuple[int, Tuple[int, int, int, int]]] = []
        for inst in gi.yolo_instances:
            bbox = getattr(inst, "bbox_xyxy", None)
            if bbox is None:
                continue
            try:
                cls_id = int(inst.cls)
            except Exception:
                cls_id = int(getattr(inst, "cls_id", 0))
            yolo_boxes.append((cls_id, bbox))

        if not yolo_boxes:
            # No YOLO detections: do not revive anything
            return fused, {}

        # Current fused boxes (cls, box)
        fused_boxes: List[Tuple[int, Tuple[int, int, int, int]]] = [
            (fi.cls, fi.bbox_xyxy) for fi in fused
        ]

        reuse_age: Dict[int, int] = {}

        # Try to reuse cached instances that:
        #  - are not too old
        #  - still have a matching YOLO box
        #  - do NOT have a corresponding fused instance this frame
        for prev, age in zip(self._inst_cache, self._inst_cache_age):
            if age >= self._inst_hold_max:
                continue

            # Check if YOLO still sees something at this location
            has_yolo = False
            for cls_id, y_box in yolo_boxes:
                if cls_id != prev.cls:
                    continue
                if self._bbox_iou_xyxy(prev.bbox_xyxy, y_box) >= self._inst_hold_iou:
                    has_yolo = True
                    break
            if not has_yolo:
                continue

            # Check if we already have a fused instance here this frame
            has_fused = False
            for cls_id, f_box in fused_boxes:
                if cls_id != prev.cls:
                    continue
                if self._bbox_iou_xyxy(prev.bbox_xyxy, f_box) >= self._inst_hold_iou:
                    has_fused = True
                    break
            if has_fused:
                continue

            # YOLO still sees it, gate dropped it -> reuse last fused instance
            fused.append(prev)
            fused_boxes.append((prev.cls, prev.bbox_xyxy))
            reuse_age[id(prev)] = age + 1

        return fused, reuse_age

    def _update_instance_cache(self,
                               fused: List[FusedInst],
                               age_override: Optional[Dict[int, int]] = None) -> None:
        """
        Refresh per-instance cache from current fused list.
        Instances that were reused get their incremented age;
        normally fused instances reset age to 0.
        """
        age_override = age_override or {}

        if not fused:
            self._inst_cache = []
            self._inst_cache_age = []
            return

        new_cache: List[FusedInst] = []
        new_age: List[int] = []

        for fi in fused:
            key = id(fi)
            if key in age_override:
                # This instance was resurrected; keep its updated age
                new_cache.append(fi)
                new_age.append(age_override[key])
                continue

            # Otherwise, it passed the gate this frame -> reset age
            new_cache.append(fi)
            new_age.append(0)

        self._inst_cache = new_cache
        self._inst_cache_age = new_age

    def _ring_contrast_delta(self, conf: np.ndarray, mask01: np.ndarray) -> float:
        m = (mask01 > 0).astype(np.uint8)
        if m.sum() == 0:
            return 0.0
        ys, xs = np.where(m > 0)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        y1 = max(0, y1 - 1); x1 = max(0, x1 - 1)
        y2 = min(conf.shape[0]-1, y2 + 1); x2 = min(conf.shape[1]-1, x2 + 1)

        roi_m    = m[y1:y2+1, x1:x2+1]
        roi_conf = conf[y1:y2+1, x1:x2+1]
        dil = cv2.dilate(roi_m, self._k7, iterations=1)
        ring = ((dil > 0) & (roi_m == 0))
        inside = float(roi_conf[roi_m > 0].mean()) if roi_m.any() else 0.0
        around = float(roi_conf[ring].mean())      if ring.any()   else inside
        return inside - around

    @staticmethod
    def _nms_by_iou_fast(fused_list: List['FusedInst'], iou_thr: float) -> List['FusedInst']:
        """Fast NMS using bounding-box IoU instead of full-mask IoU.

        This is much cheaper and good enough because fused instances already
        have reasonably tight boxes around their masks.
        """
        if len(fused_list) <= 1:
            return fused_list

        # Collect boxes and scores
        boxes = np.array([f.bbox_xyxy for f in fused_list], dtype=np.float32)
        scores = np.array([f.score_fused for f in fused_list], dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)

        order = np.argsort(-scores)  # highest score first
        keep: List[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1.0)
            h = np.maximum(0.0, yy2 - yy1 + 1.0)
            inter = w * h
            denom = (areas[i] + areas[order[1:]] - inter)
            iou = np.where(denom > 0.0, inter / denom, 0.0)

            # Keep boxes with IoU below threshold
            inds = np.where(iou < iou_thr)[0]
            order = order[inds + 1]

        return [fused_list[k] for k in keep]
    
    def _handle_thin_object_fast(self, inst, m_bin, ds_conf):
        """Optimized handler for thin objects"""
        m_keep = cv2.resize(m_bin.astype(np.uint8), (640, 640), interpolation=cv2.INTER_NEAREST)
        if m_keep.sum() < self.min_area:
            return None
            
        ys, xs = np.where(m_keep > 0)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        depth_support = float(ds_conf[m_keep > 0].mean()) if m_keep.any() else 0.0

        # Use precomputed stats where possible
        mean_prob_ds = 0.0  # Would need m_prob, but we don't have it in this context
        mean_depth_ds = float(ds_conf[m_bin > 0].mean()) if m_bin.any() else 0.0

        return FusedInst(
            mask=m_keep, cls=int(inst.cls),
            score_rgb=float(inst.score),
            score_fused=max(float(inst.score), 0.5),
            area=int(m_keep.sum()), bbox_xyxy=(x1, y1, x2, y2),
            depth_support=depth_support, mode="rgb-led",
            mean_prob_ds=mean_prob_ds, mean_depth_ds=mean_depth_ds,
            assist_rgb_frac=1.0, assist_depth_frac=0.0
        )

    def _ring_contrast_delta_ds(self, conf_ds: np.ndarray, mask01_ds: np.ndarray) -> float:
        m = (mask01_ds > 0).astype(np.uint8)
        s = int(m.sum())
        if s == 0:
            return 0.0
        k = getattr(self, "_k_ds3", cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        dil = cv2.dilate(m, k, iterations=1)
        ring = (dil > 0) & (m == 0)
        core = (m > 0)
        if not ring.any():
            return 0.0
        core_mean = float(conf_ds[core].mean()) if core.any() else 0.0
        ring_mean = float(conf_ds[ring].mean())
        return core_mean - ring_mean
    
    def _compute_mask_stats_batch(self, masks_bins, ds_conf):
        """Compute mask statistics in batch"""
        stats = []
        for m_bin in masks_bins:
            area_ds = int(m_bin.sum())
            if area_ds == 0:
                stats.append((0, 0, 0, 0.0, False))
                continue
                
            # Vectorized bounding box computation
            ys_ds, xs_ds = np.where(m_bin > 0)
            h_ds = int(ys_ds.max() - ys_ds.min() + 1) if ys_ds.size else 0
            w_ds = int(xs_ds.max() - xs_ds.min() + 1) if xs_ds.size else 0
            
            # Vectorized mean computation
            mask_pixels = ds_conf[m_bin > 0]
            mean_conf = float(mask_pixels.mean()) if mask_pixels.size > 0 else 0.0
            is_thin = (min(h_ds, w_ds) <= self._THIN_PX_AT_DS_FLOAT)
            
            stats.append((area_ds, h_ds, w_ds, mean_conf, is_thin))
        
        return stats

    def _process_yolo_masks_batch(self, gi: GateInput):
        """Process all YOLO masks in batch, respecting the same letterbox as depth."""
        masks_probs = []
        masks_bins = []
        union_ds = np.zeros((self.DS_H, self.DS_W), np.uint8)

        for inst in gi.yolo_instances:
            m = inst.mask_prob
            if m is None:
                continue

            # Ensure float32 in [0, 1]
            m = m.astype(np.float32, copy=False)
            if m.size == 0:
                continue
            if m.max() > 1.0:
                np.multiply(m, 1.0 / 255.0, out=m)

            if m.shape == (640,640):
                # Already in YOLO space, just copy into temp buffer
                self._tmp_mask640[...] = m
                m640 = self._tmp_mask640
            elif m.ndim == 2 and m.shape[0] == m.shape[1]:
                # Square mask (e.g. 160x160 from retina_masks=False) — YOLO grid space.
                # Resize directly to 640x640, do NOT letterbox as if it were a depth frame.
                cv2.resize(m, (640, 640), dst=self._tmp_mask640, interpolation=cv2.INTER_LINEAR)
                m640 = self._tmp_mask640
            else:
                # Non-square: genuine depth-camera-space frame → apply letterbox.
                m640 = depth_to_yolo_img(m, self._lb, is_mask=True, out=self._tmp_mask640)

            # 2) Downsample into the DS grid used by the gate
            cv2.resize(
                m640,
                (self.DS_W, self.DS_H),
                dst=self._ds_tmp_f32,
                interpolation=cv2.INTER_AREA,
            )
            masks_probs.append(self._ds_tmp_f32.copy())

            # 3) Threshold in-place to get binary masks
            np.greater_equal(self._ds_tmp_f32, YOLO_MASK_BIN_THR, out=self._ds_tmp_u8)
            masks_bins.append(self._ds_tmp_u8.copy())

            # 4) Accumulate union
            cv2.bitwise_or(union_ds, self._ds_tmp_u8, dst=union_ds)

        return masks_probs, masks_bins, union_ds
    
    # ----------------- Visualization -----------------
    def _heatmap01(self, x01: np.ndarray) -> np.ndarray:
        x8 = np.clip((x01 * 255.0).astype(np.uint8), 0, 255)
        return cv2.applyColorMap(x8, cv2.COLORMAP_JET)

    def _overlay_mask(self, base_bgr: np.ndarray, mask: np.ndarray, color=(0,255,0), alpha=0.5) -> np.ndarray:
        m01 = (mask > 0).astype(np.uint8)
        if m01.sum() == 0:
            return base_bgr
        ys, xs = np.where(m01 > 0)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        roi_base = base_bgr[y1:y2+1, x1:x2+1]
        roi_mask = (m01[y1:y2+1, x1:x2+1] * 255)
        color_layer = np.empty_like(roi_base)
        color_layer[:] = color
        blended = cv2.addWeighted(roi_base, 1.0 - alpha, color_layer, alpha, 0.0)
        out = base_bgr
        cv2.copyTo(blended, roi_mask, out[y1:y2+1, x1:x2+1])
        return out

    def _draw_boxes(self, img: np.ndarray, fused: List[FusedInst]):
        for f in fused:
            if getattr(f, "cls", -1) < 0 and os.getenv("MOE_VIZ_SHOW_PROPS", "1") == "0":
                continue
            x1,y1,x2,y2 = f.bbox_xyxy
            c = (0,255,0) if f.mode == "rgb-led" else (0,200,255)  # green vs teal
            cv2.rectangle(img, (x1,y1), (x2,y2), c, 2)
            label = f"cls={f.cls} m={f.mode} p={f.score_fused:.2f} d={f.depth_support:.2f}"
            cv2.putText(img, label, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235,235,235), 2, cv2.LINE_AA)

    def _stack3(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Stack depth, YOLO, and fused views horizontally with no blank tile."""
        h, w = a.shape[:2]

        # Keep a small downscale so viz stays cheap
        scale = 1.0
        nh, nw = int(h * scale), int(w * scale)

        a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA)
        c = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)

        # Just one row: [ depth | yolo | fused ]
        grid = np.hstack([a, b, c])
        return grid

    def _crop_letterbox_rows(self, img: np.ndarray) -> np.ndarray:
        """Crop out uniform grey letterbox bars (value≈114) from top/bottom for viz only.

        This runs *after* masks/boxes are drawn and never touches the data used
        for gating. It just makes the bottom panel easier to read.
        """
        if img is None or img.ndim != 3:
            return img
        h, w, _ = img.shape
        if h < 4:
            return img

        # Broker letterbox padding value (see _get_bgr_for_time).
        pad_val = 114.0
        # Work on first channel; compute per-row mean.
        row_mean = img[:, :, 0].mean(axis=1)
        mask = np.abs(row_mean - pad_val) > 1.0

        if not mask.any():
            # No obvious padding rows, keep as is.
            return img

        top = int(mask.argmax())
        bottom = int(h - mask[::-1].argmax())

        # Small safety margin so we don't cut too tight.
        top = max(0, top - 2)
        bottom = min(h, bottom + 2)

        if bottom - top < 8:
            # Degenerate, don't risk crazy crops.
            return img

        # Return a copy so later draws don't alias into the original buffer.
        return img[top:bottom, :, :].copy()

    def _render_view(self, gi: GateInput, out: 'GateOutput') -> np.ndarray:
        depth_vis = self._heatmap01(gi.conf_yolo)

        acc = self._acc_mask640
        acc[...] = 0.0

        for inst in gi.yolo_instances:
            m = inst.mask_prob
            if m is None:
                continue

            m = m.astype(np.float32, copy=False)
            if m.size == 0:
                continue
            if m.max() > 1.0:
                m *= (1.0 / 255.0)

            # Map mask into the same 640×640 canvas as depth, using the SAME letterbox
            if m.shape == (640, 640):
                # Already in YOLO canvas
                self._tmp_mask640[...] = m
                m640 = self._tmp_mask640
            elif m.ndim == 2 and m.shape[0] == m.shape[1]:
                # Square mask (e.g. 160x160) — YOLO grid space, resize directly.
                cv2.resize(m, (640, 640), dst=self._tmp_mask640, interpolation=cv2.INTER_LINEAR)
                m640 = self._tmp_mask640
            else:
                m640 = depth_to_yolo_img(
                    m,
                    self._lb,
                    is_mask=True,
                    out=self._tmp_mask640,
                )

            # Normalize to (640,640) — defensive guard for any mask resolution
            if m640.shape != (640, 640):
                m640 = cv2.resize(m640.astype(np.float32), (640, 640),
                                  interpolation=cv2.INTER_LINEAR)

            # Accumulate probabilities (clipped to [0,1])
            cv2.add(acc, np.clip(m640, 0.0, 1.0, out=self._tmp_mask640), acc)

        yolo_comp = np.clip(acc, 0.0, 1.0, out=acc)
        yolo_vis  = self._heatmap01(yolo_comp) if self.visualize else yolo_comp

        base = gi.bgr_yolo if gi.bgr_yolo is not None else depth_vis
        # Ensure base is (640,640,3) — bgr_yolo might come in at camera resolution
        if base is not None and base.shape[:2] != (640, 640):
            base = cv2.resize(base, (640, 640), interpolation=cv2.INTER_LINEAR)
        fused_mask = np.zeros((640,640), np.uint8)
        for f in out.fused:
            m = (f.mask > 0).astype(np.uint8)
            if m.shape != (640, 640):
                m = cv2.resize(m, (640, 640), interpolation=cv2.INTER_NEAREST)
            fused_mask |= m
        fused_vis = self._overlay_mask(base, fused_mask, color=(0,255,0), alpha=0.45)
        if gi.bgr_yolo is not None:
            self._draw_boxes(fused_vis, out.fused)
        
        # Keep fused image as-is; don't crop letterbox rows.
        fused_vis_viz = fused_vis

        now = cv2.getTickCount() / cv2.getTickFrequency()
        self._tbuf.append(now)
        if len(self._tbuf) >= 2:
            span = self._tbuf[-1] - self._tbuf[0]
            if span > 0:
                self._last_fps = (len(self._tbuf) - 1) / span

        grid = self._stack3(depth_vis, yolo_vis, fused_vis_viz)
        hud = [
            f"FPS={self._last_fps:4.1f}",
            f"plane_ok={out.health.plane_ok}  r={out.health.inlier_ratio:.2f}",
            f"fused={len(out.fused)}  thr={out.fuse_thr_info or {'*': FUSE_THR_DEFAULT}}",
            "mode: green=RGB-led, teal=Depth-led",
        ]
        y = 24
        for line in hud:
            cv2.putText(grid, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2, cv2.LINE_AA)
            y += 26
        return grid