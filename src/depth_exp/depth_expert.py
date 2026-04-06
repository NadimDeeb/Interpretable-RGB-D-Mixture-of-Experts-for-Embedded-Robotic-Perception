#!/usr/bin/env python3
# Depth Expert (FAST/TURBO): GPU RANSAC + confidence map with bounded work per frame
# - Keeps your original logic, fixes syntax/indent issues, and adds robust guards.
# - Decimates depth early, builds small plane mask, samples on GPU, runs RANSAC on GPU.
# - Dynamic iteration budget adjusts using inlier ratio EMA.
# - Optional color sharing path for YOLO re-use is preserved.

import os, time, threading
from collections import deque
from queue import Full, Empty

import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d

import cupy as cp
from cupyx.scipy.ndimage import zoom as cp_zoom

MOE_DEBUG = os.getenv("MOE_DEBUG", "0") not in ("0", "", "false", "False")
MOE_PROF = os.getenv("MOE_PROF", "0") not in ("0", "", "false", "False")

_pub_cnt = 0
_pub_t0  = time.perf_counter()

# -----------------------
# Env knobs (single source of truth)
# -----------------------

MAX_RANSAC_POINTS   = int(os.getenv("MOE_MAX_RANSAC_POINTS", "1000"))
TIME_BUDGET_MS      = float(os.getenv("MOE_DEPTH_BUDGET_MS", "22.0"))
USE_ALIGN           = int(os.getenv("MOE_RS_ALIGN", "1")) == 1

# RANSAC iteration knobs (dynamic)
PLANE_ITERS_DEFAULT = int(os.getenv("MOE_PLANE_ITERS", "120"))
DYN_ITERS_MIN       = 80
DYN_ITERS_MAX       = 150
DYN_ITERS_STEP_DOWN = 5
DYN_ITERS_STEP_UP   = 15
GOOD_INLIER_RATIO   = 0.80

# Confidence shaping
CONF_SIGMOID_SPAN_M = 0.006
CONF_PLANE_WEIGHT   = 0.95
CONF_RANGE_WEIGHT   = 0.05
CONF_BUMP_WEIGHT    = 0.06

# Plane acceptance threshold for downstream - Lower is more accepoting (which is good for less RANSAc runs)
PLANE_MIN_INLIER    = float(os.getenv("MOE_PLANE_MIN_INLIER", "0.05"))

# Base resolution
W, H, FPS = 640, 480, 30

# Bag replay support — set MOE_REPLAY_BAG to path of .bag file
REPLAY_BAG = os.getenv("MOE_REPLAY_BAG", "")

# run confidence maps in FP16 to save bandwidth.
# 0 (default) -> float32, 1 -> float16
CONF_FP16  = os.getenv("MOE_CONF_FP16", "0") in ("1", "true", "True")
CONF_DTYPE = cp.float16 if CONF_FP16 else cp.float32

# Mode presets (kept from your original structure)
MODE = "room"      # "tabletop" or "room"
SCALE_PLANE      = float(os.getenv("MOE_SCALE_PLANE", "0.15"))    # small plane-res relative to the (decimated) depth frame
USE_COLOR    = True    # share aligned color to YOLO if provided

def get_params(mode: str):
    if mode == "tabletop":
        return dict(
            NEAR_M=0.10, FAR_M=1.0, CLIP_M=1.0,
            RANSAC_ITERS=PLANE_ITERS_DEFAULT, RANSAC_TAU_M=0.004,
            ROI_FRAC=0.20, FAR_PERCENTILE=85,
            TAU_ON=0.010, TAU_OFF=0.002,
            CONF_SPAN=0.030,
            EDGE_SIGMA_PX=1.2, EDGE_THR_MM=0.50,
            MAX_POINTS_FOR_RANSAC=MAX_RANSAC_POINTS
        )
    else:
        return dict(
            NEAR_M=0.30, FAR_M=2.0, CLIP_M=1.0,
            RANSAC_ITERS=PLANE_ITERS_DEFAULT, RANSAC_TAU_M=0.010,
            ROI_FRAC=0.10, FAR_PERCENTILE=80,
            TAU_ON=0.020, TAU_OFF=0.015,
            CONF_SPAN=0.050,
            EDGE_SIGMA_PX=1.2, EDGE_THR_MM=2.0,
            MAX_POINTS_FOR_RANSAC=MAX_RANSAC_POINTS
        )

P = get_params(MODE)

# -----------------------
# Helpers
# -----------------------
def height_above_plane_fast(depth_m, valid, rx, ry, n, p0):
    # depth_m, valid, rx, ry, n, p0 are CuPy
    Z  = depth_m
    # Signed distance of each pixel point to plane (in meters)
    out = rx * Z - p0[0]
    out = out * n[0]
    out = out + (ry * Z - p0[1]) * n[1]
    out = out + (Z - p0[2]) * n[2]
    out = cp.where(valid, out, cp.nan)
    # Flip so "up" is positive
    if cp.nanmean(out) < 0:
        out = -out
    return out

def _log_cupy_device_once():
    try:
        n = cp.cuda.runtime.getDeviceCount()
        print(f"[depth] CuPy devices: {n}")
        if n == 0:
            print("[depth][ERROR] No CUDA devices visible to CuPy.")
        else:
            d = cp.cuda.runtime.getDevice()
            props = cp.cuda.runtime.getDeviceProperties(d)
            print(f"[depth] Using GPU {d}: {props['name'].decode()} "
                  f"(CC {props['major']}.{props['minor']}, "
                  f"{props['totalGlobalMem']/ (1024**3):.1f} GB)")
    except Exception as e:
        print("[depth][ERROR] CuPy runtime error:", e)

def fast_downsample(depth_m, valid_bool, target_height, target_width):
    """Ultra-fast downsampling using integer strides - replaces cp_zoom"""
    H, W = depth_m.shape
    stride_y = H // target_height
    stride_x = W // target_width
    
    # Use fixed slicing - faster than dynamic computation
    depth_small = depth_m[::stride_y, ::stride_x]
    valid_small = valid_bool[::stride_y, ::stride_x]
    
    # Ensure exact target size
    return depth_small[:target_height, :target_width], valid_small[:target_height, :target_width]

def fast_edge_detection_simple(Z, threshold=0.0040):
    """Fast edge detection without Gaussian filtering"""
    # Simple horizontal differences (much faster than Sobel)
    gx = cp.abs(Z[:, 1:] - Z[:, :-1])
    gx = cp.pad(gx, ((0,0), (0,1)), mode='edge')
    
    # Simple vertical differences  
    gy = cp.abs(Z[1:, :] - Z[:-1, :])
    gy = cp.pad(gy, ((0,1), (0,0)), mode='edge')
    
    # Combined magnitude (approximate)
    mag = gx + gy
    return mag > threshold

_LOGGED = False
def _ensure_logged():
    global _LOGGED
    if not _LOGGED:
        _log_cupy_device_once()
        _LOGGED = True

def ransac_plane_fast(depth_m, mask, intr, iters=25, distance_threshold=0.004):
    """Ultra-fast Open3D RANSAC with aggressive optimizations"""
    ys, xs = mask.nonzero()
    if ys.size < 50:  # Very low minimum points for speed
        return None, None, 0
    
    # Convert from GPU to CPU
    depth_cpu = depth_m.get() if hasattr(depth_m, 'get') else depth_m
    ys_cpu, xs_cpu = ys.get() if hasattr(ys, 'get') else ys, xs.get() if hasattr(xs, 'get') else xs
    
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    
    # Convert to 3D points
    z_vals = depth_cpu[ys_cpu, xs_cpu].astype(np.float32)
    X_vals = (xs_cpu.astype(np.float32) - cx) / fx * z_vals
    Y_vals = (ys_cpu.astype(np.float32) - cy) / fy * z_vals
    
    # Create point cloud
    points = np.column_stack((X_vals, Y_vals, z_vals))
    
    # Remove NaN and infinite values
    valid_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
    points = points[valid_mask]
    
    if points.shape[0] < 30:  # Very low threshold
        return None, None, 0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Run Open3D RANSAC with fewer iterations
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=iters
        )
        
        inlier_count = len(inliers)
        if inlier_count < 30:  # Very low minimum inliers
            return None, None, inlier_count
            
        # Extract plane parameters
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal_norm = np.linalg.norm(normal)
        
        if normal_norm < 1e-8:
            return None, None, inlier_count
            
        normal = normal / normal_norm
        
        # Compute centroid of inliers
        inlier_points = np.asarray(pcd.points)[inliers]
        centroid = inlier_points.mean(axis=0)
        
        # Convert back to CuPy for GPU processing
        normal_gpu = cp.asarray(normal, dtype=cp.float32)
        centroid_gpu = cp.asarray(centroid, dtype=cp.float32)
        
        return normal_gpu, centroid_gpu, inlier_count
        
    except Exception as e:
        print(f"[ransac] Open3D RANSAC error: {e}")
        return None, None, 0

# -----------------------
# CUDA Stream for Async Operations  
# -----------------------
class AsyncStreamManager:
    def __init__(self):
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.timing_data = {}
    
    def start_timer(self, name):
        self.timing_data[name] = time.perf_counter()
    
    def end_timer(self, name):
        if name in self.timing_data:
            elapsed = (time.perf_counter() - self.timing_data[name]) * 1000
            self.timing_data[name] = elapsed
            return elapsed
        return 0.0

# -----------------------
# Pinned2x 
# -----------------------
class _Pinned2x:
    """Two pinned buffers for a fixed shape/dtype, with a rolling generation id."""
    def __init__(self, shape, dtype=np.float32):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self._mem = [cp.cuda.alloc_pinned_memory(nbytes),
                     cp.cuda.alloc_pinned_memory(nbytes)]
        self._arr = [np.ndarray(shape, dtype=dtype, buffer=self._mem[0]),
                     np.ndarray(shape, dtype=dtype, buffer=self._mem[1])]
        self._idx = 0
        self._gen = 0
    def next(self):
        self._idx ^= 1
        self._gen += 1
        return self._arr[self._idx], self._gen, self._idx

# -----------------------
# Plane Stabilizer
# -----------------------
class PlaneStabilizer:
    def __init__(self, alpha=0.7, min_continuity=3):
        self.alpha = alpha
        self.continuity = 0
        self.last_good_plane = (None, None)
        self.min_continuity = min_continuity
        
    def update(self, n, p0, inlier_ratio):
        if n is None or inlier_ratio < PLANE_MIN_INLIER:
            self.continuity = 0
            return self.last_good_plane
            
        self.continuity += 1
        if self.continuity < self.min_continuity:
            return self.last_good_plane
            
        # Smooth plane parameters
        if self.last_good_plane[0] is not None:
            n_smooth = self.alpha * self.last_good_plane[0] + (1 - self.alpha) * n
            p0_smooth = self.alpha * self.last_good_plane[1] + (1 - self.alpha) * p0
            # Renormalize
            n_norm = cp.linalg.norm(n_smooth)
            if n_norm > 1e-8:
                n_smooth = n_smooth / n_norm
                self.last_good_plane = (n_smooth, p0_smooth)
        else:
            self.last_good_plane = (n, p0)
            
        return self.last_good_plane

# -----------------------
# Plane Tracker
# -----------------------
class PlaneTracker:
    """
    Fast per-frame plane tracking using weighted least squares on a sparse grid.
    Runs every frame. Async RANSAC is used only to (re)initialize.
    """
    def __init__(self,
                 tau_inlier_m=0.02,
                 min_pts=200,
                 huber_k=0.03,
                 irls_iters=1):
        self.tau = float(tau_inlier_m)
        self.min_pts = int(min_pts)
        self.huber_k = float(huber_k)
        self.irls_iters = int(irls_iters)
        self.n = None        # (3,)
        self.p0 = None       # (3,)
        self._prev_n = None

    def reset(self, n, p0):
        # Expect n normalized
        self.n = n.astype(cp.float32, copy=True)
        self.p0 = p0.astype(cp.float32, copy=True)
        self._prev_n = self.n.copy()

    def _fit_ls(self, P, w=None):
        # P: (N,3) cupy float32
        if w is None:
            mu = P.mean(axis=0)
            X = P - mu
            C = (X.T @ X) / max(1, P.shape[0])
        else:
            w = w.reshape(-1, 1)
            ws = w.sum()
            if ws <= 0:
                return None, None
            mu = (w * P).sum(axis=0) / ws
            X = P - mu
            C = (X * w).T @ X / ws
        # smallest eigenvector
        vals, vecs = cp.linalg.eigh(C)
        n = vecs[:, 0]
        n = n / (cp.linalg.norm(n) + 1e-8)
        d = -cp.dot(n, mu)
        return n, d

    def update(self, depth_small, mask_small, rx_s, ry_s):
        """
        depth_small: (ph,pw) meters, cupy
        mask_small:  (ph,pw) bool/uint8, cupy
        rx_s, ry_s:  (ph,pw) normalized rays for small grid
        Returns: ok(bool), n, p0, inlier_ratio, mad
        """
        m = mask_small.astype(cp.bool_)
        if m.sum() < self.min_pts:
            return False, None, None, 0.0, 0.0

        Z = depth_small[m]
        X = rx_s[m] * Z
        Y = ry_s[m] * Z
        P = cp.stack((X, Y, Z), axis=1)  # (N,3)

        # Initial LS
        n, d = self._fit_ls(P)
        if n is None:
            return False, None, None, 0.0, 0.0

        # Optional IRLS (1 iteration)
        for _ in range(self.irls_iters):
            r = P @ n + d
            ar = cp.abs(r)
            # Huber weights
            w = cp.where(ar <= self.huber_k, 1.0, self.huber_k / (ar + 1e-8))
            n, d = self._fit_ls(P, w=w)
            if n is None:
                break

        # Residual stats
        r = P @ n + d
        ar = cp.abs(r)
        inlier = ar < self.tau
        inlier_ratio = float(inlier.mean())
        mad = float(cp.median(ar))

        p0 = -d * n
        self._prev_n = n.copy()
        self.n = n
        self.p0 = p0
        return True, n, p0, inlier_ratio, mad

# -----------------------
# Async plane worker (decouple RANSAC from publishing)
# -----------------------
class _AsyncPlaneWorker:
    """Single background worker that fits planes on a downsampled frame.
       Only the newest job is kept (latest-only queue)."""
    def __init__(self, ransac_fn, max_age_ms=0.0):
        import queue, threading, time
        self._q = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._alive = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._ransac_fn = ransac_fn
        self._max_age_ms = float(max_age_ms)
        self._last_fit_ms = 0.0
        self._ts_ms = 0.0

        # cached plane
        self._n = None
        self._p0 = None
        self._inlier_ratio = 0.0
        self._ts_ms = 0.0
        self._thread.start()

    def stop(self):
        self._alive = False
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
    
    def last_age_ms(self):
        now_ms = time.monotonic() * 1000.0
        with self._lock:
            ts = getattr(self, "_ts_ms", 0.0)
            return None if (self._n is None or ts == 0.0) else (now_ms - ts)

    def last_fit_ms(self):
        with self._lock:
            return float(self._last_fit_ms)

    def submit_latest(self, depth_small_m, sample_mask, support_mask, intr_small, iters, tau_m):
        """Submit newest small job; drop older one if busy.
           sample_mask   -> where we actually sample points for RANSAC
           support_mask  -> full ROI support used for inlier-ratio denominator
        """
        job = (
            depth_small_m.copy(),
            sample_mask.copy(),
            support_mask.copy(),
            intr_small,
            int(iters),
            float(tau_m),
        )
        try:
            if self._q.full():
                try:
                    self._q.get_nowait()
                except Exception:
                    pass
            self._q.put_nowait(job)
        except Exception:
            pass

    def get_cached_plane(self):
        """Return (n, p0, inlier_ratio, is_fresh)."""
        now_ms = time.monotonic() * 1000.0
        with self._lock:
            n, p0 = self._n, self._p0
            inl, ts = self._inlier_ratio, self._ts_ms
        is_fresh = (n is not None) and ((now_ms - ts) <= self._max_age_ms)
        return n, p0, inl, is_fresh

    def _loop(self):
        import time
        while self._alive:
            try:
                job = self._q.get(timeout=0.25)
            except Exception:
                continue
            if (not self._alive) or (job is None):
                continue
            depth_small_m, sample_mask, support_mask, intr_small, iters, tau_m = job
            try:
                t0 = time.perf_counter_ns()
                # RANSAC sees only the candidate band (sample_mask)...
                n, p0, inlier_cnt = self._ransac_fn(
                    depth_small_m, sample_mask, intr_small,
                    iters=iters, distance_threshold=tau_m
                )
                t1 = time.perf_counter_ns()
                fit_ms = (t1 - t0) / 1e6
                if n is None:
                    continue

                # ...but the inlier ratio is measured vs the full ROI support.
                support_pts = int(support_mask.sum())
                ratio = float(inlier_cnt) / max(1, support_pts)

                with self._lock:
                    self._n = n
                    self._p0 = p0
                    self._inlier_ratio = ratio
                    self._ts_ms = time.monotonic() * 1000.0
                    self._last_fit_ms = float(fit_ms)
                if MOE_PROF:
                    print(f"[plane/fit] inlier_ratio={ratio:.2f} pts={support_pts}")
            except Exception as e:
                # print(f"[plane/worker] error: {e}")
                pass

# -----------------------
# Headless generator for run_moe.py
# -----------------------
def stream_depth(share_color=None, share_color_ts=None):
    global _pub_cnt, _pub_t0
    """
    Yields: (conf_480x640_float32, plane_ok_bool, inlier_ratio_float)
    If share_color is a queue(maxsize=1), pushes latest aligned BGR frames into it.
    """
    # Async capture thread pipes frames here (to decouple viz/YOLO from depth compute)
    local_buf = deque(maxlen=1)
    local_stop = False

    def _grabber(pipeline, align_obj):
        nonlocal local_stop
        _prof_cnt = 0
        _prof_t0  = time.perf_counter()
        _grab_frame = 0
        while not local_stop:
            t0 = time.time()
            frames = pipeline.wait_for_frames()
            t1 = time.time()
            t_wait = t1 - t0  # seconds

            if USE_ALIGN and align_obj is not None:
                frames = align_obj.process(frames)

            t2 = time.time()
            t_align = t2 - t1  # seconds

            # Insert this small profiler print:
            if MOE_PROF and (_grab_frame % 30) == 0:
                print(f"[depth/prof] wait={(t_wait*1000):.1f}ms align={(t_align*1000):.1f}ms")
            _grab_frame += 1

            # publish color for YOLO
            if USE_COLOR and share_color is not None:
                c = frames.get_color_frame()
                if c:
                    bgr = np.asanyarray(c.get_data())
                    # Bags record as RGB — convert to BGR for YOLO/OpenCV
                    if REPLAY_BAG:
                        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
                    try:
                        # Safely handle race between full() and get_nowait()
                        try:
                            if share_color.full():
                                share_color.get_nowait()
                        except Empty:
                            pass
                        share_color.put_nowait(bgr)
                    except Full:
                        # Drop oldest and retry once
                        try:
                            share_color.get_nowait()
                        except Empty:
                            pass
                        try:
                            share_color.put_nowait(bgr)
                        except Full:
                            pass
                    if MOE_PROF:
                        _prof_cnt += 1
                        if (_prof_cnt % 60) == 0:
                            _t1 = time.perf_counter()
                            fps = 60.0 / max(1e-6, (_t1 - _prof_t0))
                            print(f"[prof/color] {fps:.1f} FPS from camera")
                            _prof_t0 = _t1

            local_buf.append(frames)

    # ---- RealSense setup ----
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    if USE_COLOR:
        config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

    # ---- Bag replay mode ----
    if REPLAY_BAG:
        rs.config.enable_device_from_file(config, REPLAY_BAG, repeat_playback=False)
        config.enable_stream(rs.stream.depth)
        config.enable_stream(rs.stream.color)
        print(f"[depth] Replay mode: {REPLAY_BAG}")

    profile  = pipeline.start(config)

    # ---- Bag replay: enforce real-time rate and wait for YOLO to be ready ----
    if REPLAY_BAG:
        try:
            import pathlib
            playback_dev = profile.get_device().as_playback()
            playback_dev.set_real_time(True)   # play at 30fps, same as live camera
            playback_dev.pause()
            print("[depth] Bag paused — waiting for YOLO ready signal...")
            ready_flag = pathlib.Path("/tmp/moe_yolo_ready.flag")
            ready_flag.unlink(missing_ok=True)
            while not ready_flag.exists():
                time.sleep(0.05)
            playback_dev.resume()
            ready_flag.unlink(missing_ok=True)
            print("[depth] YOLO ready — bag resumed at real-time rate.")
        except Exception as e:
            print(f"[depth] Bag pause/resume failed: {e} — continuing anyway")

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color) if USE_ALIGN else None
    try:
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
        if depth_sensor.supports(rs.option.laser_power):
            rng = depth_sensor.get_option_range(rs.option.laser_power)
            depth_sensor.set_option(rs.option.laser_power, min(360.0, rng.max))
    except Exception:
        pass

    # Light filter chain
    decim    = rs.decimation_filter()
    decim.set_option(rs.option.filter_magnitude, 2)
    spatial  = rs.spatial_filter();  spatial.set_option(rs.option.holes_fill, 1)
    holefill = rs.hole_filling_filter()

    # Determine decimated size and precompute intrinsics/grids
    frames0 = pipeline.wait_for_frames()
    if align is not None:
        frames0 = align.process(frames0)
    d0 = frames0.get_depth_frame()
    d0 = decim.process(d0)
    d0 = spatial.process(d0)
    d0 = holefill.process(d0)
    depth0 = np.asanyarray(d0.get_data())
    Hd, Wd = int(depth0.shape[0]), int(depth0.shape[1])

    # === Option 1: DS geometry MUST be defined before use ===
    CONF_DS = int(os.getenv("MOE_CONF_DOWNSAMPLE", "192"))
    H_DS = min(CONF_DS, Hd)
    W_DS = min(CONF_DS, Wd)
    _conf_ds_gpu = cp.empty((H_DS, W_DS), dtype=CONF_DTYPE)

   # --- Option 1: true depth->color geometry (no rs.align) ---
    depth_prof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr_depth = depth_prof.get_intrinsics()

    if USE_COLOR:
        color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr_color = color_prof.get_intrinsics()
        extr_d2c   = depth_prof.get_extrinsics_to(color_prof)
    else:
        intr_color = None
        extr_d2c   = None

    # Depth intrinsics scaled to decimated depth size (Hd x Wd)
    scale_w = Wd / float(W)
    scale_h = Hd / float(H)
    fx_d  = intr_depth.fx  * scale_w
    fy_d  = intr_depth.fy  * scale_h
    ppx_d = intr_depth.ppx * scale_w
    ppy_d = intr_depth.ppy * scale_h

    # Color intrinsics (color stream stays 640x480)
    if intr_color is not None:
        fx_c, fy_c = float(intr_color.fx),  float(intr_color.fy)
        ppx_c, ppy_c = float(intr_color.ppx), float(intr_color.ppy)
        Wc, Hc = int(intr_color.width), int(intr_color.height)

        R = np.array(extr_d2c.rotation, dtype=np.float32).reshape(3, 3)
        t = np.array(extr_d2c.translation, dtype=np.float32)
        R_gpu = cp.asarray(R)
        t_gpu = cp.asarray(t)
    else:
        R_gpu = None
        t_gpu = None

    ys_d, xs_d = cp.mgrid[0:Hd, 0:Wd].astype(cp.float32)
    rx = (xs_d - ppx_d) / fx_d
    ry = (ys_d - ppy_d) / fy_d

    # --- Option 1: low-res reprojection sampling grid (about 128x128 points) ---
    sh = max(1, Hd // H_DS)
    sw = max(1, Wd // W_DS)

    ys_s, xs_s = cp.mgrid[0:Hd:sh, 0:Wd:sw].astype(cp.int32)
    ys_s = ys_s.ravel()
    xs_s = xs_s.ravel()

    rx_s = rx[ys_s, xs_s]
    ry_s = ry[ys_s, xs_s]

    # --- Option 1: color->YOLO letterbox constants (640x480 -> 640x640) ---
    # For D435i color 640x480, scale=1 and pad_y=80, but compute generically:
    yolo_sz = 640.0
    scale_c2y = min(yolo_sz / float(Wc), yolo_sz / float(Hc))
    pad_x = (yolo_sz - float(Wc) * scale_c2y) * 0.5
    pad_y = (yolo_sz - float(Hc) * scale_c2y) * 0.5

    # Small plane-res (in decimated domain)
    pw, ph = int(Wd * SCALE_PLANE), int(Hd * SCALE_PLANE)
    intr_small = rs.intrinsics()
    intr_small.width, intr_small.height = pw, ph
    sxs, sys = pw / float(Wd), ph / float(Hd)
    intr_small.ppx = ppx_d * sxs
    intr_small.ppy = ppy_d * sys
    intr_small.fx  = fx_d  * sxs
    intr_small.fy  = fy_d  * sys

    # Static ROI and border mask at small size
    rf = P["ROI_FRAC"]
    Hc0, Hc1 = int(ph*rf), int(ph*(1-rf))
    Wc0, Wc1 = int(pw*rf), int(pw*(1-rf))
    roi_small_static = np.zeros((ph, pw), np.bool_)
    roi_small_static[Hc0:Hc1, Wc0:Wc1] = True
    bf = 0.035
    bx, by = int(pw*bf), int(ph*bf)
    border_mask = np.ones((ph, pw), np.bool_)
    border_mask[:, :bx]  = False; border_mask[:, -bx:] = False
    border_mask[:by, :]  = False; border_mask[-by:, :] = False
    roi_border_static = cp.asarray(roi_small_static & border_mask)

    # === NEW: integer-stride downsampling helpers + preallocs (GPU) ===
    k_y = max(1, int(round(1.0 / SCALE_PLANE)))
    k_x = max(1, int(round(1.0 / SCALE_PLANE)))
    ph, pw = Hd // k_y, Wd // k_x  # ensure integer grid from decimated size

    # Rebuild intr_small to match the stride-decimated size (ph, pw)
    intr_small = rs.intrinsics()
    intr_small.width, intr_small.height = pw, ph
    sxs, sys = pw / float(Wd), ph / float(Hd)
    intr_small.ppx = ppx_d * sxs
    intr_small.ppy = ppy_d * sys
    intr_small.fx  = fx_d  * sxs
    intr_small.fy  = fy_d  * sys

    # Persistent buffers
    depth_m         = cp.empty((Hd, Wd), dtype=cp.float32)
    valid_bool      = cp.empty((Hd, Wd), dtype=cp.bool_)
    conf_bump_full  = cp.empty((Hd, Wd), dtype=CONF_DTYPE)
    conf_plane_full = cp.empty((Hd, Wd), dtype=CONF_DTYPE)
    conf_range_full = cp.empty((Hd, Wd), dtype=CONF_DTYPE)
    in_range_full   = cp.empty((Hd, Wd), dtype=cp.bool_)
    depth_norm      = cp.empty((Hd, Wd), dtype=CONF_DTYPE)
    height_map      = cp.empty((Hd, Wd), dtype=CONF_DTYPE)

    CONF_HOLD_FRAMES = int(os.getenv("MOE_CONF_HOLD_FRAMES", "3"))
    _last_conf_ds_gpu = cp.empty((H_DS, W_DS), dtype=CONF_DTYPE)
    _have_last_conf = False
    _last_conf_age = 0

    # double-buffered pinned host storage for the *small* map (float32)
    _conf_buf = _Pinned2x((H_DS, W_DS), np.float32)

    frame_id = 0
    _poor_ransac_streak = 0
    RANSAC_SKIP_AFTER   = 30   # frames of bad inlier ratio before skipping
    RANSAC_REPROBE_EVERY = 60  # re-attempt every N frames even when skipping
    dynamic_iters = int(np.clip(int(os.getenv("MOE_RANSAC_ITERS", str(P["RANSAC_ITERS"]))),
                            DYN_ITERS_MIN, DYN_ITERS_MAX))

    # async plane worker (keeps latest job only)
    plane_worker = _AsyncPlaneWorker(
        ransac_fn=ransac_plane_fast,
        max_age_ms=float(os.getenv("MOE_PLANE_MAX_AGE_MS", "200.0")) # To reuse  planes -----------------------
    )

    # plane refit cadence (every N frames)
    PLANE_BG_EVERY = int(os.getenv("MOE_PLANE_BG_EVERY", "1"))

    # Initialize async stream manager and pinned memory pool
    if not hasattr(stream_depth, 'stream_mgr'):
        stream_depth.stream_mgr = AsyncStreamManager()
    
    stream_mgr = stream_depth.stream_mgr

    # Start capture thread
    t_cap = threading.Thread(target=_grabber, args=(pipeline, align), daemon=True)
    t_cap.start()

    try:
        _conf_ema = None
        _EMA_ALPHA = float(os.getenv("MOE_CONF_EMA", "0.35")) 
        while True:
            if not local_buf:
                continue
            aligned = local_buf[-1]

            # Reset timing for this frame
            stream_mgr.timing_data.clear()
            stream_mgr.start_timer('total_frame')
            
            stream_mgr.start_timer('filters')
            d = aligned.get_depth_frame()
            if USE_COLOR:
                c = aligned.get_color_frame()
                if not d or not c:
                    continue
            else:
                if not d:
                    continue
            bgr_aligned = np.asanyarray(c.get_data())
            # Bags record as RGB — convert to BGR for YOLO/OpenCV
            if REPLAY_BAG:
                bgr_aligned = cv2.cvtColor(bgr_aligned, cv2.COLOR_RGB2BGR)
            
            # Check for cupy to be using correct GPU
            _ensure_logged()

            # Filters
            f = decim.process(d)
            f = spatial.process(f)
            f = holefill.process(f)
            filters_ms = stream_mgr.end_timer('filters')

            if MOE_DEBUG and MOE_PROF and (frame_id % 30 == 0):
                print("[prof/depth] filters done")
            
            # Depth to GPU + validity WITH STREAM (u16→f32 fused)
            stream_mgr.start_timer('h2d_transfer')
            with stream_mgr.stream:
                # 1) Bring raw depth as uint16 (1/2 the bytes vs f32)
                depth_u16 = cp.asarray(np.asanyarray(f.get_data()), dtype=cp.uint16)

                # 2) valid_bool = depth_u16 > 0  (no cast needed)
                cp.greater(depth_u16, 0, out=valid_bool)

                # 3) depth_m = depth_u16 * depth_scale  (write final f32 directly into depth_m)
                cp.multiply(depth_u16, depth_scale, out=depth_m, dtype=cp.float32)
            h2d_ms = stream_mgr.end_timer('h2d_transfer')

            if MOE_PROF and (frame_id % 30 == 0):
                print("[prof/depth] depth to GPU done")

            # Downsampling with timing
            stream_mgr.start_timer('downsampling')
            depth_m_small, valid_small = fast_downsample(depth_m, valid_bool, ph, pw)
            downsample_ms = stream_mgr.end_timer('downsampling')

            # ROI processing with timing
            stream_mgr.start_timer('roi_processing')

            # === compute validity on the small grid directly ===
            in_rng_small = (valid_small &
                            (depth_m_small >= P["NEAR_M"]) &
                            (depth_m_small <= P["FAR_M"]))
            roi_small = in_rng_small & roi_border_static

            # Support mask for RANSAC denominator: all valid depth in ROI,
            # regardless of NEAR/FAR. This prevents a tiny foreground blob from becoming "the plane".
            plane_support_small = valid_small & roi_border_static

            # --- Histogram-based masked percentiles (GPU-only, no gathers, no D2H) ---
            # Quantize depths within a sane range to B bins, then bincount with mask.
            B = 256  # small, fast; tune if needed
            dmin = 0.10  # meters  (keep consistent with your pipeline)
            dmax = 3.00  # meters

            # clamp & quantize (elementwise) without compressing to 1-D
            depth_q = cp.floor(
                cp.clip((depth_m_small - dmin) / (dmax - dmin), 0.0, 1.0) * (B - 1)
            ).astype(cp.int32, copy=False)

            # weighted histogram: only ROI contributes (mask as weights 0/1)
            w = roi_small.astype(cp.int32, copy=False)
            hist = cp.bincount(depth_q.ravel(), weights=w.ravel(), minlength=B)

            tot = hist.sum()
            if tot > 0:
                # compute cumulative
                cdf = cp.cumsum(hist)

                p_ctr = float(P["FAR_PERCENTILE"])  # e.g. 90
                band  = 80.0                        # you used ±80 in percent-space
                p_lo = max(50.0, p_ctr - band) * 0.01
                p_hi = min(99.5, p_ctr + band) * 0.01

                i_lo = int(cp.searchsorted(cdf, tot * p_lo))
                i_hi = int(cp.searchsorted(cdf, tot * p_hi))

                far_lo = dmin + (dmax - dmin) * (i_lo / max(1, B - 1))
                far_hi = dmin + (dmax - dmin) * (i_hi / max(1, B - 1))

                # plane window entirely on-GPU; no host syncs
                plane_mask = roi_small & (depth_m_small >= far_lo) & (depth_m_small <= far_hi)
            else:
                plane_mask = roi_small

            roi_ms = stream_mgr.end_timer('roi_processing')

            # --- Robust edge suppression + fallback ---
            stream_mgr.start_timer('edge_detection')
            # Always start from plane_mask
            plane_mask2 = plane_mask

            # Only try edge removal if we have some candidates
            if plane_mask.sum() > 50:
                # GPU-only fast edge suppression (toggle via env)
                USE_FAST_EDGES = int(os.getenv("MOE_FAST_EDGES", "1")) == 1
                if USE_FAST_EDGES:
                    # Estimate local high-pass by first-order diffs (already on GPU)
                    edges_gpu = fast_edge_detection_simple(depth_m_small, 
                                                        threshold=(0.0015 if MODE=="tabletop" else 0.0025))
                    plane_mask2 = plane_mask & (~edges_gpu)
                    # Fallback if over-pruned
                    if plane_mask2.sum() < 300:
                        plane_mask2 = plane_mask
                else:
                    # Original CPU path (kept behind the toggle)
                    Z = cp.asnumpy(depth_m_small)
                    valid_np = cp.asnumpy(valid_small)
                    pm_np    = cp.asnumpy(plane_mask)
                    if valid_np.any():
                        med = float(np.median(Z[valid_np]))
                        Z[~valid_np] = med
                    sigma = float(P.get("EDGE_SIGMA_PX", 1.2))
                    low  = cv2.GaussianBlur(Z, (0, 0), sigmaX=sigma, sigmaY=sigma)
                    hp   = Z - low
                    gx = cv2.Sobel(hp, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(hp, cv2.CV_32F, 0, 1, ksize=3)
                    mag = np.sqrt(gx * gx + gy * gy)
                    thr_m = 0.0015 if MODE=="tabletop" else 0.0025
                    edges_np = (mag > thr_m)
                    pm2_np = np.logical_and(pm_np, ~edges_np)
                    if pm2_np.sum() < 300:
                        pm2_np = pm_np
                    plane_mask2 = cp.asarray(pm2_np, dtype=cp.bool_)
                if MOE_PROF and (frame_id % 30 == 0):
                    print(f"[depth/dbg] plane_mask2 sum={int(plane_mask2.sum())} phxpw={ph}x{pw}")

            edge_ms = stream_mgr.end_timer('edge_detection')

            if frame_id == 0:
                # Use plane_mask2 as the sampling mask, but plane_support_small
                # as the extent used for the inlier ratio.
                if plane_mask2.any():
                    plane_worker.submit_latest(
                        depth_m_small,
                        plane_mask2,          # sampling mask (near plane band)
                        plane_support_small,  # support mask (all valid in ROI)
                        intr_small,
                        iters=dynamic_iters,
                        tau_m=P["RANSAC_TAU_M"],
                    )

            # ULTRA-SIMPLE SAMPLING
            stream_mgr.start_timer('sampling')
            # --- lightweight quasi-stratified sampling (fast) ---
            ys, xs = cp.where(plane_mask2)
            n = ys.size
            if n > MAX_RANSAC_POINTS:
                # spread samples roughly evenly through the set
                stride = n // MAX_RANSAC_POINTS
                idx = cp.arange(0, n, stride, dtype=cp.int32)[:MAX_RANSAC_POINTS]
                sampled_mask = cp.zeros_like(plane_mask2, dtype=cp.bool_)
                sampled_mask[ys[idx], xs[idx]] = True
                plane_mask2 = sampled_mask
            sampling_ms = stream_mgr.end_timer('sampling')

            # Check time budget - simplify if running out of time
            time_so_far = (time.perf_counter() - stream_mgr.timing_data['total_frame']) * 1000
            if time_so_far > TIME_BUDGET_MS * 0.7:
                # Use faster but less accurate methods if behind schedule
                dynamic_iters = max(DYN_ITERS_MIN, dynamic_iters // 2)

            # --- Async plane refit (non-blocking) ---
            _skip_ransac = (
                _poor_ransac_streak > RANSAC_SKIP_AFTER and
                (frame_id % RANSAC_REPROBE_EVERY) != 0
            )
            if not _skip_ransac and (frame_id % PLANE_BG_EVERY) == 0 and plane_mask2.any():
                plane_worker.submit_latest(
                    depth_m_small,
                    plane_mask2,          # sampling mask
                    plane_support_small,  # support mask
                    intr_small,
                    iters=dynamic_iters,
                    tau_m=P["RANSAC_TAU_M"],
                )

            # read cached plane (fast)
            n_cached, p0_cached, inlier_ratio, is_fresh = plane_worker.get_cached_plane()
            if inlier_ratio < 0.20:
                _poor_ransac_streak += 1
            else:
                _poor_ransac_streak = 0

            if not hasattr(stream_depth, 'plane_stabilizer'):
                stream_depth.plane_stabilizer = PlaneStabilizer(alpha=0.6, min_continuity=2)
                
            if n_cached is not None and is_fresh:
                n_cached, p0_cached = stream_depth.plane_stabilizer.update(n_cached, p0_cached, inlier_ratio)

            # --- Freshness gate + synchronous catch-up ---
            did_sync = False
            # Treat as "needs sync" if no plane, or stale, or we insist on per-frame updates
            need_sync = (n_cached is None) or (not is_fresh)

            if need_sync and plane_mask2.any():
                stream_mgr.start_timer('sync_ransac')
                # Synchronous RANSAC on the *current* frame's downscaled depth and mask
                n_sync, p0_sync, inlier_cnt = ransac_plane_fast(
                    depth_m_small, plane_mask2, intr_small,
                    iters=dynamic_iters, distance_threshold=P["RANSAC_TAU_M"]
                )
                sync_ms = stream_mgr.end_timer('sync_ransac')

                if n_sync is not None:
                    support_pts = int(plane_support_small.sum())
                    inlier_ratio = float(inlier_cnt) / max(1, support_pts)
                    n_cached, p0_cached = n_sync, p0_sync
                    is_fresh = True
                    did_sync = True
                    if MOE_PROF:
                        print(f"[plane/fit/sync]  iters={dynamic_iters} pts={support_pts} "
                              f"inlier={inlier_ratio:.3f} time={sync_ms:.2f} ms")

            plane_ok_flag = bool(is_fresh and (inlier_ratio >= PLANE_MIN_INLIER))
            
            # Dynamic RANSAC adjustment based on timing
            stream_mgr.start_timer('ransac_total')
            current_ransac_time = sync_ms if did_sync else (plane_worker.last_fit_ms() or 0.0)

            # Dynamic adjustment based on both quality AND performance
            if current_ransac_time > 25.0:  # If RANSAC is taking too long
                # Aggressive reduction
                dynamic_iters = max(DYN_ITERS_MIN, dynamic_iters - 15)
                if MOE_PROF:
                    print(f"[depth/adapt] Reducing iters to {dynamic_iters} (slow RANSAC: {current_ransac_time:.1f}ms)")
            elif current_ransac_time < 10.0 and inlier_ratio < 0.25:
                # Room for more quality without hurting performance
                dynamic_iters = min(DYN_ITERS_MAX, dynamic_iters + 10)
                if MOE_PROF:
                    print(f"[depth/adapt] Increasing iters to {dynamic_iters} (fast RANSAC: {current_ransac_time:.1f}ms)")

            # Keep your existing quality-based adjustment too
            if is_fresh and inlier_ratio > GOOD_INLIER_RATIO:
                dynamic_iters = max(DYN_ITERS_MIN, dynamic_iters - DYN_ITERS_STEP_DOWN)
            else:
                dynamic_iters = min(DYN_ITERS_MAX, dynamic_iters + DYN_ITERS_STEP_UP)

            ransac_total_ms = stream_mgr.end_timer('ransac_total')

            depth_norm = cp.clip((depth_m - P["NEAR_M"]) / (P["FAR_M"] - P["NEAR_M"] + 1e-8), 0.0, 1.0)

            # OPTIMIZED confidence computation using the ASYNC worker's cached plane
            stream_mgr.start_timer('confidence_calc')
            plane_ok = plane_ok_flag

            if plane_ok:
                # n_cached, p0_cached already come from the worker (GPU arrays)
                height_map = height_above_plane_fast(
                    depth_m, valid_bool, rx, ry,
                    n_cached, p0_cached
                )

                base_tau_on = 0.006
                tau_on_z = base_tau_on * (1.0 + 0.6 * depth_norm)

                k = max(1e-6, CONF_SIGMOID_SPAN_M / 6.0)
                
                # Use pre-allocated buffer for conf_plane
                x = (height_map - tau_on_z) / k
                cp.clip(x, -6.0, 6.0, out=x)  # tanh is accurate enough in this range
                conf_plane_full = 0.5 + 0.5 * cp.tanh(0.5 * x)  # 3x faster than exp
                
                # Fix: cp.where doesn't have 'out' parameter, so assign the result
                conf_plane_full = cp.where(cp.isfinite(conf_plane_full), conf_plane_full, 0.0)

                # Compute bump using pre-allocated buffer
                hp_pos = cp.clip(height_map, 0.0, None)
                xb = (hp_pos - 0.0008) / (0.0009 + 1e-9)
                xb = cp.clip(xb, -20.0, 20.0)
                cp.reciprocal(1.0 + cp.exp(-xb), out=conf_bump_full)
                cp.clip(conf_bump_full, 0.0, 1.0, out=conf_bump_full)
                
            else:
                # --- GPU-Only Proximity Fallback (MORE ROBUST) ---
                valid_depths = depth_m[valid_bool]
                if valid_depths.size > 100:  # Need sufficient data
                    try:
                        ref = cp.percentile(valid_depths, 60)
                        # Ensure ref is valid
                        if cp.isnan(ref) or cp.isinf(ref):
                            ref = cp.median(valid_depths)
                        if cp.isnan(ref) or cp.isinf(ref):
                            # Last resort: use mean of valid range
                            ref = float(P["NEAR_M"] + P["FAR_M"]) / 2.0
                        
                        delta = cp.clip(ref - depth_m, 0.0, 0.6)
                        conf_plane_full = delta / 0.6
                        # Explicit NaN/Inf cleaning
                        conf_plane_full = cp.nan_to_num(conf_plane_full, nan=0.0, posinf=0.0, neginf=0.0)
                        conf_plane_full = cp.clip(conf_plane_full, 0.0, 1.0)
                        conf_plane_full *= valid_bool.astype(CONF_DTYPE)
                    except Exception as e:
                        if MOE_DEBUG:
                            print(f"[depth/error] Fallback confidence failed: {e}")
                        conf_plane_full.fill(0.0)
                else:
                    # Not enough valid depth data
                    conf_plane_full.fill(0.0)
                
                plane_ok = True  # treat as valid expert output

            # Use pre-allocated buffers for range calculations
            cp.subtract(depth_m, (P["NEAR_M"]+0.05), out=conf_range_full)
            cp.divide(conf_range_full, ((P["FAR_M"]-0.05) - (P["NEAR_M"]+0.05) + 1e-8), out=conf_range_full)
            cp.clip(conf_range_full, 0.0, 1.0, out=conf_range_full)
            cp.subtract(1.0, conf_range_full, out=conf_range_full)
            cp.multiply(conf_range_full, valid_bool.astype(CONF_DTYPE), out=conf_range_full)

            # Use pre-allocated buffer for in_range
            cp.logical_and(depth_m >= P["NEAR_M"], depth_m <= P["FAR_M"], out=in_range_full)
            cp.logical_and(in_range_full, valid_bool, out=in_range_full)

            # Final confidence combination using pre-allocated buffers
            conf_bump_full *= (1.0 - conf_plane_full) * in_range_full.astype(CONF_DTYPE)

            conf = (CONF_PLANE_WEIGHT * conf_plane_full
                    + CONF_BUMP_WEIGHT  * conf_bump_full
                    + CONF_RANGE_WEIGHT * conf_range_full)
            conf = cp.clip(conf, 0.0, 1.0) * in_range_full.astype(CONF_DTYPE)

            # Inlier_ratio already set from the worker earlier
            plane_ok = plane_ok_flag

            # Keep confidence on the decimated grid (Hd x Wd); no need for a 480x640 upsample here.
            conf = cp.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0).astype(cp.float32)
            confidence_ms = stream_mgr.end_timer('confidence_calc')

            # === Option 1: conf aligned into YOLO-square at DS resolution (no rs.align) ===
            if (not USE_ALIGN) and USE_COLOR and (R_gpu is not None) and (t_gpu is not None):
                if MOE_DEBUG and (frame_id % 30) == 0:
                    print("[depth] OPT1 ACTIVE (reproject)")
                _conf_ds_gpu.fill(0)

                # sample depth & conf on decimated grid
                Z  = depth_m[ys_s, xs_s]          # meters
                C  = conf[ys_s, xs_s].astype(CONF_DTYPE, copy=False)
                vv = valid_bool[ys_s, xs_s] & (Z > 0)

                # 3D in depth camera
                X = rx_s * Z
                Y = ry_s * Z

                # depth -> color
                Xc = R_gpu[0,0]*X + R_gpu[0,1]*Y + R_gpu[0,2]*Z + t_gpu[0]
                Yc = R_gpu[1,0]*X + R_gpu[1,1]*Y + R_gpu[1,2]*Z + t_gpu[1]
                Zc = R_gpu[2,0]*X + R_gpu[2,1]*Y + R_gpu[2,2]*Z + t_gpu[2]

                ok = vv & (Zc > 1e-6)

                # project into color pixels
                uc = (Xc / Zc) * fx_c + ppx_c
                vc = (Yc / Zc) * fy_c + ppy_c

                # map color pixels -> YOLO square
                uy = uc * scale_c2y + pad_x
                vy = vc * scale_c2y + pad_y

                # YOLO square -> DS indices
                ix = (uy * (W_DS / 640.0)).astype(cp.int32)
                iy = (vy * (H_DS / 640.0)).astype(cp.int32)

                ok = ok & (ix >= 0) & (ix < W_DS) & (iy >= 0) & (iy < H_DS)

                # max-pool into DS grid (preserves thin tools)
                cp.maximum.at(_conf_ds_gpu, (iy[ok], ix[ok]), C[ok])

            else:
                if MOE_DEBUG and (frame_id % 30) == 0:
                    print("[depth] OPT1 FALLBACK (NO reproject)")
                # fallback: old behavior (ONLY correct if aligned)
                _src_h, _src_w = int(conf.shape[0]), int(conf.shape[1])
                if (_src_h, _src_w) != (H_DS, W_DS):
                    try:
                        cp_zoom(conf, (H_DS / _src_h, W_DS / _src_w), order=0, output=_conf_ds_gpu)
                    except Exception:
                        sh = max(1, _src_h // H_DS)
                        sw = max(1, _src_w // W_DS)
                        _conf_ds_gpu[...] = conf[0:H_DS*sh:sh, 0:W_DS*sw:sw]
                else:
                    _conf_ds_gpu[...] = conf

            conf = _conf_ds_gpu
            # Fill 1px holes (important for thin tools)
            # Fill gaps with a wider max-dilation (handles curved surfaces like back of head)
            from cupyx.scipy.ndimage import grey_closing as cp_closing
            conf = cp_closing(conf, size=3)

            conf = cp.clip(conf, 0.0, 1.0)
             # Temporal EMA — kills frame-to-frame noise
            if _conf_ema is None or _conf_ema.shape != conf.shape:
                _conf_ema = conf.copy()
            else:
                _conf_ema = (1.0 - _EMA_ALPHA) * _conf_ema + _EMA_ALPHA * conf
            conf = _conf_ema

            # Spatial smoothing — fills remaining gaps, rounds blobs
            conf = cp.clip(conf, 0.0, 1.0)
            _conf_ds_gpu[...] = conf
            # --- Confidence hold (GPU-side) ---
            # Detect degenerate/invalid confidence on the small grid.
            # NOTE: float(cp.nanmean/nanmax) will synchronize, but the map is small (e.g., 128x128).
            _m = float(cp.nanmean(_conf_ds_gpu))
            _x = float(cp.nanmax(_conf_ds_gpu))
            conf_bad = (
                (not np.isfinite(_m))
                or (not np.isfinite(_x))
                or (_m < 0.005 and _x < 0.02)
            )

            if conf_bad and plane_ok_flag and _have_last_conf and (_last_conf_age < CONF_HOLD_FRAMES):
                # Reuse last valid confidence map (eliminates blue-frame flicker)
                _conf_ds_gpu[...] = _last_conf_ds_gpu
                _last_conf_age += 1
            else:
                # Accept current confidence map and refresh cache
                _last_conf_ds_gpu[...] = _conf_ds_gpu
                _have_last_conf = True
                _last_conf_age = 0

            # publish FPS (count every frame)
            _pub_cnt += 1
            if MOE_PROF and (frame_id % 30 == 0):
                total_frame_ms = (time.perf_counter() - stream_mgr.timing_data['total_frame']) * 1000
                now = time.perf_counter()
                elapsed = now - _pub_t0
                pub_fps = (_pub_cnt / elapsed) if elapsed > 0.5 else 0.0  # avoid noise

                print("[depth/detailed_timers] "
                      f"total={total_frame_ms:.1f}ms | "
                      f"filters={filters_ms:.1f}ms | "
                      f"h2d={h2d_ms:.1f}ms | "
                      f"downsample={downsample_ms:.1f}ms | "
                      f"roi={roi_ms:.1f}ms | "
                      f"edge={edge_ms:.1f}ms | "
                      f"sampling={sampling_ms:.1f}ms | "
                      f"ransac={ransac_total_ms:.1f}ms | "
                      f"conf={confidence_ms:.1f}ms | "
                      f"pub_fps={pub_fps:.2f} | iters={dynamic_iters}")
                
                _pub_cnt = 0
                _pub_t0 = now

            frame_id += 1

            t_frame = time.perf_counter()
            # Hardware timestamp from RealSense bag (milliseconds → seconds)
            hw_ts_s = aligned.get_timestamp() / 1000.0
            # Send timestamped color to the broker
            if share_color_ts is not None:
                try:
                    try:
                        if share_color_ts.full():
                            share_color_ts.get_nowait()
                    except Empty:
                        pass
                    share_color_ts.put_nowait((t_frame, bgr_aligned))
                except Full:
                    try:
                        share_color_ts.get_nowait()
                    except Empty:
                        pass
                    try:
                        share_color_ts.put_nowait((t_frame, bgr_aligned))
                    except Full:
                        pass

            # Include the same timestamp in the generator output
            stream_mgr.start_timer('final_transfer')

            # === NEW: async D2H into pinned buffer + event fence ===
            # 1) get next pinned host page
            host_buf, gen_id, _slot = _conf_buf.next()

            # 2) start an event on the current CuPy stream
            stream = stream_mgr.stream
            evt_conf_done = cp.cuda.Event()

            # 3) enqueue async device->host copy into pinned memory
            cp.cuda.runtime.memcpyAsync(
                host_buf.ctypes.data,          # dst (host pinned)
                _conf_ds_gpu.data.ptr,         # src (device)
                _conf_ds_gpu.nbytes,
                cp.cuda.runtime.memcpyDeviceToHost,
                stream.ptr
            )

            # 4) record event after the copy
            evt_conf_done.record(stream)

            final_xfer_ms = stream_mgr.end_timer('final_transfer') # DON"T REMOVE BRO

            # 5) package data for the broker (DepthItem format)
            depth_msg = {
                "conf": host_buf,              # pinned numpy buffer (float32)
                "plane_ok": plane_ok,
                "inlier_ratio": float(inlier_ratio),
                "sync_evt": evt_conf_done,
                "gen_id": gen_id,
                "t": t_frame,
                "hw_ts": hw_ts_s,              # RealSense hardware timestamp (seconds)
            }
            yield depth_msg

    finally:
        local_stop = True
        try:
            plane_worker.stop()
            pipeline.stop()
        except Exception:
            pass

def main():
    if MOE_DEBUG:
        print("[depth_expert] Run headless via run_moe.py (stream_depth).")

if __name__ == "__main__":
    main()