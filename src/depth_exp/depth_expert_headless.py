#!/usr/bin/env python3
# Depth Expert (FAST mode) — HEADLESS - Formerly called zabre.py (gone but not forgotten)
# Same math as your current script, but NO HUD/NO WINDOWS.
# Prints per-stage timings and FPS to the console at a fixed interval.

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from collections import deque
import threading
import argparse

# =========================
# Mode presets
# =========================
MODE = "tabletop"      # "tabletop" or "room"
FAST_PLANE_EVERY = 1   # keep RANSAC every frame
SCALE_PLANE  = 0.5     # lower if you need more headroom (same plane, just faster)
SHOW_BG_REMOVED = False
USE_COLOR = True       # toggle RGB stream + alignment (True = same behavior as before)

# RANSAC subsample + dynamic iters
MAX_POINTS_FOR_RANSAC = 500
DYN_ITERS_MIN = 60
DYN_ITERS_MAX = 120
DYN_ITERS_STEP_DOWN = 5
DYN_ITERS_STEP_UP   = 25
GOOD_INLIER_RATIO   = 0.80

# Confidence shaping
CONF_SIGMOID_SPAN_M = 0.010   # effective transition width (sharper -> smaller)
CONF_PLANE_WEIGHT   = 0.95    # plane dominates
CONF_RANGE_WEIGHT   = 0.05    # tiny bias by range

# Base resolution
W, H, FPS = 640, 480, 30
SMOOTH_KSIZE  = 5

# ---------- params ----------
def get_params(mode: str):
    if mode == "tabletop":
        return dict(
            NEAR_M=0.10, FAR_M=1.0, CLIP_M=1.0,
            RANSAC_ITERS=120, RANSAC_TAU_M=0.004,
            ROI_FRAC=0.20, FAR_PERCENTILE=85,
            TAU_ON=0.010, TAU_OFF=0.004,
            CONF_SPAN=0.030,
            EDGE_SIGMA_PX=1.2, EDGE_THR_MM=0.50,
        )
    else:
        return dict(
            NEAR_M=0.30, FAR_M=3.0, CLIP_M=3.0,
            RANSAC_ITERS=120, RANSAC_TAU_M=0.010,
            ROI_FRAC=0.10, FAR_PERCENTILE=80,
            TAU_ON=0.020, TAU_OFF=0.015,
            CONF_SPAN=0.050,
            EDGE_SIGMA_PX=1.2, EDGE_THR_MM=2.0,
        )

P = get_params(MODE)

# ---------- helpers ----------
def normalize01(x, lo, hi, out=None):
    if out is None:
        out = np.empty_like(x, dtype=np.float32)
    np.subtract(x, lo, out=out)
    np.divide(out, (hi - lo + 1e-6), out=out)
    np.clip(out, 0.0, 1.0, out=out)
    return out

# Height above plane (same math, fewer ops): X=rx*Z, Y=ry*Z
def height_above_plane_fast(depth_m, valid, rx, ry, n, p0, out=None):
    Z = depth_m
    if out is None:
        out = np.empty_like(Z, dtype=np.float32)
    # (X - p0x)*nx
    np.multiply(rx, Z, out=out)
    out -= p0[0]
    out *= n[0]
    # + (Y - p0y)*ny
    tmp = ry * Z - p0[1]
    out += tmp * n[1]
    # + (Z - p0z)*nz
    out += (Z - p0[2]) * n[2]
    out[~valid] = np.nan
    if np.nanmean(out) < 0:
        np.negative(out, out=out)
    return out

# RANSAC plane fit (optimized)
def ransac_plane_optimized(depth_m, mask, intr, iters=120, tau_m=0.003):
    ys, xs = np.where(mask)
    if ys.size < 200:
        return None, None, -1

    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    z_vals = depth_m[ys, xs].astype(np.float32)
    X_vals = (xs.astype(np.float32) - cx) / fx * z_vals
    Y_vals = (ys.astype(np.float32) - cy) / fy * z_vals
    Pts = np.stack((X_vals, Y_vals, z_vals), axis=1)  # [N,3]

    best_n, best_p0, best_count = None, None, -1
    N = Pts.shape[0]

    for _ in range(iters):
        i1, i2, i3 = np.random.permutation(N)[:3]
        p1, p2, p3 = Pts[i1], Pts[i2], Pts[i3]
        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        d = np.abs((Pts - p1).dot(n))
        count = int((d < tau_m).sum())
        if count > best_count:
            best_count, best_n, best_p0 = count, n.astype(np.float32), p1.astype(np.float32)

    if best_count < 100:
        return None, None, best_count

    # refine with inliers
    d = np.abs((Pts - best_p0).dot(best_n))
    inl = d < tau_m
    if inl.sum() < 50:
        return None, None, best_count

    Psel = Pts[inl]
    p0 = Psel.mean(0).astype(np.float32)
    _, _, Vt = np.linalg.svd(Psel - p0, full_matrices=False)
    n = Vt[-1].astype(np.float32)
    n /= (np.linalg.norm(n) + 1e-9)
    return n, p0, best_count

# ===== non-blocking capture thread =====
frame_buf = deque(maxlen=1)
stop_flag = False
def grabber(pipeline, align_obj):
    global stop_flag
    while not stop_flag:
        frames = pipeline.wait_for_frames()
        if align_obj is not None:
            frames = align_obj.process(frames)
        frame_buf.append(frames)

def main():
    global stop_flag

    ap = argparse.ArgumentParser(description="Depth Expert (headless)")
    ap.add_argument("--print-every", type=float, default=0.5,
                    help="Seconds between console prints (default 0.5)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="Optional run duration in seconds (0 = infinite)")
    ap.add_argument("--use-color", type=int, default=1,
                    help="1 = align depth to color (default), 0 = depth only")
    ap.add_argument("--scale-plane", type=float, default=SCALE_PLANE,
                    help=f"Plane compute scale (default {SCALE_PLANE})")
    args = ap.parse_args()

    use_color = bool(args.use_color)
    scale_plane = float(args.scale_plane)
    print_every = max(0.05, float(args.print_every))
    run_duration = float(args.duration)

    params = P
    NEAR_M, FAR_M, CLIP_M = params["NEAR_M"], params["FAR_M"], params["CLIP_M"]

    # OpenCV perf flags
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(max(1, os.cpu_count()-1))
    except Exception:
        pass

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    if use_color:
        config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()
    print(f"[INFO] Mode={MODE} | Headless | Depth scale: {depth_scale:.8f} | USE_COLOR={use_color} | SCALE_PLANE={scale_plane}")

    align = rs.align(rs.stream.color) if use_color else None
    try:
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
        if depth_sensor.supports(rs.option.laser_power):
            rng = depth_sensor.get_option_range(rs.option.laser_power)
            depth_sensor.set_option(rs.option.laser_power, min(360.0, rng.max))
    except Exception:
        pass

    spatial  = rs.spatial_filter();  spatial.set_option(rs.option.holes_fill, 3)
    holefill = rs.hole_filling_filter()

    # Intrinsics (match stream actually used)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics() if use_color \
           else profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    # === Precompute full-res grids + ray scales once ===
    ys_full, xs_full = np.mgrid[0:H, 0:W].astype(np.float32)
    rx = (xs_full - intr.ppx) / intr.fx
    ry = (ys_full - intr.ppy) / intr.fy

    # === Precompute morphology kernels ===
    k_open3  = np.ones((3,3), np.uint8)

    # === Precompute downscale constants ===
    pw, ph = int(W*scale_plane), int(H*scale_plane)
    intr_small = rs.intrinsics()
    intr_small.width, intr_small.height = pw, ph
    scale_x, scale_y = pw / W, ph / H
    intr_small.ppx = intr.ppx * scale_x
    intr_small.ppy = intr.ppy * scale_y
    intr_small.fx  = intr.fx  * scale_x
    intr_small.fy  = intr.fy  * scale_y

    # Static ROI/border mask at small scale
    rf = params["ROI_FRAC"]
    Hc0, Hc1 = int(ph*rf), int(ph*(1-rf))
    Wc0, Wc1 = int(pw*rf), int(pw*(1-rf))
    roi_small_static = np.zeros((ph, pw), bool); roi_small_static[Hc0:Hc1, Wc0:Wc1] = True
    bf = 0.03
    bx, by = int(pw*bf), int(ph*bf)
    border_mask = np.ones((ph, pw), bool)
    border_mask[:, :bx]  = False
    border_mask[:, -bx:] = False
    border_mask[:by, :]  = False
    border_mask[-by:, :] = False
    roi_border_static = roi_small_static & border_mask

    # Precompute stratified subsampling cell boundaries
    Gx, Gy = 8, 6
    cell_w = max(1, pw // Gx)
    cell_h = max(1, ph // Gy)
    cells = [(gy*cell_h, min(ph, (gy+1)*cell_h), gx*cell_w, min(pw, (gx+1)*cell_w))
             for gy in range(Gy) for gx in range(Gx)]
    per_cell = max(1, MAX_POINTS_FOR_RANSAC // (Gx * Gy))

    # Pre-allocate working buffers
    depth_m         = np.empty((H, W), dtype=np.float32)
    valid_bool      = np.empty((H, W), dtype=bool)
    depth_m_small   = np.empty((ph, pw), dtype=np.float32)
    valid_small_u8  = np.empty((ph, pw), dtype=np.uint8)
    conf_bump_full  = np.empty((H, W), dtype=np.float32)

    cached_n, cached_p0 = None, None
    frame_id = 0
    dynamic_iters = int(np.clip(params["RANSAC_ITERS"], DYN_ITERS_MIN, DYN_ITERS_MAX))

    # ---- start the non-blocking capture thread ----
    t_cap = threading.Thread(target=grabber, args=(pipeline, align), daemon=True)
    t_cap.start()

    # Console metrics accumulators
    sum_filters = sum_down = sum_ransac = sum_conf = sum_total = 0.0
    n_print = 0
    last_print = time.perf_counter()
    t_start = last_print

    try:
        while True:
            t0 = time.perf_counter()

            # Optional duration cap
            if run_duration > 0 and (t0 - t_start) >= run_duration:
                break

            # ---- non-blocking fetch ----
            if not frame_buf:
                time.sleep(0.001)
                continue
            aligned = frame_buf[-1]
            d = aligned.get_depth_frame()
            c = aligned.get_color_frame() if use_color else None
            if use_color and (not d or not c):
                continue
            if (not use_color) and (not d):
                continue

            # Filters (on depth only)
            f = d
            f = spatial.process(f)
            f = holefill.process(f)
            t_filters = time.perf_counter()

            depth_raw = np.asanyarray(f.get_data())                 # uint16

            # Full-res depth + validity
            np.multiply(depth_raw.astype(np.float32), depth_scale, out=depth_m)
            np.greater(depth_raw, 0, out=valid_bool)

            # Downscale
            cv2.resize(depth_m, (pw, ph), dst=depth_m_small, interpolation=cv2.INTER_AREA)
            cv2.resize(valid_bool.astype(np.uint8), (pw, ph), dst=valid_small_u8, interpolation=cv2.INTER_NEAREST)
            valid_small = valid_small_u8.view(np.bool_)
            t_down = time.perf_counter()

            # ---------- Range FG (for diagnostics only, not displayed) ----------
            in_rng = (depth_m >= P["NEAR_M"]) & (depth_m <= P["FAR_M"]) & valid_bool
            fg_range = (in_rng.astype(np.uint8) * 255)
            fg_range = cv2.morphologyEx(fg_range, cv2.MORPH_OPEN, k_open3, iterations=1)

            # ---------- Plane candidates (ROI + far percentile) ----------
            roi_small = valid_small & roi_border_static
            z_valid = depth_m_small[roi_small]
            if z_valid.size:
                p_ctr = params["FAR_PERCENTILE"]; band = 65.0
                p_lo = max(50.0, p_ctr - band); p_hi = min(99.5, p_ctr + band)
                far_lo = float(np.percentile(z_valid, p_lo))
                far_hi = float(np.percentile(z_valid, p_hi))
                plane_mask = roi_small & (depth_m_small >= far_lo) & (depth_m_small <= far_hi)
            else:
                plane_mask = roi_small

            # High-pass (no NaNs): fill invalid with median once, then blur/diff
            Z = depth_m_small.copy()
            valid_vals = Z[valid_small]
            med = float(np.median(valid_vals)) if valid_vals.size else 0.0
            Z[~valid_small] = med
            low = cv2.GaussianBlur(Z, (0,0), 5.0)
            hp  = Z - low

            gx = cv2.Sobel(hp, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(hp, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx*gx + gy*gy)

            edges_small = (mag > 0.0012)
            non_edge_small = plane_mask & (~edges_small)
            plane_mask2 = plane_mask & non_edge_small
            if plane_mask2.sum() < 300:
                plane_mask2 = plane_mask

            # Stratified subsampling
            total = int(plane_mask2.sum())
            if total > 0:
                sampled_mask = np.zeros_like(plane_mask2, dtype=bool)
                for (y0, y1, x0, x1) in cells:
                    cell = plane_mask2[y0:y1, x0:x1]
                    ys, xs = np.where(cell)
                    n = ys.size
                    if n == 0:
                        continue
                    if n > per_cell:
                        idx = np.random.choice(n, per_cell, replace=False)
                        ys, xs = ys[idx], xs[idx]
                    sampled_mask[y0 + ys, x0 + xs] = True
                remaining = MAX_POINTS_FOR_RANSAC - int(sampled_mask.sum())
                if remaining > 0 and total > int(sampled_mask.sum()):
                    leftover = plane_mask2 & ~sampled_mask
                    yy, xx = np.where(leftover)
                    if yy.size > 0:
                        take = min(remaining, yy.size)
                        idx = np.random.choice(yy.size, take, replace=False)
                        sampled_mask[yy[idx], xx[idx]] = True
                plane_mask2 = sampled_mask

            # ---------- Per-frame plane fit (RANSAC) ----------
            t_before_ransac = time.perf_counter()
            if (frame_id % FAST_PLANE_EVERY == 0) or (cached_n is None):
                n_small, p0_small, inlier_count = ransac_plane_optimized(
                    depth_m_small, plane_mask2, intr_small,
                    iters=dynamic_iters, tau_m=params["RANSAC_TAU_M"]
                )
                if n_small is not None:
                    cached_n, cached_p0 = n_small, p0_small
                    sampled_pts = int(plane_mask2.sum())
                    ratio = (inlier_count / max(1, sampled_pts)) if sampled_pts > 0 else 0.0
                    if ratio > GOOD_INLIER_RATIO:
                        dynamic_iters = max(DYN_ITERS_MIN, dynamic_iters - DYN_ITERS_STEP_DOWN)
                    else:
                        dynamic_iters = min(DYN_ITERS_MAX, dynamic_iters + DYN_ITERS_STEP_UP)
                else:
                    dynamic_iters = P["RANSAC_ITERS"]
            t_after_ransac = time.perf_counter()

            # ---------- Confidence ----------
            t_before_conf = time.perf_counter()
            if cached_n is not None:
                # Height via ray scales
                height_map = height_above_plane_fast(depth_m, valid_bool, rx, ry, cached_n, cached_p0)

                base_tau_on = 0.005
                tau_on_z = normalize01(depth_m, P["NEAR_M"], P["FAR_M"])
                tau_on_z = base_tau_on * (1.0 + 0.6 * tau_on_z)

                k = max(1e-6, CONF_SIGMOID_SPAN_M / 6.0)
                x = (height_map - tau_on_z) / k
                x = np.clip(x, -20.0, 20.0)

                conf_plane = 1.0 / (1.0 + np.exp(-x))
                conf_plane[~np.isfinite(conf_plane)] = 0.0

                # bump from HP residual (small scale → upsample)
                hp_pos = np.clip(hp, 0.0, None)
                xb = (hp_pos - 0.0008) / (0.0009 + 1e-9)
                xb = np.clip(xb, -20.0, 20.0)
                conf_bump_small = 1.0 / (1.0 + np.exp(-xb))
                cv2.resize(conf_bump_small, (W, H), dst=conf_bump_full, interpolation=cv2.INTER_LINEAR)
                np.clip(conf_bump_full, 0.0, 1.0, out=conf_bump_full)
            else:
                conf_plane = np.zeros_like(depth_m, dtype=np.float32)
                conf_bump_full.fill(0.0)

            conf_range = normalize01(depth_m, P["NEAR_M"]+0.05, P["FAR_M"]-0.05) * valid_bool.astype(np.float32)
            conf_range = 1.0 - conf_range

            in_range = ((depth_m >= P["NEAR_M"]) & (depth_m <= P["FAR_M"]) & valid_bool).astype(np.float32)
            conf_bump_full *= (1.0 - conf_plane) * in_range

            CONF_BUMP_WEIGHT = 0.06
            conf = (CONF_PLANE_WEIGHT * conf_plane
                    + CONF_BUMP_WEIGHT  * conf_bump_full
                    + CONF_RANGE_WEIGHT * conf_range)
            conf = np.clip(conf, 0.0, 1.0)
            conf *= in_range
            t_after_conf = time.perf_counter()

            # ===== Metrics (console only) =====
            t_total = time.perf_counter()
            ms_filters= (t_filters - t0) * 1000.0
            ms_down   = (t_down - t_filters) * 1000.0
            ms_ransac = (t_after_ransac - t_before_ransac) * 1000.0
            ms_conf   = (t_after_conf - t_before_conf) * 1000.0
            ms_total  = (t_total - t0) * 1000.0
            fps_inst  = 1000.0 / max(1e-3, ms_total)

            # accumulate for print interval
            sum_filters += ms_filters
            sum_down    += ms_down
            sum_ransac  += ms_ransac
            sum_conf    += ms_conf
            sum_total   += ms_total
            n_print     += 1
            frame_id    += 1

            now = t_total
            if now - last_print >= print_every:
                # robust plane health numbers
                sampled_pts = int(plane_mask2.sum()) if 'plane_mask2' in locals() else 0
                inliers = int(inlier_count) if 'inlier_count' in locals() and inlier_count is not None else 0
                ratio = (inliers / sampled_pts) if sampled_pts > 0 else 0.0

                avg_filters = sum_filters / max(1, n_print)
                avg_down    = sum_down / max(1, n_print)
                avg_ransac  = sum_ransac / max(1, n_print)
                avg_conf    = sum_conf / max(1, n_print)
                avg_total   = sum_total / max(1, n_print)
                fps_avg     = 1000.0 / max(1e-3, avg_total)

                print(f"[{frame_id:6d}] FPS_inst={fps_inst:5.1f} | FPS_avg={fps_avg:5.1f} "
                      f"| ms: total {avg_total:6.1f} | filt {avg_filters:5.1f} | down {avg_down:5.1f} "
                      f"| ransac {avg_ransac:5.1f} | conf {avg_conf:5.1f} "
                      f"| plane inliers={inliers}/{sampled_pts} ({ratio*100:4.1f}%) | dyn_iters={dynamic_iters}")

                # reset window
                sum_filters = sum_down = sum_ransac = sum_conf = sum_total = 0.0
                n_print = 0
                last_print = now

    except KeyboardInterrupt:
        pass
    finally:
        stop_flag = True
        try:
            pipeline.stop()
        except Exception:
            pass
        # No windows to destroy (headless)

if __name__ == "__main__":
    main()
