#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple
from collections import deque
import os, time

import numpy as np
import cv2  # ensure present to build the preview canvas

MOE_DEBUG = os.getenv("MOE_DEBUG", "0") not in ("0", "", "false", "False")
MOE_PROF = os.getenv("MOE_PROF", "0") not in ("0", "", "false", "False")


@dataclass
class YoloInst:
    mask_prob: np.ndarray
    cls: int
    score: float


@dataclass
class YoloItem:
    t: float
    dets: List[YoloInst]


@dataclass
class DepthItem:
    t: float
    conf: np.ndarray
    plane_ok: bool
    inlier_ratio: float
    sync_evt: object | None = None
    gen_id: int = -1
    hw_ts: float = 0.0          # RealSense hardware timestamp (seconds)


@dataclass
class GateHealth:
    plane_ok: bool
    inlier_ratio: float


@dataclass
class GateInput:
    t: float
    yolo_instances: List[YoloInst]
    conf_yolo: np.ndarray
    health: GateHealth
    bgr_yolo: Optional[np.ndarray] = None
    hw_ts: float = 0.0          # RealSense hardware timestamp (seconds)


class Broker:
    def __init__(self,
                 letterbox_fn: Callable,
                 depth2yolo_img: Callable,
                 src_wh_depth: Tuple[int, int] = (640, 480),
                 dst_wh_yolo: Tuple[int, int] = (640, 640),
                 hold_last: bool = False,
                 metrics_every: float = 1.0,
                 tol_floor: float = 0.010,
                 tol_ceiling: float = 0.120,
                 tol_fraction: float = 0.90,
                 max_queue: int = 1):
        self.compute_letterbox = letterbox_fn
        self.depth2yolo_img = depth2yolo_img

        self.src_wh_depth = tuple(src_wh_depth)
        self.dst_wh_yolo = tuple(dst_wh_yolo)

        self.lb = self.compute_letterbox(self.src_wh_depth, self.dst_wh_yolo)

        self.qy: Deque[YoloItem] = deque(maxlen=max_queue)
        self.qd: Deque[DepthItem] = deque(maxlen=max_queue)
        self._gate_cb: Optional[Callable[[GateInput], None]] = None

        self.metrics_every = float(metrics_every)
        self.last_metrics = time.perf_counter()
        self.paired = 0
        self.drop_y = 0
        self.drop_d = 0
        self.hold_last = bool(hold_last)

        self._ema_dy = None
        self._ema_dd = None
        self._last_t_y = None
        self._last_t_d = None
        self._ema_alpha = 0.12

        self.tol_floor = float(tol_floor)
        self.tol_ceiling = float(tol_ceiling)
        self.tol_fraction = float(tol_fraction)

        # max allowed age
        self._depth_max_age_s = float(os.getenv('MOE_DEPTH_MAX_AGE_MS', '150')) / 1000.0
        self._yolo_max_age_s  = float(os.getenv('MOE_YOLO_MAX_AGE_MS',  '150')) / 1000.0

        self._color_buf: Deque[Tuple[float, np.ndarray]] = deque(maxlen=6)

        self._need_rgb = os.getenv("MOE_BROKER_RGB", "1") not in ("0","false","False")

        self._color_q = None
        self._last_bgr = None

        # Pre-allocate buffers to avoid runtime allocations
        H, W = self.dst_wh_yolo[1], self.dst_wh_yolo[0]

        self._bgr_canvas = np.full((H, W, 3), 114, dtype=np.uint8)
        self._conf_buf = np.empty((H, W), np.float32)

        self._lb_pad = (self.lb.pad[0], self.lb.pad[1], self.lb.pad[2], self.lb.pad[3])
        self._lb_size = tuple(self.lb.new_wh)

    def set_color_queue(self, q):
        self._color_q = q

    def set_gate_callback(self, fn: Callable[[GateInput], None]):
        self._gate_cb = fn

    def push_yolo(self, msg: YoloItem):
        self._update_ema('y', msg.t)
        self.qy.append(msg)
        self._try_pair()

    def push_depth(self, msg: DepthItem):
        self._update_ema('d', msg.t)
        self.qd.append(msg)
        # Trigger pairing if we have any recent YOLO, or hold_last, or an item already in qy.
        if self._gate_cb and (self.hold_last or self.qy):
            self._try_pair()

    def _update_ema(self, which: str, t: float):
        if which == 'y':
            if self._last_t_y is not None:
                dt = max(1e-6, t - self._last_t_y)
                self._ema_dy = dt if self._ema_dy is None else (1 - self._ema_alpha) * self._ema_dy + self._ema_alpha * dt
            self._last_t_y = t
        else:
            if self._last_t_d is not None:
                dt = max(1e-6, t - self._last_t_d)
                self._ema_dd = dt if self._ema_dd is None else (1 - self._ema_alpha) * self._ema_dd + self._ema_alpha * dt
            self._last_t_d = t

    def _adaptive_tolerance(self) -> float:
        candidates = [v for v in (self._ema_dy, self._ema_dd) if v is not None]
        if not candidates:
            base = 0.025
        else:
            fast_frame = min(candidates)
            base = fast_frame * self.tol_fraction
        if base < self.tol_floor:
            return self.tol_floor
        if base > self.tol_ceiling:
            return self.tol_ceiling
        return base

    def _emit_metrics_if_needed(self):
        now = time.perf_counter()
        if now - self.last_metrics >= self.metrics_every:
            fps_y = (1.0 / self._ema_dy) if self._ema_dy else 0.0
            fps_d = (1.0 / self._ema_dd) if self._ema_dd else 0.0
            
            # NEW: Age diagnostics for queues
            qy_ages = [(now - msg.t) * 1000 for msg in self.qy] if self.qy else [0]
            qd_ages = [(now - msg.t) * 1000 for msg in self.qd] if self.qd else [0]
            
            if MOE_DEBUG:
                print(f"[broker] paired={self.paired} drop_yolo={self.drop_y} drop_depth={self.drop_d} | "
                    f"qy={len(self.qy)}({max(qy_ages):.1f}ms) qd={len(self.qd)}({max(qd_ages):.1f}ms) | "
                    f"tol={self._adaptive_tolerance()*1000:.1f}ms | "
                    f"fps_yolo={fps_y:.1f} fps_depth={fps_d:.1f}")
                
            self.paired = self.drop_y = self.drop_d = 0
            self.last_metrics = now

    def _get_bgr_for_time(self, pair_t: float, tol_s: float) -> Optional[np.ndarray]:
        # Non-blocking: pull at most one new color sample
        if self._color_q is not None:
            try:
                item = self._color_q.get_nowait()
                if isinstance(item, tuple) and len(item) == 2:
                    ct, cbgr = item
                    self._color_buf.append((ct, cbgr))
                else:
                    # legacy: bare bgr → treat as “now”
                    self._color_buf.append((time.perf_counter(), item))
            except Exception:
                pass

        if not self._color_buf:
            return None

        # Prefer frames not newer than pair_t; fall back to nearest within tol
        best = None
        best_abs = 1e9
        for ct, cbgr in reversed(self._color_buf):
            dt = pair_t - ct
            if dt < 0:
                continue
            if dt <= tol_s and dt < best_abs:
                best = (ct, cbgr); best_abs = dt

        if best is None:
            for ct, cbgr in reversed(self._color_buf):
                adt = abs(pair_t - ct)
                if adt <= tol_s and adt < best_abs:
                    best = (ct, cbgr); best_abs = adt

        if best is None:
            return None

        # Letterbox into preallocated canvas
        if (self._bgr_canvas.shape[0] != self.dst_wh_yolo[1] or self._bgr_canvas.shape[1] != self.dst_wh_yolo[0]):
            H, W = self.dst_wh_yolo[1], self.dst_wh_yolo[0]
            self._bgr_canvas = np.full((H, W, 3), 114, dtype=np.uint8)

        new_w, new_h = self._lb_size
        L, T, R, B = self._lb_pad
        self._bgr_canvas[...] = 114
        cv2.resize(best[1], (new_w, new_h),
                interpolation=cv2.INTER_LINEAR,
                dst=self._bgr_canvas[T:T+new_h, L:L+new_w])
        return self._bgr_canvas

    def _map_conf_full(self, conf_depth01: np.ndarray) -> np.ndarray:
        H, W = self.dst_wh_yolo[1], self.dst_wh_yolo[0]
        if (self._conf_buf is None) or (self._conf_buf.shape != (H, W)):
            self._conf_buf = np.empty((H, W), np.float32)

        # Let depth2yolo_img handle the resize from whatever DS we have
        h, w = conf_depth01.shape[:2]

        # OPTION 1: if DS×DS (square), it is already YOLO-square space -> ONLY upsample to 640×640
        if h == w:
            cv2.resize(conf_depth01, self.dst_wh_yolo, dst=self._conf_buf, interpolation=cv2.INTER_LINEAR)
            return self._conf_buf

        # Otherwise: old path (depth-space -> YOLO letterbox)
        try:
            return self.depth2yolo_img(conf_depth01, self.lb, is_mask=False, out=self._conf_buf)
        except TypeError:
            tmp = self.depth2yolo_img(conf_depth01, self.lb, is_mask=False)
            self._conf_buf[...] = tmp
            return self._conf_buf

    def _try_pair(self):
        if self._gate_cb is None:
            return
        now = time.perf_counter()
        tol = self._adaptive_tolerance()

        # 1) Hard-age filter: purge stale items from the back (oldest)
        def _purge_stale():
            # Depth
            while self.qd and (now - self.qd[0].t) > self._depth_max_age_s:
                self.qd.popleft(); self.drop_d += 1
            # YOLO
            while self.qy and (now - self.qy[0].t) > self._yolo_max_age_s:
                self.qy.popleft(); self.drop_y += 1

        _purge_stale()

        if not self.qy or not self.qd:
            return

        # 2) Always pair newest-with-newest to minimize drift
        y = self.qy[-1]
        d = self.qd[-1]

        # 3) If timestamps are too far apart, drop the older stream item and retry
        #    until we either have a match or queues empty.
        while self.qy and self.qd:
            y = self.qy[-1]
            d = self.qd[-1]
            dt = abs(y.t - d.t)

            if dt <= tol:
                break  # fresh pair

            # Drop the older one
            if y.t < d.t:
                # YOLO older than depth → drop YOLO
                self.qy.pop(); self.drop_y += 1
            else:
                self.qd.pop(); self.drop_d += 1

            _purge_stale()

        if not self.qy or not self.qd:
            return

        # Final check
        y = self.qy[-1]; d = self.qd[-1]
        if abs(y.t - d.t) > tol:
            return  # nothing matchable right now

        # Fence on depth D2H if it was async
        if getattr(d, "sync_evt", None) is not None:
            try:
                d.sync_evt.synchronize()
                if MOE_DEBUG and MOE_PROF and d.gen_id >= 0:
                    print(f"[broker/depth] gen={d.gen_id}")
            except Exception:
                pass

        # 4) Build GateInput using preallocated buffers if possible
        pair_t = max(y.t, d.t)
        conf_yolo = self._map_conf_full(d.conf)

        gi = GateInput(
            t=pair_t,
            yolo_instances=y.dets,
            conf_yolo=conf_yolo,
            health=GateHealth(plane_ok=d.plane_ok, inlier_ratio=d.inlier_ratio),
            bgr_yolo = self._get_bgr_for_time(pair_t, tol) if self._need_rgb else None,
            hw_ts=d.hw_ts,
        )

        self.paired += 1
        self._gate_cb(gi)

        if len(self.qy) > 1: 
            last = self.qy[-1]; self.qy.clear(); self.qy.append(last)
        if len(self.qd) > 1: 
            last = self.qd[-1]; self.qd.clear(); self.qd.append(last)
        self._emit_metrics_if_needed()