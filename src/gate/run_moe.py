#!/usr/bin/env python3
import argparse, importlib, time, sys, os, cv2
import numpy as np
from queue import Full, Empty, Queue
from threading import Thread, Event
import cv2
cv2.setNumThreads(5)
cv2.ocl.setUseOpenCL(False)

MOE_DEBUG = os.getenv("MOE_DEBUG", "0") not in ("0", "", "false", "False")

# make gate/ importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import broker as broker_mod           # uses Broker, YoloItem, DepthItem
import gate as gate_mod               # Gate().process_cb(...)
from adapter import compute_letterbox, depth_to_yolo_img

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-module", type=str, default="depth_exp.depth_expert",
                    help="module exposing stream_depth(share_color=Queue|None)")
    ap.add_argument("--yolo-module",  type=str, default="rgb_exp.yolo_trt_stream",
                    help="module exposing stream_yolo(engine_path, frame_queue=Queue|None)")
    ap.add_argument("--engine", type=str, default="rgb_exp/fulltilt.engine")
    ap.add_argument("--hold-last", action="store_true")
    ap.add_argument("--tol-frac", type=float, default=0.9)
    ap.add_argument("--tol-floor", type=float, default=0.010)
    ap.add_argument("--tol-ceiling", type=float, default=0.120)
    ap.add_argument("--fake-yolo", type=int, default=0)
    return ap.parse_args()

def put_overwrite(q, item):
    try:
        q.put_nowait(item)
    except Full:
        try:
            _ = q.get_nowait()
        except Empty:
            pass
        q.put_nowait(item)

def depth_worker(depth_module_path, out_q, stop_evt, share_color_q, share_color_ts_q):
    mod = importlib.import_module(depth_module_path)
    # Try new API (share_color_ts); fall back to legacy
    try:
        gen = mod.stream_depth(share_color=share_color_q, share_color_ts=share_color_ts_q)
    except TypeError:
        try:
            gen = mod.stream_depth(share_color=share_color_q)
        except TypeError:
            gen = mod.stream_depth()

    for item in gen:
        if stop_evt.is_set():
            break
        # Accept dict from depth_expert, or tuple legacy formats
        # Accept dict from depth_expert, or tuple legacy formats
        sync_evt = None
        gen_id   = -1

        if isinstance(item, dict):
            conf         = item.get("conf")
            plane_ok     = bool(item.get("plane_ok", True))
            inlier_ratio = float(item.get("inlier_ratio", 0.0))
            tstamp       = float(item.get("t", time.perf_counter()))
            sync_evt     = item.get("sync_evt", None)
            gen_id       = int(item.get("gen_id", -1))
            hw_ts        = float(item.get("hw_ts", 0.0))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            conf, plane_ok, inlier_ratio = item[:3]
            if len(item) >= 6:
                tstamp = item[3]
                sync_evt = item[4]
                gen_id  = int(item[5])
            elif len(item) >= 5:
                tstamp = item[3]
                sync_evt = item[4]
            else:
                tstamp = item[3] if len(item) >= 4 else time.perf_counter()
        else:
            # Extremely legacy / defensive fallback
            conf, plane_ok, inlier_ratio = item, True, 0.0
            tstamp = time.perf_counter()

        put_overwrite(out_q, broker_mod.DepthItem(
            t=tstamp, conf=conf, plane_ok=plane_ok, inlier_ratio=inlier_ratio,
            sync_evt=sync_evt, gen_id=gen_id, hw_ts=hw_ts))

def yolo_worker(yolo_module_path, engine_path, out_q, stop_evt, share_color_q, fake_yolo=0):
    if fake_yolo:
        import numpy as np
        H = W = 640
        t = 0.0; dt = 0.05
        while not stop_evt.is_set():
            t += dt
            y, x = np.ogrid[:H, :W]
            cx = int((np.sin(t) * 0.4 + 0.5) * W)
            cy = int((np.cos(t) * 0.4 + 0.5) * H)
            r = 80
            m = np.zeros((H, W), np.float32)
            m[(x-cx)**2 + (y-cy)**2 <= r*r] = 1.0
            put_overwrite(out_q, broker_mod.YoloItem(
                t=time.perf_counter(),
                dets=[broker_mod.YoloInst(mask_prob=m, cls=0, score=0.99)]))
            time.sleep(dt)
        return

    mod = importlib.import_module(yolo_module_path)
    stream_yolo = getattr(mod, "stream_yolo")
    for det_list in stream_yolo(engine_path, frame_queue=share_color_q):
        if stop_evt.is_set():
            break
        # det_list: list of (mask_640x640_float, cls_int, score_float)
        put_overwrite(out_q, broker_mod.YoloItem(
            t=time.perf_counter(),
            dets=[broker_mod.YoloInst(mask_prob=m, cls=c, score=s) for (m, c, s) in det_list]))

def gate_worker(gate_obj, in_q, stop_evt):
    """Background worker that runs the heavy Gate.process_cb asynchronously.

    It consumes GateInput objects from in_q and calls gate_obj.process_cb(gi).
    """
    while not stop_evt.is_set():
        try:
            gi = in_q.get(timeout=0.05)
        except Empty:
            continue
        gate_obj.process_cb(gi)

def main():
    args = parse_args()

    # queues from workers into broker + shared color (for viz)
    depth_q = Queue(maxsize=1)
    yolo_q  = Queue(maxsize=1)
    color_q = Queue(maxsize=1)      # passed into experts; not read here
    color_ts_q = Queue(maxsize=1)   # Broker-only (timestamped (t, BGR))

    stop_evt = Event()
    # Use larger gate queue for bag replay so nothing gets dropped.
    # Live camera keeps small queue (put_overwrite drops stale frames by design).
    IS_REPLAY = bool(os.getenv("MOE_REPLAY_BAG", ""))
    gate_q = Queue(maxsize=2 if IS_REPLAY else 2)

    # start workers (depth publishes color frames into color_q; yolo reads it)
    t_depth = Thread(target=depth_worker,
                    args=(args.depth_module, depth_q, stop_evt, color_q, color_ts_q),
                    name="depth_worker", daemon=True)
    t_yolo  = Thread(target=yolo_worker,
                     args=(args.yolo_module, args.engine, yolo_q, stop_evt, color_q, args.fake_yolo),
                     name="yolo_worker", daemon=True)
    t_depth.start(); t_yolo.start()

    # build broker (new API: no letterbox/depth mapping, no callbacks)
    broker = broker_mod.Broker(
        letterbox_fn=compute_letterbox,
        depth2yolo_img=depth_to_yolo_img,
        src_wh_depth=(640,480),
        dst_wh_yolo=(640,640),
        hold_last=args.hold_last,
        metrics_every=2.0,
        tol_floor=args.tol_floor,
        tol_ceiling=args.tol_ceiling,
        tol_fraction=args.tol_frac,
        max_queue=1
    )

    # gate turn viz on off here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    gate = gate_mod.Gate(visualize=False, window="MoE Fusion Viz")

    # --- Viz tuning: how often and how big to render ---
    gate._viz_every = 1      # 1 = draw every gate frame (no skipping)
    gate._viz_scale = 1.0    # 1.0 = full size; <1.0 to shrink viz and save CPU

    # asynchronous gate worker (no more blocking in Broker._try_pair)
    t_gate = Thread(
        target=gate_worker,
        args=(gate, gate_q, stop_evt),
        name="gate_worker",
        daemon=True,
    )
    t_gate.start()

    try:
        # Let the broker handle pairing and callbacks automatically
        broker.set_gate_callback(lambda gi: put_overwrite(gate_q, gi))
        broker.set_color_queue(color_ts_q)

        consecutive_empty = 0
        while True:
            # pump queues into broker
            got_y = got_d = False
            try:
                y = yolo_q.get(timeout=0.002)
                broker.push_yolo(y)
                got_y = True
                consecutive_empty = 0
            except Empty:
                pass
            try:
                d = depth_q.get(timeout=0.002)
                broker.push_depth(d)
                got_d = True
                consecutive_empty = 0
            except Empty:
                pass
            if not got_y and not got_d:
                consecutive_empty += 1
                time.sleep(0.001)
                # For bag replay: detect end and wait for gate queue to drain
                if (IS_REPLAY
                        and consecutive_empty > 500
                        and not t_depth.is_alive()
                        and not t_yolo.is_alive()):
                    print("[run_moe] Bag finished — draining gate queue...")
                    while not gate_q.empty():
                        time.sleep(0.05)
                    time.sleep(1.0)  # let gate worker finish last item
                    print("[run_moe] Gate queue drained. Exiting.")
                    break

    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        for _ in range(200):
            if not (t_depth.is_alive() or t_yolo.is_alive() or t_gate.is_alive()):
                break
            time.sleep(0.01)

if __name__ == "__main__":
    main()