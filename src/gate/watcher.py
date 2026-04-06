#!/usr/bin/env python3
import argparse, importlib.util, os, sys, math
from dataclasses import dataclass
from typing import Optional, Tuple, List

def die(msg, code=1):
    print(f"[ERROR] {msg}"); sys.exit(code)

# -------- Depth Expert config --------
@dataclass
class DepthInfo:
    W:int; H:int; FPS:int; USE_COLOR:bool

def read_depth_info(depth_py:str)->DepthInfo:
    spec = importlib.util.spec_from_file_location("depth_mod", depth_py)
    if spec is None: die(f"Cannot load depth module from: {depth_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return DepthInfo(
        W=int(getattr(mod,"W",640)),
        H=int(getattr(mod,"H",480)),
        FPS=int(getattr(mod,"FPS",30)),
        USE_COLOR=bool(getattr(mod,"USE_COLOR",True))
    )

# -------- TensorRT inspect (new + old API) --------
@dataclass
class TrtBinding:
    name:str; is_input:bool; dtype:str
    shape_decl:Tuple[int,...]
    prof_min:Optional[Tuple[int,...]]; prof_opt:Optional[Tuple[int,...]]; prof_max:Optional[Tuple[int,...]]

@dataclass
class TrtInfo:
    version:str; name:Optional[str]
    num_profiles:int
    inputs:List[TrtBinding]; outputs:List[TrtBinding]

def _np_dtype_name(dt):
    import numpy as np, tensorrt as trt
    try: return np.dtype(trt.nptype(dt)).name
    except Exception: return str(dt)

def read_trt_info(engine_path:str, profile_idx:Optional[int])->TrtInfo:
    import tensorrt as trt
    if not os.path.isfile(engine_path): die(f"Engine not found: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path,"rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())

    ver = getattr(trt,"__version__","unknown")
    nprof = getattr(engine, "num_optimization_profiles", 0)
    if profile_idx is not None and not (0 <= profile_idx < nprof):
        print(f"[WARN] Profile {profile_idx} out of range 0..{nprof-1}; using 0")
        profile_idx = 0
    if profile_idx is None: profile_idx = 0

    inputs, outputs = [], []

    if hasattr(engine, "num_io_tensors"):  # NEW API (TRT ≥ 9/10)
        N = engine.num_io_tensors
        for i in range(N):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)   # trt.TensorIOMode.INPUT/OUTPUT
            is_in = (int(mode) == int(trt.TensorIOMode.INPUT))
            dt = engine.get_tensor_dtype(name)
            shp_decl = tuple(int(x) for x in engine.get_tensor_shape(name))
            # profile shapes for inputs only (if dynamic)
            pmin=popt=pmax=None
            if is_in and nprof>0:
                try:
                    lo,op,hi = engine.get_profile_shape(profile_idx, name)
                    pmin = tuple(int(x) for x in lo)
                    popt = tuple(int(x) for x in op)
                    pmax = tuple(int(x) for x in hi)
                except Exception:
                    pass
            b = TrtBinding(name,is_in,_np_dtype_name(dt),shp_decl,pmin,popt,pmax)
            (inputs if is_in else outputs).append(b)
        eng_name = getattr(engine,"name",None)

    else:  # OLD API (TRT ≤ 8.x)
        N = engine.num_bindings
        for i in range(N):
            name = engine.get_binding_name(i)
            is_in = engine.binding_is_input(i)
            dt = engine.get_binding_dtype(i)
            shp_decl = tuple(int(x) for x in engine.get_binding_shape(i))
            pmin=popt=pmax=None
            if is_in and nprof>0:
                try:
                    lo,op,hi = engine.get_profile_shape(profile_idx, i)
                    pmin = tuple(int(x) for x in lo)
                    popt = tuple(int(x) for x in op)
                    pmax = tuple(int(x) for x in hi)
                except Exception:
                    pass
            b = TrtBinding(name,is_in,_np_dtype_name(dt),shp_decl,pmin,popt,pmax)
            (inputs if is_in else outputs).append(b)
        eng_name = getattr(engine,"name",None)

    return TrtInfo(ver, eng_name, nprof, inputs, outputs)

# -------- Compatibility verdict --------
def aspect(w,h): return w/float(h)

def letterbox_map(src_wh, dst_wh):
    sw,sh = src_wh; dw,dh = dst_wh
    r = min(dw/sw, dh/sh)
    new_w, new_h = int(round(sw*r)), int(round(sh*r))
    pad_w, pad_h = dw-new_w, dh-new_h
    left, top = pad_w//2, pad_h//2
    return r,(left,top),(new_w,new_h)

def pick_image_input(bindings:List[TrtBinding])->Optional[TrtBinding]:
    for b in bindings:
        shp = b.prof_opt or b.shape_decl
        if not shp: continue
        # heuristics: rank 4/3 with channel==1 or 3 on either side
        if len(shp)==4:
            if shp[1] in (1,3) or shp[-1] in (1,3): return b
        if len(shp)==3:
            if shp[0] in (1,3) or shp[-1] in (1,3): return b
    return None

def image_wh(binding:TrtBinding)->Optional[Tuple[int,int]]:
    shp = binding.prof_opt or binding.shape_decl
    if len(shp)==4:  # try NCHW/NHWC
        if shp[1] in (1,3): return (shp[3], shp[2])  # NCHW
        else:               return (shp[2], shp[1])  # NHWC
    if len(shp)==3:  # CHW/HWC
        if shp[0] in (1,3): return (shp[2], shp[1])  # CHW
        else:               return (shp[1], shp[0])  # HWC
    return None

def verdict(depth:DepthInfo, trt:TrtInfo)->str:
    img_in = pick_image_input(trt.inputs)
    if img_in is None:
        return "❓ Could not identify the image input from the engine."
    iw, ih = image_wh(img_in)
    dw, dh = depth.W, depth.H
    lines = []
    lines.append(f"Depth Expert:  {dw}x{dh} @ {depth.FPS} FPS   USE_COLOR={depth.USE_COLOR}")
    lines.append(f"TRT Image In:  {iw}x{ih} (opt/dcl)")
    lines.append(f"Aspect Ratios: TRT={iw/ih:.3f}  vs  Depth={dw/dh:.3f}")
    lines.append("")
    if (iw==dw and ih==dh):
        lines.append("✅ Spatial compatibility: SAME resolution. No remap needed.")
    elif abs(aspect(iw,ih)-aspect(dw,dh))<1e-3:
        lines.append("🟡 Same aspect ratio, different sizes → simple resize OK.")
        lines.append(f"   Suggest: resize depth conf → {iw}x{ih}, or feed TRT with {dw}x{dh}.")
    else:
        lines.append("🔴 Aspect ratio mismatch (e.g., 640x640 vs 640x480). Use letterbox mapping.")
        r,(lx,ly),(nw,nh) = letterbox_map((dw,dh),(iw,ih))
        lines.append(f"   Letterbox {dw}x{dh} → {iw}x{ih}: scale r={r:.6f}, new={nw}x{nh}, "
                     f"pad L={lx} T={ly} R={iw-nw-lx} B={ih-nh-ly}")
        lines.append("   Map YOLO→Depth: unpad, scale by 1/r back to 640x480.")
        lines.append("   Map Depth→YOLO: scale by r, then pad (L,T,R,B) to 640x640.")
    return "\n".join(lines)

# -------- Main --------
def main():
    ap = argparse.ArgumentParser("Experts Watcher (TRT new/old compatible)")
    ap.add_argument("--engine","-e",required=True)
    ap.add_argument("--depth","-d",default="depth_expert.py")
    ap.add_argument("--profile","-p",type=int,default=None)
    args = ap.parse_args()

    depth = read_depth_info(args.depth)
    trtinfo = read_trt_info(args.engine, args.profile)

    print("=== Experts Watcher ===")
    print(f"TensorRT: {trtinfo.version}")
    if trtinfo.name: print(f"Engine:   {trtinfo.name}")
    print(f"Profiles: {trtinfo.num_profiles}\n")

    print("Inputs:")
    for b in trtinfo.inputs:
        shp = b.prof_opt or b.shape_decl
        extra = ""
        if b.prof_min or b.prof_opt or b.prof_max:
            extra = f"  min/opt/max={b.prof_min}/{b.prof_opt}/{b.prof_max}"
        print(f"  - {b.name:24s} {str(shp):>15s}  dtype={b.dtype}{extra}")
    print("Outputs:")
    for b in trtinfo.outputs:
        shp = b.prof_opt or b.shape_decl
        print(f"  - {b.name:24s} {str(shp):>15s}  dtype={b.dtype}")

    print()
    print(verdict(depth, trtinfo))

if __name__ == "__main__":
    main()
