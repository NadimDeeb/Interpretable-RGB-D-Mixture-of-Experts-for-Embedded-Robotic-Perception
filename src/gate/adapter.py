import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union

Array = np.ndarray

@dataclass
class Letterbox:
    src_wh: Tuple[int, int]
    dst_wh: Tuple[int, int]
    r: float
    pad: Tuple[int, int, int, int]
    new_wh: Tuple[int, int]

def compute_letterbox(src_wh: Tuple[int,int], dst_wh: Tuple[int,int]) -> Letterbox:
    sw, sh = src_wh; dw, dh = dst_wh
    r = min(dw / sw, dh / sh)
    new_w, new_h = int(round(sw * r)), int(round(sh * r))
    pad_w, pad_h = dw - new_w, dh - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top
    return Letterbox(src_wh, dst_wh, r, (left, top, right, bottom), (new_w, new_h))

# ------------------------
# OPTIMIZED Depth -> YOLO (letterbox)
# ------------------------
def depth_to_yolo_img(img: Array, lb: Letterbox, is_mask: bool = False, out: Array | None = None) -> Array:
    """Optimized mapping with pre-allocated buffers and faster interpolation"""
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    
    # Use pre-allocated output buffer if provided
    if out is None:
        if img.ndim == 2:
            canvas = np.zeros((lb.dst_wh[1], lb.dst_wh[0]), dtype=img.dtype)
        else:
            canvas = np.zeros((lb.dst_wh[1], lb.dst_wh[0], img.shape[2]), dtype=img.dtype)
    else:
        canvas = out
        canvas[...] = 0  # Clear canvas efficiently
    
    # Direct resize into target region (avoid intermediate array)
    L, T, R, B = lb.pad
    target_region = canvas[T:T + lb.new_wh[1], L:L + lb.new_wh[0]]
    
    # Resize directly into the target region
    cv2.resize(img, lb.new_wh, target_region, interpolation=interp)
    
    return canvas

def depth_to_yolo_boxes(boxes_xyxy: Array, lb: Letterbox) -> Array:
    """Vectorized box transformation"""
    if boxes_xyxy.size == 0: 
        return boxes_xyxy.copy()  # Avoid modifying input
    
    # Precompute scale and offset arrays
    scale = np.array([lb.r, lb.r, lb.r, lb.r], dtype=np.float32)
    offset = np.array([lb.pad[0], lb.pad[1], lb.pad[0], lb.pad[1]], dtype=np.float32)
    
    return boxes_xyxy.astype(np.float32) * scale + offset

def depth_to_yolo_points(pts_xy: Array, lb: Letterbox) -> Array:
    """Vectorized point transformation"""
    if pts_xy.size == 0: 
        return pts_xy.copy()
    
    scale_offset = np.array([lb.r, lb.r], dtype=np.float32)
    offset = np.array([lb.pad[0], lb.pad[1]], dtype=np.float32)
    
    return pts_xy.astype(np.float32) * scale_offset + offset

# ------------------------
# OPTIMIZED YOLO -> Depth (unletterbox)
# ------------------------
def yolo_to_depth_img(img_lb: Array, lb: Letterbox, is_mask: bool = False, out: Array | None = None) -> Array:
    """Optimized unletterboxing with direct cropping"""
    L, T, R, B = lb.pad
    H_dst, W_dst = lb.dst_wh[1], lb.dst_wh[0]
    
    # Extract the valid region
    cropped = img_lb[T:H_dst-B, L:W_dst-R]
    
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    
    if out is not None:
        # Resize directly into output buffer
        cv2.resize(cropped, lb.src_wh, out, interpolation=interp)
        return out
    else:
        return cv2.resize(cropped, lb.src_wh, interpolation=interp)

def yolo_to_depth_boxes(boxes_xyxy: Array, lb: Letterbox) -> Array:
    """Vectorized inverse box transformation with bounds checking"""
    if boxes_xyxy.size == 0: 
        return boxes_xyxy.copy()
    
    offset = np.array([lb.pad[0], lb.pad[1], lb.pad[0], lb.pad[1]], dtype=np.float32)
    r_inv = 1.0 / max(lb.r, 1e-6)
    
    # Vectorized transformation
    scaled = (boxes_xyxy.astype(np.float32) - offset) * r_inv
    
    # Clip to source bounds
    W, H = lb.src_wh
    scaled[:, [0,2]] = np.clip(scaled[:, [0,2]], 0, W-1)
    scaled[:, [1,3]] = np.clip(scaled[:, [1,3]], 0, H-1)
    
    return scaled

def yolo_to_depth_points(pts_xy: Array, lb: Letterbox) -> Array:
    """Vectorized inverse point transformation"""
    if pts_xy.size == 0: 
        return pts_xy.copy()
    
    offset = np.array([lb.pad[0], lb.pad[1]], dtype=np.float32)
    r_inv = 1.0 / max(lb.r, 1e-6)
    
    return (pts_xy.astype(np.float32) - offset) * r_inv

# ------------------------
# OPTIMIZED Convenience helpers
# ------------------------
# Pre-compute common letterbox configurations
_COMMON_LETTERBOXES = {}

def get_letterbox(src_wh, dst_wh):
    """Cache common letterbox configurations"""
    key = (src_wh, dst_wh)
    if key not in _COMMON_LETTERBOXES:
        _COMMON_LETTERBOXES[key] = compute_letterbox(src_wh, dst_wh)
    return _COMMON_LETTERBOXES[key]

def map_depth_conf_to_yolo(conf_depth01: Array, dst_wh=(640,640), out: Array | None = None) -> Array:
    """Optimized with cached letterbox"""
    lb = get_letterbox((conf_depth01.shape[1], conf_depth01.shape[0]), dst_wh)
    return depth_to_yolo_img(conf_depth01, lb, is_mask=False, out=out)

def map_yolo_mask_to_depth(mask_prob_or_bin: Array, src_wh=(640,640), out: Array | None = None) -> Array:
    """Optimized with cached letterbox"""
    lb = get_letterbox((640,480), src_wh)
    return yolo_to_depth_img(mask_prob_or_bin, lb, is_mask=True, out=out)