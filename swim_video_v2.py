"""
swim_video_v2.py
水上/水下游泳视频融合工具 - 工业级音视频对齐版 (音频同步 + AI头部打点 + 10x4抽样)

核心架构：
Step 1: 视频配对与颠倒检测。
Step 2: 时间同步：提取两个视频的音频轨道，进行 FFT 互相关，实现毫秒级绝对时间对齐。
Step 3: 空间定标 (10段x4帧)：
        - 居中放大：优先对水上视频应用放大倍率（拉近视角）。
        - 水位线：沿用最稳健的第一代物理环境特征算法 (上: Sobel梯度, 下: 镜像反射轴)。
        - 横向偏移：使用 MediaPipe 骨骼追踪，提取“鼻子”绝对 X 坐标。
Step 4: 全局静态融合（抗折射重构版）：
        - 引入 TOP/BOTTOM_WATERLINE_OFFSET，将上下画面拉开，重构因水面折射丢失的躯干。
        - 羽化过渡 (FEATHER_PX) 将严格限制在额外拉开的像素带中，绝对不侵占已保留的画面主体。
使用教程：
python swim_video_v2.py --batch-top video_merge\video1_min --batch-bottom video_merge\video2_min --batch-output video_merge\融合
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import imageio.v2 as iio
import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, butter, sosfilt
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, PoseLandmark
    from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as vrm
    RunningMode = getattr(vrm, 'RunningMode', vrm.VisionTaskRunningMode)
    PoseLandmarkerResult = None 
except ImportError:
    print("请先安装 MediaPipe: pip install mediapipe")
    exit(1)

# ============================================================
# 融合参数配置
# ============================================================
TOP_ZOOM_FACTOR = 1.15             # 水上视频全局放大倍率（拉近镜头，主体更大）
TOP_WATERLINE_OFFSET = 10         # 针对水上视频：将水线向下扩展的像素大小（展示更多人物上半身）
BOTTOM_WATERLINE_OFFSET = 10      # 针对水下视频：将水线向上扩展的像素大小（展示更多人物下半身）
FEATHER_PX = 40                    # 融合羽化宽度（在新拓展的重合带内进行过渡，0为硬切，不占有已有内容）
BLUE, GREEN, RED, YELLOW = (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)
PINK = (180, 105, 255)       # 粉色 - 头部
ORANGE = (0, 165, 255)       # 橙色 - 肩膀
PURPLE = (255, 0, 180)       # 紫色 - 臀部
CYAN = (255, 255, 0)         # 青色 - 膝盖
WHITE = (255, 255, 255)      # 白色 - 脚踝
GRAY = (128, 128, 128)       # 灰色 - 连接线

@dataclass
class VideoMeta:
    path: Path; fps: float; width: int; height: int; duration: float; frame_count: int

@dataclass
class FusionConfig:
    seam: int; feather_px: int; horizontal_offset: int; time_offset_sec: float
    top_rotate_180: bool; bottom_rotate_180: bool
    raw_top_waterline: int; raw_bottom_waterline: int
    top_waterline_offset: int; bottom_waterline_offset: int
    top_zoom_factor: float

# ============================================================
# 基础工具
# ============================================================
def rotate_frame(frame: np.ndarray, rotate_180: bool) -> np.ndarray:
    return cv2.rotate(frame, cv2.ROTATE_180) if rotate_180 else frame

def load_video_meta(path: Path) -> VideoMeta:
    reader = iio.get_reader(str(path))
    meta = reader.get_meta_data()
    reader.close()
    fps = float(meta.get("fps", 30.0))
    w, h = meta.get("size", (0, 0))
    duration = float(meta.get("duration", 0.0))
    fc = meta.get("nframes")
    return VideoMeta(path, fps, int(w), int(h), duration, int(fc) if fc and fc != float("inf") else int(duration * fps))

def sample_frame_at(reader, time_sec: float, fps: float) -> np.ndarray | None:
    if time_sec < 0: return None
    try: return reader.get_data(int(round(time_sec * fps)))
    except: return None

def apply_top_zoom(frame: np.ndarray, zoom_factor: float) -> np.ndarray:
    if zoom_factor == 1.0: return frame
    h, w = frame.shape[:2]
    new_w, new_h = int(round(w * zoom_factor)), int(round(h * zoom_factor))
    zoomed = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if zoom_factor > 1.0:
        x_start, y_start = (new_w - w) // 2, (new_h - h) // 2
        return zoomed[y_start:y_start+h, x_start:x_start+w]
    else:
        canvas = np.zeros((h, w, frame.shape[2]), dtype=frame.dtype)
        x_start, y_start = (w - new_w) // 2, (h - new_h) // 2
        canvas[y_start:y_start+new_h, x_start:x_start+new_w] = zoomed
        return canvas

# ============================================================
# Step 1: 颠倒检测 - 基于水位线视觉特征
# ============================================================

def analyze_waterline_color_features(frame: np.ndarray, waterline_row: int, search_range: int = 30) -> dict:
    """
    分析水位线附近的颜色特征
    返回：上方和下方区域的蓝色强度、波动程度等信息
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 转换到HSV来分析蓝色
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # 蓝色范围 (室内泳池通常是蓝色/青色)
    # 提高亮度V下限至80，避免将黑色的室内环境误判为蓝色
    lower_blue = np.array([90, 40, 80])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 定义分析区域
    upper_start = max(0, waterline_row - search_range)
    upper_end = waterline_row
    lower_start = waterline_row
    lower_end = min(h, waterline_row + search_range)
    
    # 计算各区域的蓝色比例和平均亮度
    upper_blue_ratio = np.mean(blue_mask[upper_start:upper_end] > 0) if upper_end > upper_start else 0
    lower_blue_ratio = np.mean(blue_mask[lower_start:lower_end] > 0) if lower_end > lower_start else 0
    
    upper_brightness = np.mean(gray[upper_start:upper_end]) if upper_end > upper_start else 0
    lower_brightness = np.mean(gray[lower_start:lower_end]) if lower_end > lower_start else 0
    
    # 计算波动程度（使用边缘密度作为波动的近似）
    upper_edges = cv2.Canny(gray[upper_start:upper_end], 50, 150) if upper_end > upper_start else np.array([])
    lower_edges = cv2.Canny(gray[lower_start:lower_end], 50, 150) if lower_end > lower_start else np.array([])
    upper_edge_density = np.mean(upper_edges > 0) if upper_edges.size > 0 else 0
    lower_edge_density = np.mean(lower_edges > 0) if lower_edges.size > 0 else 0
    
    return {
        "upper_blue_ratio": float(upper_blue_ratio),
        "lower_blue_ratio": float(lower_blue_ratio),
        "upper_brightness": float(upper_brightness),
        "lower_brightness": float(lower_brightness),
        "upper_edge_density": float(upper_edge_density),
        "lower_edge_density": float(lower_edge_density),
    }


def detect_top_video_rotation_by_waterline(
    frame: np.ndarray, 
    detected_waterline: int,
    debug_info: dict = None
) -> dict:
    """
    水上视频旋转检测：通过水位线颜色和人头位置判断
    
    原理：
    - 室内泳池，水是蓝色的，水位线下方应该是蓝色的水
    - 增加人头判断：人头应该在水位线上方
    
    返回：{"rotated": bool, "reason": str, "scores": dict}
    """
    h, w = frame.shape[:2]
    features = analyze_waterline_color_features(frame, detected_waterline, search_range=40)
    
    # 1. 环境特征判断 (综合考虑蓝色、亮度、边缘波动)
    # 因为室内环境可能是黑色的，如果只用蓝色阈值可能受噪点影响，水面反光通常更亮且有波动。
    blue_diff = features["lower_blue_ratio"] - features["upper_blue_ratio"]
    bright_diff = (features["lower_brightness"] - features["upper_brightness"]) / 255.0
    edge_diff = features["lower_edge_density"] - features["upper_edge_density"]
    
    normal_score = blue_diff * 0.4 + bright_diff * 0.4 + edge_diff * 0.2
    is_rotated_env = normal_score < 0
    
    # 2. 人头判断
    head_y = None
    head_x = None
    try:
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, PoseLandmark
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
        
        model_path = Path.home() / ".mediapipe" / "models" / "pose_landmarker_lite.task"
        if model_path.exists():
            options = PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=PoseLandmarkerOptions.running_mode.IMAGE
            )
            landmarker = PoseLandmarker.create_from_options(options)
            img = Image(image_format=ImageFormat.SRGB, data=frame)
            result = landmarker.detect(img)
            landmarker.close()
            
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                head_points_y = []
                head_points_x = []
                # 提取鼻子、眼睛、耳朵作为头部关键点
                for point_idx in [PoseLandmark.NOSE, PoseLandmark.LEFT_EYE, PoseLandmark.RIGHT_EYE, PoseLandmark.LEFT_EAR, PoseLandmark.RIGHT_EAR]:
                    landmark = lm[point_idx]
                    if landmark.visibility > 0.3:
                        head_points_y.append(landmark.y * h)
                        head_points_x.append(landmark.x * w)
                if head_points_y:
                    head_y = np.mean(head_points_y)
                    head_x = np.mean(head_points_x)
    except Exception as e:
        pass

    # 综合判断
    is_rotated = is_rotated_env
    reason_parts = []
    
    if is_rotated_env:
        reason_parts.append(f"环境特征显示翻转(上方蓝/亮/波纹多)")
    else:
        reason_parts.append(f"环境特征显示正常(下方蓝/亮/波纹多)")

    if head_y is not None:
        if head_y < detected_waterline:
            # 人头在水位线上方，说明正常
            is_rotated_head = False
            reason_parts.append(f"人头(Y={head_y:.0f})在水位线({detected_waterline})上方(正常)")
        else:
            # 人头在水位线下方，说明翻转
            is_rotated_head = True
            reason_parts.append(f"人头(Y={head_y:.0f})在水位线({detected_waterline})下方(翻转)")
        
        # 只要人头判断明确，优先以人头为准，或者结合考虑。这里明确以人头为准。
        is_rotated = is_rotated_head
        reason_parts.append("=> 综合以人头位置为准")
    else:
        reason_parts.append("未检测到清晰人头，仅以颜色为准")

    reason = "；".join(reason_parts)

    result = {
        "rotated": is_rotated,
        "reason": reason,
        "scores": features,
        "waterline": detected_waterline,
        "head_y": head_y,
        "head_x": head_x
    }
    
    if debug_info is not None:
        debug_info.update(result)
    
    return result


def find_human_in_underwater_frame(frame: np.ndarray, detected_waterline: int) -> dict:
    """
    使用MediaPipe检测水下图像中的人体姿态，确定人体位置
    """
    try:
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, PoseLandmark
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
    except ImportError:
        return _find_human_by_contours(frame, detected_waterline)
    
    model_path = Path.home() / ".mediapipe" / "models" / "pose_landmarker_lite.task"
    if not model_path.exists():
        return _find_human_by_contours(frame, detected_waterline)
    
    options = PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=PoseLandmarkerOptions.running_mode.IMAGE
    )
    landmarker = PoseLandmarker.create_from_options(options)
    
    img = Image(image_format=ImageFormat.SRGB, data=frame)
    result = landmarker.detect(img)
    landmarker.close()
    
    if not result.pose_landmarks:
        return _find_human_by_contours(frame, detected_waterline)
    
    lm = result.pose_landmarks[0]
    h, w = frame.shape[:2]
    
    # 计算人体中心点（使用所有可见关键点的中心）
    visible_points = []
    key_points_y = []  # 收集所有关键点的Y坐标
    
    # 头部
    nose = lm[PoseLandmark.NOSE]
    if nose.visibility > 0.3:
        visible_points.append(("nose", nose.y * h))
        key_points_y.append(nose.y * h)
    
    # 肩膀
    for name, landmark in [("left_shoulder", lm[PoseLandmark.LEFT_SHOULDER]),
                           ("right_shoulder", lm[PoseLandmark.RIGHT_SHOULDER])]:
        if landmark.visibility > 0.3:
            visible_points.append((name, landmark.y * h))
            key_points_y.append(landmark.y * h)
    
    # 臀部
    for name, landmark in [("left_hip", lm[PoseLandmark.LEFT_HIP]),
                           ("right_hip", lm[PoseLandmark.RIGHT_HIP])]:
        if landmark.visibility > 0.3:
            visible_points.append((name, landmark.y * h))
            key_points_y.append(landmark.y * h)
    
    # 膝盖
    for name, landmark in [("left_knee", lm[PoseLandmark.LEFT_KNEE]),
                           ("right_knee", lm[PoseLandmark.RIGHT_KNEE])]:
        if landmark.visibility > 0.3:
            visible_points.append((name, landmark.y * h))
            key_points_y.append(landmark.y * h)
    
    # 脚踝
    for name, landmark in [("left_ankle", lm[PoseLandmark.LEFT_ANKLE]),
                           ("right_ankle", lm[PoseLandmark.RIGHT_ANKLE])]:
        if landmark.visibility > 0.3:
            visible_points.append((name, landmark.y * h))
            key_points_y.append(landmark.y * h)
    
    if not visible_points:
        return _find_human_by_contours(frame, detected_waterline)
    
    # 人体中心Y坐标（所有关键点的平均值）
    human_center_y = np.mean(key_points_y)
    
    # 人体的上下边界
    human_top = min(key_points_y)
    human_bottom = max(key_points_y)
    
    # 判断人体相对于水位线的位置
    # 使用人体中心点判断
    if human_center_y < detected_waterline - 20:
        location = "above_waterline"
    elif human_center_y > detected_waterline + 20:
        location = "below_waterline"
    else:
        # 人体中心在水位线附近，检查人体上下边界
        if human_bottom < detected_waterline:
            location = "above_waterline"
        elif human_top > detected_waterline:
            location = "below_waterline"
        else:
            location = "straddle"
    
    details = (
        f"MediaPipe检测到{len(visible_points)}个人体关键点, "
        f"人体中心Y={human_center_y:.0f}, 范围=[{human_top:.0f}, {human_bottom:.0f}], "
        f"水位线={detected_waterline}"
    )
    
    return {
        "has_human": True,
        "human_center_y": human_center_y,
        "human_top": human_top,
        "human_bottom": human_bottom,
        "confidence": min(1.0, len(visible_points) / 6),  # 最多6个关键点
        "location": location,
        "details": details,
        "landmarks": visible_points,
        "bbox": None,
        "candidates": [],
        "method": "mediapipe"
    }


def _find_human_by_contours(frame: np.ndarray, detected_waterline: int) -> dict:
    """
    使用轮廓检测作为备选方法
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 边缘检测
    edges = cv2.Canny(gray, 30, 100)
    
    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找最大的几个轮廓
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    human_candidates = []
    for cnt in contours_sorted:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        
        x, y, cw, ch = cv2.boundingRect(cnt)
        center_y = y + ch / 2
        human_candidates.append({
            "bbox": (x, y, cw, ch),
            "center_y": center_y,
            "area": area,
            "score": min(1.0, area / 5000)
        })
    
    if not human_candidates:
        return {
            "has_human": False,
            "human_center_y": None,
            "confidence": 0.0,
            "location": None,
            "details": "未检测到人形",
            "bbox": None,
            "candidates": []
        }
    
    # 找出最大/最明显的候选
    best = human_candidates[0]
    
    # 判断位置
    center_y = best["center_y"]
    if center_y < detected_waterline - 20:
        location = "above_waterline"
    elif center_y > detected_waterline + 20:
        location = "below_waterline"
    else:
        location = "on_waterline"
    
    return {
        "has_human": True,
        "human_center_y": center_y,
        "confidence": best["score"],
        "location": location,
        "details": f"轮廓检测: 中心Y={center_y:.0f}, 水位线={detected_waterline}, 面积={best['area']:.0f}",
        "bbox": best["bbox"],
        "candidates": human_candidates,
        "method": "contours"
    }


def detect_underwater_video_rotation_by_waterline(
    frame: np.ndarray, 
    detected_waterline: int,
    debug_info: dict = None,
    debug_dir: Path = None,
    frame_idx: int = 0
) -> dict:
    """
    水下视频旋转检测：检测整幅图像中的人体，判断其相对于分界线的位置
    
    原理：
    - 水下拍摄时，运动员在水下（分界线下方）
    - 如果运动员出现在分界线上方，说明视频被上下翻转了
    
    判断：
    - 人体在分界线上方 → 需要翻转
    - 人体在分界线下方 → 正常
    """
    h, w = frame.shape[:2]
    
    # 检测水下图像中的人体
    human_result = find_human_in_underwater_frame(frame, detected_waterline)
    
    # 输出到控制台
    print(f"\n  === 水下旋转检测 帧 {frame_idx} ===")
    print(f"  水位线位置: {detected_waterline}")
    print(f"  {human_result['details']}")
    
    # 判断是否翻转
    if not human_result["has_human"]:
        # 无法检测到人体，保守处理
        is_rotated = False
        verdict = "无法判断（未检测到人体）"
        detail = "未能检测到明确的人形，假设为正常"
        location_text = "未知"
    else:
        location = human_result["location"]
        location_text = {
            "above_waterline": "分界线上方",
            "below_waterline": "分界线下方",
            "on_waterline": "分界线附近"
        }.get(location, "未知")
        
        # 人体在分界线上方 → 需要翻转
        is_rotated = (location == "above_waterline")
        
        if is_rotated:
            verdict = "需要翻转180°"
            detail = f"人体位于{location_text}，说明画面上下颠倒"
        elif location == "below_waterline":
            verdict = "正常（不需要翻转）"
            detail = f"人体位于{location_text}，说明画面方向正确"
        else:
            verdict = "位置不明确"
            detail = f"人体位于{location_text}，无法确定"
    
    print(f"  [判断结果] {verdict}")
    print(f"  [判断依据] {detail}")
    
    result = {
        "rotated": is_rotated,
        "reason": f"{verdict}。{detail}。",
        "verdict": verdict,
        "human_detection": human_result,
        "waterline": detected_waterline
    }
    
    if debug_info is not None:
        debug_info.update(result)
    
    # 生成调试图像
    if debug_dir is not None:
        annotated = frame.copy()
        
        # 绘制水位线（粗黄线）
        cv2.line(annotated, (0, detected_waterline), (w, detected_waterline), YELLOW, 3)
        
        # 标注水位线
        cv2.putText(annotated, f"水位线 Y={detected_waterline}", 
                   (10, detected_waterline - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
        
        # 绘制检测到的人体（边框和中心点）
        if human_result["has_human"] and human_result["bbox"]:
            x, y, cw, ch = human_result["bbox"]
            
            # 绿色边框表示水下（正确），红色边框表示水上（需要翻转）
            box_color = GREEN if not is_rotated else RED
            cv2.rectangle(annotated, (x, y), (x + cw, y + ch), box_color, 3)
            
            # 中心点
            cx = x + cw // 2
            cy = int(human_result["human_center_y"])
            cv2.circle(annotated, (cx, cy), 10, box_color, -1)
            cv2.circle(annotated, (cx, cy), 15, box_color, 2)
            
            # 标注人体类型
            label = human_result.get("candidates", [{}])[0].get("type", "human") if human_result.get("candidates") else "human"
            cv2.putText(annotated, f"人体:{label}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # 绘制所有候选区域（浅色）
        for cand in human_result.get("candidates", [])[:5]:
            if cand.get("bbox"):
                x, y, cw, ch = cand["bbox"]
                cv2.rectangle(annotated, (x, y), (x + cw, y + ch), GRAY, 1)
        
        # 绘制MediaPipe骨骼
        landmarks = human_result.get("landmarks", [])
        if landmarks and human_result.get("method") == "mediapipe":
            lm_dict = {name: y for name, y in landmarks}
            lm_colors = {
                "nose": PINK, "left_shoulder": ORANGE, "right_shoulder": ORANGE,
                "left_hip": PURPLE, "right_hip": PURPLE,
                "left_knee": CYAN, "right_knee": CYAN,
                "left_ankle": WHITE, "right_ankle": WHITE,
            }
            connections = [
                ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"),
                ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
                ("left_knee", "left_hip"), ("right_knee", "right_hip"),
                ("left_ankle", "left_knee"), ("right_ankle", "right_knee"),
            ]
            x_center = w // 2
            for a, b in connections:
                if a in lm_dict and b in lm_dict:
                    cv2.line(annotated, (x_center, int(lm_dict[a])), (x_center, int(lm_dict[b])), GRAY, 2)
            for name, y in landmarks:
                color = lm_colors.get(name, WHITE)
                cv2.circle(annotated, (x_center, int(y)), 8, color, -1)
                cv2.circle(annotated, (x_center, int(y)), 10, WHITE, 2)
        
        y_offset = 30
        result_color = GREEN if not is_rotated else RED
        
        # 标题
        cv2.putText(annotated, f"[帧{frame_idx}] {verdict}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
        y_offset += 35
        
        # 水位线
        cv2.putText(annotated, f"水位线: Y={detected_waterline}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        y_offset += 28
        
        # 人体位置
        if human_result["has_human"]:
            cv2.putText(annotated, f"人体Y坐标: {human_result['human_center_y']:.0f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PINK, 1)
            y_offset += 28
            cv2.putText(annotated, f"位置: {location_text}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, result_color, 1)
            y_offset += 28
            cv2.putText(annotated, f"置信度: {human_result['confidence']:.2f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        else:
            cv2.putText(annotated, "未检测到人体", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
        
        # 保存调试图像
        output_path = debug_dir / f"rotation_check_underwater_frame_{frame_idx:03d}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"  [调试图像已保存] {output_path}")
    
    return result


def is_top_camera_rotated(frames: list[np.ndarray], debug_suffix: str = "", debug_dir: Path = None) -> tuple[bool, dict]:
    """
    水上视频旋转检测 - 基于水位线蓝色特征
    
    返回: (是否翻转, 详细信息字典)
    """
    normal_count, flipped_count, no_waterline_count = 0, 0, 0
    frame_details = []
    
    for idx, frame in enumerate(frames):
        # 先找水位线
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        waterline = find_best_waterline_row(gray)
        
        frame_info = {
            "frame_idx": idx,
            "waterline": waterline,
            "verdict": "unknown",
            "reason": ""
        }
        
        if waterline is None:
            no_waterline_count += 1
            frame_info["verdict"] = "unknown"
            frame_info["reason"] = "未能检测到水位线"
            frame_details.append(frame_info)
            continue
        
        # 基于水位线颜色特征判断
        debug_info = {}
        result = detect_top_video_rotation_by_waterline(frame, waterline, debug_info)
        
        if result["rotated"]:
            flipped_count += 1
            frame_info["verdict"] = "flipped"
        else:
            normal_count += 1
            frame_info["verdict"] = "normal"
        
        frame_info["reason"] = result["reason"]
        frame_info["human_detection"] = result.get("human_detection", {})
        frame_info["scores"] = result.get("human_detection", {})
        frame_details.append(frame_info)
        
        # 保存调试图像
        if debug_dir is not None:
            annotated = frame.copy()
            h, w = frame.shape[:2]
            
            # 绘制水位线
            cv2.line(annotated, (0, waterline), (w, waterline), YELLOW, 2)
            cv2.putText(annotated, f"Waterline Y={waterline}", (w - 200, waterline - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
            
            # 绘制人头位置
            head_y = result.get("head_y")
            head_x = result.get("head_x")
            if head_y is not None and head_x is not None:
                cx, cy = int(head_x), int(head_y)
                box_color = GREEN if not result["rotated"] else RED
                cv2.circle(annotated, (cx, cy), 15, box_color, -1)
                cv2.circle(annotated, (cx, cy), 20, box_color, 3)
                cv2.putText(annotated, f"Head Y={cy}", (cx + 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # 标注颜色比例和判断结果
            scores = result["scores"]
            y_offset = 30
            
            # 主标题（判断结果）
            verdict_text = "Flipped (180 deg)" if result['rotated'] else "Normal"
            result_color = GREEN if not result["rotated"] else RED
            cv2.putText(annotated, f"[Frame {idx}] {verdict_text}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
            y_offset += 35
            
            # 详细原因
            # 将 reason 按分号拆分多行显示
            for reason_line in result["reason"].split("；"):
                cv2.putText(annotated, reason_line, 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                y_offset += 25
                
            y_offset += 10
            cv2.putText(annotated, f"Upper Bright: {scores['upper_brightness']:.0f} | Lower Bright: {scores['lower_brightness']:.0f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
            y_offset += 25
            cv2.putText(annotated, f"Upper Blue: {scores['upper_blue_ratio']:.2f} | Lower Blue: {scores['lower_blue_ratio']:.2f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
            y_offset += 25
            cv2.putText(annotated, f"Upper Edge: {scores['upper_edge_density']:.3f} | Lower Edge: {scores['lower_edge_density']:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PINK, 1)
            y_offset += 25
            cv2.imwrite(str(debug_dir / f"rotation_check_top_{debug_suffix}_frame_{idx:02d}.jpg"),
                       cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    
    total_valid = normal_count + flipped_count
    rotated = flipped_count > normal_count
    
    details = {
        "total_frames": len(frames),
        "valid_detections": total_valid,
        "no_waterline": no_waterline_count,
        "normal_count": normal_count,
        "flipped_count": flipped_count,
        "final_verdict": "flipped" if rotated else "normal",
        "frame_details": frame_details,
        "explanation": ""
    }
    
    if rotated:
        details["explanation"] = (
            f"检测到视频需要翻转180°。\n"
            f"原因：在{len(frames)}帧中，有{flipped_count}帧判断为翻转，"
            f"而{normal_count}帧判断为正常。\n"
            f"翻转意味着视频画面上下颠倒，需要旋转180°才能恢复正常。"
        )
    else:
        details["explanation"] = (
            f"检测到视频不需要翻转。\n"
            f"原因：在{len(frames)}帧中，有{normal_count}帧判断为正常，"
            f"而{flipped_count}帧判断为翻转。\n"
            f"这表明视频方向正确，无需旋转。"
        )
    
    print(f"   [水上旋转检测] 有效:{total_valid}帧 (未检测水线:{no_waterline_count}), 正常:{normal_count}, 翻转:{flipped_count} => {'翻转180°' if rotated else '正常'}")
    print(f"   详细: {details['explanation']}")
    
    return rotated, details


def is_underwater_camera_rotated(frames: list[np.ndarray], debug_suffix: str = "", debug_dir: Path = None) -> tuple[bool, dict]:
    """
    水下视频旋转检测 - 判断整个视频最像人体的部分，并且看处于分界线哪里
    
    返回: (是否翻转, 详细信息字典)
    """
    frame_details = []
    best_human_score = -1.0
    best_rotated = False
    best_frame_idx = -1
    best_reason = "未检测到人体"
    no_waterline_count = 0
    
    for idx, frame in enumerate(frames):
        # 先找水位线（使用水下反射轴检测）
        waterline = find_underwater_reflection_axis(frame)
        
        frame_info = {
            "frame_idx": idx,
            "waterline": waterline,
            "verdict": "unknown",
            "reason": ""
        }
        
        if waterline is None:
            no_waterline_count += 1
            frame_info["verdict"] = "unknown"
            frame_info["reason"] = "未能检测到水位线"
            frame_details.append(frame_info)
            continue
        
        # 基于人体检测判断
        debug_info = {}
        result = detect_underwater_video_rotation_by_waterline(
            frame, waterline, debug_info, 
            debug_dir=debug_dir, 
            frame_idx=idx
        )
        
        frame_info["verdict"] = "flipped" if result["rotated"] else "normal"
        frame_info["reason"] = result["reason"]
        frame_info["human_detection"] = result.get("human_detection", {})
        frame_info["scores"] = result.get("human_detection", {})
        frame_details.append(frame_info)

        human_res = result.get("human_detection", {})
        if human_res.get("has_human"):
            conf = human_res.get("confidence", 0.0)
            if conf > best_human_score:
                best_human_score = conf
                best_rotated = result["rotated"]
                best_frame_idx = idx
                best_reason = result["reason"]
    
    rotated = best_rotated
    
    details = {
        "total_frames": len(frames),
        "no_waterline": no_waterline_count,
        "best_frame_idx": best_frame_idx,
        "best_human_score": best_human_score,
        "final_verdict": "flipped" if rotated else "normal",
        "frame_details": frame_details,
        "explanation": ""
    }
    
    if best_frame_idx != -1:
        details["explanation"] = (
            f"基于最高置信度({best_human_score:.2f})的人体检测结果（来自帧 {best_frame_idx}）。\n"
            f"详细原因：{best_reason}\n"
            f"结论：{'需要翻转180°' if rotated else '正常，无需翻转'}。"
        )
    else:
        details["explanation"] = "无法在任何帧中检测到足够清晰的人体或分界线，默认判定为正常。"

    print(f"   [水下旋转检测] 最高置信度:{best_human_score:.2f} (来自帧 {best_frame_idx}) => {'翻转180°' if rotated else '正常'}")
    print(f"   详细: {details['explanation']}")
    
    return rotated, details

def detect_rotations(top_video: Path, bottom_video: Path, debug_dir: Path = None) -> tuple[bool, bool, dict]:
    """
    检测两个视频的旋转状态，返回详细信息
    
    参数：
    - top_video: 水上视频（已通过命令行参数指定）
    - bottom_video: 水下视频（已通过命令行参数指定）
    
    逻辑：
    1. 对水上视频使用 is_top_camera_rotated 检测
    2. 对水下视频使用 is_underwater_camera_rotated 检测
    """
    ma, mb = load_video_meta(top_video), load_video_meta(bottom_video)
    fa, fb = [], []
    
    # 水上视频提取 6 帧用于旋转检测
    reader_top = iio.get_reader(str(top_video))
    for t in np.linspace(1.0, max(1.0, ma.duration - 1.0), 6):
        if (fr := sample_frame_at(reader_top, t, ma.fps)) is not None:
            fa.append(fr)
    reader_top.close()

    # 水下视频提取 10 帧用于旋转检测（分成10份，各取一帧）
    reader_bot = iio.get_reader(str(bottom_video))
    for t in np.linspace(1.0, max(1.0, mb.duration - 1.0), 10):
        if (fr := sample_frame_at(reader_bot, t, mb.fps)) is not None:
            fb.append(fr)
    reader_bot.close()

    if len(fa) < 3: fa = fb[:6] if len(fb) >= 6 else fb[:]
    if len(fb) < 3: fb = fa[:]
    
    print(f"\n  [视频分类] top_video={top_video.name}(水上), bottom_video={bottom_video.name}(水下)")
    
    # 创建旋转检测调试目录
    rotation_debug_dir = None
    if debug_dir is not None:
        rotation_debug_dir = debug_dir / "rotation_check"
        rotation_debug_dir.mkdir(parents=True, exist_ok=True)
    
    # 对水上视频使用 is_top_camera_rotated，对水下视频使用 is_underwater_camera_rotated
    rot_top, details_top = is_top_camera_rotated(fa, "top_video", rotation_debug_dir)
    rot_bottom, details_bottom = is_underwater_camera_rotated(fb, "bottom_video", rotation_debug_dir)
    
    all_details = {
        "top_video": details_top,
        "bottom_video": details_bottom,
        "top_source": "top_video",
        "bottom_source": "bottom_video",
    }
    
    return rot_top, rot_bottom, all_details

def sky_water_separation_score(frame: np.ndarray) -> float:
    small = cv2.resize(frame, (320, 180))
    energy = np.abs(cv2.Sobel(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 0, 1, ksize=3)).mean(axis=1)
    b = small[:,:,2].astype(np.float32) / 255.0
    rg = ((small[:,:,0] + small[:,:,1]) / 2).astype(np.float32) / 255.0
    return float(energy[int(180*0.4):int(180*0.65)].max()) * 10.0 + (float(np.mean(b[90:] - rg[90:])) - float(np.mean(b[:90] - rg[:90]))) * 3.0

# ============================================================
# Step 2: 时间对齐
# ============================================================
def _extract_wav(vid_path: Path, tmpdir: Path, name: str) -> Path | None:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out = tmpdir / f"{name}.wav"
    subprocess.run([ffmpeg_exe, "-y", "-i", str(vid_path), "-vn", "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out if out.exists() else None

def compute_audio_time_offset(video_top: Path, video_bot: Path) -> tuple[float, dict]:
    from scipy.signal import spectrogram
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_t = _extract_wav(video_top, Path(tmpdir), "top")
        wav_b = _extract_wav(video_bot, Path(tmpdir), "bot")
        if not wav_t or not wav_b: return 0.0, {"quality": "no_audio", "issues": ["至少有一个视频无音频"]}

        sr_t, raw_t = wavfile.read(wav_t)
        sr_b, raw_b = wavfile.read(wav_b)

        data_t = raw_t.astype(np.float32) / (np.max(np.abs(raw_t)) + 1e-9)
        data_b = raw_b.astype(np.float32) / (np.max(np.abs(raw_b)) + 1e-9)
        strategies = {}
        issues = []

        nperseg = 1024
        noverlap = int(nperseg * 0.85)
        f_t, t_t, S_t = spectrogram(data_t, fs=sr_t, nperseg=nperseg, noverlap=noverlap)
        f_b, t_b, S_b = spectrogram(data_b, fs=sr_b, nperseg=nperseg, noverlap=noverlap)
        S_t_log, S_b_log = gaussian_filter(np.log10(S_t + 1e-8), 1.2), gaussian_filter(np.log10(S_b + 1e-8), 1.2)

        lag_samples_per_bin = []
        for fb_idx in range(S_t_log.shape[0]):
            vec_t, vec_b = S_t_log[fb_idx, :], S_b_log[fb_idx, :]
            if np.std(vec_t) < 1e-6 or np.std(vec_b) < 1e-6: continue
            corr_1d = fftconvolve(vec_b, vec_t[::-1], mode='full')
            lag_samples_per_bin.append(int(np.argmax(corr_1d) - (len(vec_t) - 1)))

        if lag_samples_per_bin:
            lag_sec_stft = float(np.median(lag_samples_per_bin)) * float(t_t[1] - t_t[0]) if len(t_t) > 1 else 0.0
            strategies['stft_spectrogram'] = {'offset_sec': lag_sec_stft, 'lag_samples_per_bin': lag_samples_per_bin}

        frame_len, step = int(0.25 * sr_t), int(0.1 * sr_t)
        offsets_per_frame = []
        for start in range(0, len(data_t) - frame_len * 3, step):
            seg_t = data_t[start:start + frame_len]
            search_start = max(0, start - int(5 * sr_t))
            search_end = min(len(data_b), start + frame_len + int(5 * sr_t))
            seg_b = data_b[search_start:search_end]
            if len(seg_b) < len(seg_t): continue
            c = fftconvolve(seg_b, seg_t[::-1], mode='valid')
            if len(c) > 0:
                offsets_per_frame.append((np.argmax(c) - (len(seg_t) - 1)) / float(sr_t) + (search_start - start) / float(sr_t))

        if offsets_per_frame:
            counts, bin_edges = np.histogram(offsets_per_frame, bins=100)
            peak_bin = np.argmax(counts)
            lag_sec_sliding = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
            strategies['sliding_window'] = {'offset_sec': float(lag_sec_sliding), 'histogram_peak': float(lag_sec_sliding), 'all_offsets': offsets_per_frame}

        sw_info, stft_info = strategies.get('sliding_window', {}), strategies.get('stft_spectrogram', {})
        best_offset = sw_info.get('histogram_peak', stft_info.get('offset_sec', 0.0))

        # 策略间一致性检查
        if sw_info and stft_info:
            diff = abs(sw_info.get('histogram_peak', 0) - stft_info.get('offset_sec', 0))
            if diff > 0.5:
                issues.append(f"两种策略时间差过大: STFT={stft_info['offset_sec']:.3f}s, 滑动窗口={sw_info['histogram_peak']:.3f}s, 差={diff:.3f}s")
            elif diff > 0.2:
                issues.append(f"两种策略略有分歧: STFT={stft_info['offset_sec']:.3f}s, 滑动窗口={sw_info['histogram_peak']:.3f}s")

        # 绝对值合理性检查
        if abs(best_offset) > 10.0:
            issues.append(f"时间偏移绝对值过大: {best_offset:.3f}s，可能非同一视频")
        elif abs(best_offset) > 3.0:
            issues.append(f"时间偏移较大: {best_offset:.3f}s，请确认是否正确配对")

        # 滑动窗口直方图峰值锐度检查
        if offsets_per_frame:
            offsets_arr = np.array(offsets_per_frame)
            if np.std(offsets_arr) > 2.0:
                issues.append(f"滑动窗口偏移离散度大(σ={np.std(offsets_arr):.3f}s)，对齐质量可能不佳")

        quality = "ok"
        if len(issues) == 1:
            quality = "warn"
        elif len(issues) >= 2:
            quality = "bad"

        strategies['quality'] = quality
        strategies['issues'] = issues

        return best_offset, strategies

# ============================================================
# Step 3/4: 高级物理对齐与重构拼接
# ============================================================
def find_best_waterline_row(gray: np.ndarray) -> int | None:
    h = gray.shape[0]
    row_start, row_end = int(h * 0.20), int(h * 0.65)
    energy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)).mean(axis=1)
    if row_end <= row_start: return None
    best_row, best_energy = None, -1.0
    for r in range(row_start + 2, row_end - 2):
        if energy[r] == energy[r-2:r+3].max() and energy[r] > best_energy:
            best_energy, best_row = energy[r], r
    return int(np.argmax(energy[row_start:row_end])) + row_start if best_row is None else best_row

def find_underwater_reflection_axis(frame: np.ndarray) -> int | None:
    h = frame.shape[0]
    gray_s = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (320, 180))
    hs = gray_s.shape[0]
    r_start, r_end, max_band = int(hs * 0.18), int(hs * 0.82), min(70, max(12, int(hs * 0.64 // 3)))
    grad = np.abs(cv2.Sobel(gray_s, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
    grad /= max(float(grad.max()), 1e-6)
    best_row, best_score = None, -1e9
    for r in range(r_start + max_band, r_end - max_band):
        u, l = gray_s[r - max_band:r, :], gray_s[r:r + max_band, :]
        den = np.linalg.norm(u - u.mean()) * np.linalg.norm(l - l.mean())
        int_score = float(np.sum((u[::-1] - u.mean()) * (l - l.mean())) / den) if den >= 1e-6 else -1.0
        score = int_score * 0.75 + float(grad[r]) * 0.25 - abs(r - int(hs * 0.5)) / hs * 0.08
        if score > best_score: best_score, best_row = score, r
    return int(round(best_row * h / hs)) if best_row else None

def get_motion_x_centroid(frame: np.ndarray, bg: np.ndarray) -> float | None:
    if frame.shape != bg.shape: bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
    diff = cv2.cvtColor(cv2.absdiff(frame, bg), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    M = cv2.moments(mask)
    return float(M["m10"] / M["m00"]) if M["m00"] > 1000 else None

# ==============================================================================
# 核心渲染与重构逻辑：将上下水面物理分离，羽化在延长线上进行
# ==============================================================================
def blend_waterline_fusion(
    top_frame: np.ndarray, bottom_frame: np.ndarray, top_wl: int, bot_wl: int,
    horizontal_offset: int, feather_px: int, top_expand: int, bot_expand: int
) -> np.ndarray:
    """
    通过拉开两个视频补偿水面折射损失的人体。
    羽化（若大于0）严格发生在新拉出的两个辅助拓展重叠带内，绝不干扰已有的画面。
    """
    h_top, w = top_frame.shape[:2]
    h_bot = bottom_frame.shape[0]

    # delta_y：拉开画布重构后，底部视频整体向下平移的 Y 轴绝对距离
    delta_y = top_wl - bot_wl + top_expand + bot_expand + feather_px
    out_h = max(h_top, h_bot + delta_y)
    if out_h <= 0: return top_frame.copy()

    # 利用 cv2.warpAffine 矩阵特性：一次性实现下视频横向平移与纵向物理下沉，超出水面以上的部分由 BORDER_REPLICATE 安全生成
    M = np.float32([[1, 0, horizontal_offset], [0, 1, delta_y]])
    bottom_shifted = cv2.warpAffine(bottom_frame, M, (w, out_h), borderMode=cv2.BORDER_REPLICATE)

    # 上半部分放入画布 (利用 BORDER_REPLICATE 防止扩充过度露黑底)
    if out_h > h_top:
        top_shifted = cv2.copyMakeBorder(top_frame, 0, out_h - h_top, 0, 0, cv2.BORDER_REPLICATE)
    else:
        top_shifted = top_frame[:out_h].copy()

    # 制定无损羽化区间：只作用于 `top_wl + top_expand` 到 `+ feather_px` 的这层额外区域
    y_feather_start = top_wl + top_expand
    y_feather_end = y_feather_start + feather_px
    
    mask = np.zeros((out_h, 1, 1), dtype=np.float32)
    if y_feather_start > 0:
        mask[0:min(out_h, y_feather_start)] = 1.0
    
    if y_feather_end > y_feather_start:
        y_s = max(0, min(out_h, y_feather_start))
        y_e = max(0, min(out_h, y_feather_end))
        if y_e > y_s:
            mask[y_s:y_e] = np.linspace(1.0, 0.0, y_e - y_s).reshape(-1, 1, 1)

    acc = top_shifted.astype(np.float32) * mask + bottom_shifted.astype(np.float32) * (1.0 - mask)
    return acc.astype(np.uint8)

def score_fused_body_similarity(fused: np.ndarray, seam_y: int) -> float:
    h, fw = fused.shape[:2]
    top_region, bot_region = fused[max(0, seam_y - 25):seam_y], fused[seam_y:min(h, seam_y + 25)]
    if top_region.size == 0 or bot_region.size == 0: return 0.0
    t_prof = np.abs(cv2.Sobel(cv2.cvtColor(top_region, cv2.COLOR_RGB2GRAY).astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)).mean(axis=0)
    b_prof = np.abs(cv2.Sobel(cv2.cvtColor(bot_region, cv2.COLOR_RGB2GRAY).astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)).mean(axis=0)
    t_prof, b_prof = (t_prof - t_prof.mean()) / (t_prof.std() + 1e-6), (b_prof - b_prof.mean()) / (b_prof.std() + 1e-6)
    if len(t_prof) < 5 or len(b_prof) < 5: return 0.0
    corr = fftconvolve(t_prof, b_prof[::-1], mode='full')
    mid = len(b_prof) - 1
    search_l, search_r = max(0, mid - 50), min(len(corr) - 1, mid + 50)
    if search_r <= search_l: return 0.0
    return float(np.max(corr[search_l:search_r + 1])) * 0.85 + (1.0 / (1.0 + float(np.std(fused[seam_y] if seam_y < h else fused[-1])) / 10.0)) * 15.0

def find_horizontal_offset_by_body_fusion(
    top_frame: np.ndarray, bot_frame: np.ndarray, top_wl: int, bot_wl: int,
    top_expand: int, bot_expand: int,
    debug_dir: Path | None = None, frame_idx: int = 0
) -> float:
    """
    遍历横向偏移 -80~+80 px（步长 4），找使拼接后躯干连续性最强的偏移。
    若 debug_dir 不为空，生成中间结果截图。
    """
    scores = {}
    fused_samples = {}
    for offset in range(-80, 81, 4):
        fused = blend_waterline_fusion(top_frame, bot_frame, top_wl, bot_wl, offset, 0, top_expand, bot_expand)
        scores[offset] = score_fused_body_similarity(fused, top_wl + top_expand)
        fused_samples[offset] = fused

    best_offset = float(max(scores, key=lambda k: scores[k])) if scores else 0.0

    if debug_dir is not None:
        _save_body_fusion_debug(
            debug_dir, frame_idx, top_frame, bot_frame,
            top_wl, bot_wl, top_expand, bot_expand,
            fused_samples, scores, best_offset
        )

    return best_offset


def _save_body_fusion_debug(
    debug_dir: Path, frame_idx: int,
    top_frame: np.ndarray, bot_frame: np.ndarray,
    top_wl: int, bot_wl: int, top_expand: int, bot_expand: int,
    fused_samples: dict[int, np.ndarray], scores: dict[int, float], best_offset: float
) -> None:
    try:
        h, w = top_frame.shape[:2]

        top_labeled = top_frame.copy()
        bot_labeled = bot_frame.copy()
        cv2.line(top_labeled, (0, int(top_wl)), (w, int(top_wl)), (0, 255, 255), 2)
        cv2.line(top_labeled, (0, int(top_wl + top_expand)), (w, int(top_wl + top_expand)), (255, 255, 0), 1)
        cv2.putText(top_labeled, f"top_wl={int(top_wl)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.line(bot_labeled, (0, int(bot_wl)), (w, int(bot_wl)), (0, 255, 255), 2)
        cv2.line(bot_labeled, (0, int(bot_wl + bot_expand)), (w, int(bot_wl + bot_expand)), (255, 255, 0), 1)
        cv2.putText(bot_labeled, f"bot_wl={int(bot_wl)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        side_by_side = np.hstack([top_labeled, bot_labeled])
        cv2.imwrite(str(debug_dir / f"frame_{frame_idx:03d}_00_src.jpg"), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

        best_offset_int = int(round(best_offset))
        sorted_offsets = sorted(scores.keys())
        candidates = [best_offset_int]
        for o in sorted_offsets:
            if o not in candidates and abs(o - best_offset_int) >= 16:
                candidates.append(o)
                if len(candidates) >= 5:
                    break
        candidates = sorted(set(candidates))

        fusion_rows = []
        for offset in candidates:
            fused = fused_samples[offset]
            label = f"off={offset:+d}  score={scores[offset]:.2f}"
            labeled = fused.copy()
            seam_y = int(top_wl + top_expand)
            cv2.line(labeled, (0, seam_y), (int(labeled.shape[1]), seam_y), (0, 255, 0), 2)
            cv2.putText(labeled, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if offset == best_offset_int else (200, 200, 200), 2)
            fusion_rows.append(labeled)

        if fusion_rows:
            fusion_grid = np.vstack(fusion_rows)
            cv2.imwrite(str(debug_dir / f"frame_{frame_idx:03d}_01_fusions.jpg"), cv2.cvtColor(fusion_grid, cv2.COLOR_RGB2BGR))

        fig, ax = plt.subplots(figsize=(10, 4))
        xs, ys = zip(*sorted(scores.items()))
        ax.plot(xs, ys, 'b-o', linewidth=1.5, markersize=4)
        ax.axvline(float(best_offset_int), color='r', linestyle='--', label=f'best={best_offset_int}')
        ax.scatter([float(best_offset_int)], [float(scores[best_offset_int])], color='r', zorder=5, s=80)
        ax.set_title(f'body_fusion score vs horizontal_offset  (frame={int(frame_idx)})')
        ax.set_xlabel('horizontal_offset (px)')
        ax.set_ylabel('similarity score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(str(debug_dir / f"frame_{frame_idx:03d}_02_scores.png"), dpi=100, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        import traceback; traceback.print_exc()


def compute_spatial_alignment_10x4(
    top_vid: Path, bot_vid: Path,
    time_offset: float,
    top_rot: bool, bot_rot: bool,
    top_zoom_factor: float = 1.0,
    top_expand: int = 20, bot_expand: int = 20,
    debug_dir: Path | None = None
) -> tuple[int, int, int, dict]:
    meta_t, meta_b = load_video_meta(top_vid), load_video_meta(bot_vid)
    ov_start, ov_end = 0.0, min(meta_t.duration, meta_b.duration - time_offset)
    if ov_end <= ov_start: raise ValueError("没有有效重叠时间")

    seg_dur = (ov_end - ov_start) / 10.0
    sample_times = [ov_start + (i + 0.5) * seg_dur + offset * (1.0 / meta_t.fps) for i in range(10) for offset in range(4)]

    t_reader, b_reader = iio.get_reader(str(top_vid)), iio.get_reader(str(bot_vid))
    wl_tops, wl_bots, x_offsets_body_fusion = [], [], []
    total_sampled, valid_offset_count = 0, 0
    spatial_issues = []

    for frame_idx, t in enumerate(sample_times):
        total_sampled += 1
        tf, bf = sample_frame_at(t_reader, t, meta_t.fps), sample_frame_at(b_reader, max(0.0, t + time_offset), meta_b.fps)
        if tf is None or bf is None: continue
        tf, bf = rotate_frame(tf, top_rot), rotate_frame(bf, bot_rot)

        if top_zoom_factor != 1.0:
            tf = apply_top_zoom(tf, top_zoom_factor)

        if tf.shape != bf.shape:
            bf = cv2.resize(bf, (tf.shape[1], tf.shape[0]))

        wt, wb = find_best_waterline_row(cv2.cvtColor(tf, cv2.COLOR_RGB2GRAY)), find_underwater_reflection_axis(bf)
        if wt is not None: wl_tops.append(wt)
        if wb is not None: wl_bots.append(wb)

        if wt is not None and wb is not None:
            try:
                b_off = find_horizontal_offset_by_body_fusion(tf, bf, wt, wb, top_expand, bot_expand, debug_dir, frame_idx)
                if b_off != 0.0:
                    x_offsets_body_fusion.append(b_off)
                    valid_offset_count += 1
            except Exception:
                import traceback; traceback.print_exc()

    t_reader.close(); b_reader.close()
    if not wl_tops or not wl_bots: raise ValueError("未能识别到相机的水位线")

    used_fallback = False
    if not x_offsets_body_fusion:
        spatial_issues.append("身体融合算法未能找到有效偏移，使用运动质心Fallback")
        x_offsets = _fallback_by_motion_centroid(top_vid, bot_vid, time_offset, top_rot, bot_rot, meta_t, meta_b, top_zoom_factor)
        used_fallback = True
    else:
        h_offset_raw = float(np.median(x_offsets_body_fusion))
        x_offsets = [h_offset_raw]

    # 空间对齐质量评估
    if not used_fallback and total_sampled > 0:
        ratio = valid_offset_count / total_sampled
        if ratio < 0.3:
            spatial_issues.append(f"有效偏移帧比例过低: {valid_offset_count}/{total_sampled} ({ratio:.0%})，对齐可能不可靠")
        offsets_std = float(np.std(x_offsets_body_fusion)) if len(x_offsets_body_fusion) > 1 else 0.0
        if offsets_std > 20.0:
            spatial_issues.append(f"横向偏移离散度过大(σ={offsets_std:.1f}px)，各帧结果不一致")
        elif offsets_std > 10.0:
            spatial_issues.append(f"横向偏移略有离散(σ={offsets_std:.1f}px)")

    spatial_quality = "ok"
    if used_fallback:
        spatial_quality = "warn"
    if len(spatial_issues) >= 2:
        spatial_quality = "bad"

    spatial_info = {
        "quality": spatial_quality,
        "issues": spatial_issues,
        "total_sampled": total_sampled,
        "valid_offset_count": valid_offset_count,
        "used_fallback": used_fallback,
        "offset_std": float(np.std(x_offsets_body_fusion)) if len(x_offsets_body_fusion) > 1 else 0.0,
    }

    return int(round(float(np.median(x_offsets)))), int(np.median(wl_tops)), int(np.median(wl_bots)), spatial_info

def _fallback_by_motion_centroid(
    top_vid: Path, bot_vid: Path, time_offset: float,
    top_rot: bool, bot_rot: bool, meta_t, meta_b, top_zoom_factor: float
) -> list:
    try:
        tr, br = iio.get_reader(str(top_vid)), iio.get_reader(str(bot_vid))
        bg_t = np.median([apply_top_zoom(rotate_frame(tr.get_data(i), top_rot), top_zoom_factor) for i in range(10)], axis=0).astype(np.uint8)
        bg_b = np.median([rotate_frame(br.get_data(i), bot_rot) for i in range(10)], axis=0).astype(np.uint8)
        tr.close(); br.close()
        
        fallback_x = []
        tr, br = iio.get_reader(str(top_vid)), iio.get_reader(str(bot_vid))
        for t in np.linspace(1.0, max(1.0, min(meta_t.duration, meta_b.duration - time_offset) - 1.0), 15):
            tf, bf = sample_frame_at(tr, t, meta_t.fps), sample_frame_at(br, max(0.0, t + time_offset), meta_b.fps)
            if tf is None or bf is None: continue
            tf, bf = rotate_frame(tf, top_rot), rotate_frame(bf, bot_rot)
            
            if top_zoom_factor != 1.0: tf = apply_top_zoom(tf, top_zoom_factor)
            if tf.shape != bf.shape: bf = cv2.resize(bf, (tf.shape[1], tf.shape[0]))
                
            if (cx_t := get_motion_x_centroid(tf, bg_t)) is not None and (cx_b := get_motion_x_centroid(bf, bg_b)) is not None:
                fallback_x.append(cx_t - cx_b)
        tr.close(); br.close()
        return fallback_x if fallback_x else [0]
    except Exception: return [0]


def write_fused_video_v2(top_video: Path, bottom_video: Path, cfg: FusionConfig, output_path: Path, output_fps: float = 30.0, max_width: int = 1280) -> None:
    top_meta, bottom_meta = load_video_meta(top_video), load_video_meta(bottom_video)
    top_reader, bottom_reader = iio.get_reader(str(top_video)), iio.get_reader(str(bottom_video))
    overlap_end = min(top_meta.duration, bottom_meta.duration - cfg.time_offset_sec)

    writer, current, frame_count = None, 0.0, 0
    while current < overlap_end:
        top_frame = sample_frame_at(top_reader, current, top_meta.fps)
        bottom_frame = sample_frame_at(bottom_reader, max(0.0, current + cfg.time_offset_sec), bottom_meta.fps)
        if top_frame is None or bottom_frame is None:
            current += 1.0 / output_fps; continue

        top_frame, bottom_frame = rotate_frame(top_frame, cfg.top_rotate_180), rotate_frame(bottom_frame, cfg.bottom_rotate_180)

        if cfg.top_zoom_factor != 1.0:
            top_frame = apply_top_zoom(top_frame, cfg.top_zoom_factor)

        if top_frame.shape != bottom_frame.shape:
            bottom_frame = cv2.resize(bottom_frame, (top_frame.shape[1], top_frame.shape[0]))

        # 全局宽度缩放
        scale = max_width / top_frame.shape[1] if top_frame.shape[1] > max_width else 1.0
        if scale != 1.0:
            top_frame = cv2.resize(top_frame, (max_width, int(top_frame.shape[0] * scale)))
            bottom_frame = cv2.resize(bottom_frame, (max_width, int(bottom_frame.shape[0] * scale)))

        fused = blend_waterline_fusion(
            top_frame, bottom_frame,
            int(round(cfg.raw_top_waterline * scale)),
            int(round(cfg.raw_bottom_waterline * scale)),
            int(round(cfg.horizontal_offset * scale)),
            int(round(cfg.feather_px * scale)),
            int(round(cfg.top_waterline_offset * scale)),
            int(round(cfg.bottom_waterline_offset * scale))
        )

        if fused.shape[0] % 2 != 0: fused = fused[:-1, :]

        if writer is None:
            output_h, output_w = fused.shape[:2]
            writer = iio.get_writer(str(output_path), fps=output_fps, codec="libx264", quality=8, macro_block_size=None, output_params=["-pix_fmt", "yuv420p"])

        if fused.shape[0] != output_h or fused.shape[1] != output_w:
            fused = cv2.resize(fused, (output_w, output_h), interpolation=cv2.INTER_AREA)

        writer.append_data(fused)
        current += 1.0 / output_fps
        frame_count += 1

    if writer is not None: writer.close()
    top_reader.close(); bottom_reader.close()
    print(f"\n✅ 融合视频已生成: {output_path} (共 {frame_count} 帧)")

# ============================================================
# 主控制器
# ============================================================
def run_full_pipeline(top_video: Path, bottom_video: Path, output_dir: Path, output_fps: float = 30.0, max_width: int = 1280, force_top_rotated: str = None, force_bot_rotated: str = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / "debug_body_fusion"
    debug_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}\n  处理: {top_video.name}  x  {bottom_video.name}\n{'='*60}")

    print("Step 1/4: 画面解析与身份辨别...")
    rot_top, rot_bottom, rotation_details = detect_rotations(top_video, bottom_video, debug_dir)
    
    if force_top_rotated is not None: rot_top = force_top_rotated == "true"
    if force_bot_rotated is not None: rot_bottom = force_bot_rotated == "true"
    
    # 打印详细的旋转判断依据
    print(f"\n{'='*60}")
    print(f"  旋转检测详细分析")
    print(f"{'='*60}")
    print(f"\n  【水上视频: {top_video.name}】")
    print(f"  {rotation_details['top_video']['explanation']}")
    print(f"\n  【水下视频: {bottom_video.name}】")
    print(f"  {rotation_details['bottom_video']['explanation']}")
    
    print(f"\n  最终决定:")
    print(f"    水上视频 ({top_video.name}): {'翻转180°' if rot_top else '正常方向'}")
    print(f"    水下视频 ({bottom_video.name}): {'翻转180°' if rot_bottom else '正常方向'}")
    print(f"{'='*60}\n")

    print("Step 2/4: 音频波形对齐 (语谱图互相关 + 多策略) ...")
    time_offset, strategies = compute_audio_time_offset(top_video, bottom_video)
    print(f"   => 水下相对水上时间偏移: {time_offset:.4f} 秒")

    print("Step 3/4: 10段x4帧空间定标 (身体融合相似度算法 + 运动质心 Fallback) ...")
    h_offset, top_wl, bot_wl, spatial_info = compute_spatial_alignment_10x4(
        top_video, bottom_video, time_offset, rot_top, rot_bottom,
        TOP_ZOOM_FACTOR, TOP_WATERLINE_OFFSET, BOTTOM_WATERLINE_OFFSET,
        debug_dir=debug_dir
    )

    print(f"   => 放大倍率: {TOP_ZOOM_FACTOR}x")
    print(f"   => 横向偏移对齐: {int(h_offset)} px")
    print(f"   => 物理水面上切线: {int(top_wl)} px, 扩展后保留至 {int(top_wl) + int(TOP_WATERLINE_OFFSET)} px")
    print(f"   => 物理水面下切线: {int(bot_wl)} px, 扩展后保留至 {int(bot_wl) - int(BOTTOM_WATERLINE_OFFSET)} px")
    print(f"   => 调试截图已保存: {debug_dir}")

    print("Step 4/4: 全局静态切片物理重构并渲染...")
    cfg = FusionConfig(
        seam=0, feather_px=FEATHER_PX, horizontal_offset=h_offset, time_offset_sec=time_offset,
        top_rotate_180=rot_top, bottom_rotate_180=rot_bottom, 
        raw_top_waterline=top_wl, raw_bottom_waterline=bot_wl, 
        top_waterline_offset=TOP_WATERLINE_OFFSET, bottom_waterline_offset=BOTTOM_WATERLINE_OFFSET,
        top_zoom_factor=TOP_ZOOM_FACTOR
    )

    write_fused_video_v2(top_video, bottom_video, cfg, output_dir / "fused_swim_v2.mp4", output_fps, max_width)

    report = {
        "time_offset_sec": time_offset,
        "horizontal_offset_px": h_offset,
        "top_wl": top_wl,
        "bot_wl": bot_wl,
        "rotation_detection": {
            "top_video": {
                "file": str(top_video),
                "rotated_180": rot_top,
                "source": "top_video",
                "details": rotation_details['top_video']
            },
            "bottom_video": {
                "file": str(bottom_video),
                "rotated_180": rot_bottom,
                "source": "bottom_video",
                "details": rotation_details['bottom_video']
            }
        }
    }
    return report

def run_batch_pipeline(top_dir: Path, bottom_dir: Path, output_dir: Path, output_fps: float = 30.0, max_width: int = 1280) -> list[dict]:
    import re
    top_dir, bottom_dir, output_dir = Path(top_dir), Path(bottom_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_files, bot_files = sorted(top_dir.glob("*.mp4")), sorted(bottom_dir.glob("*.mp4"))

    # DJI 文件名格式: DJI_<录制时间戳14位>_<剪辑序号4位>_D[_min].mp4
    # 例如: DJI_20260508195343_0001_D_min.mp4
    clip_pattern = re.compile(r'^DJI_(\d{14})_\d+_D(_min)?\.mp4$', re.IGNORECASE)

    def parse_dji_timestamp(ts_str: str) -> int:
        """将 DJI 时间戳字符串转换为 Unix 时间戳（秒）"""
        try:
            from datetime import datetime
            dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
            return int(dt.timestamp())
        except:
            return 0

    # 解析所有视频的时间戳
    top_videos = []  # (file, timestamp_str, timestamp_int)
    for f in top_files:
        m = clip_pattern.match(f.name)
        if m:
            ts_str = m.group(1)
            top_videos.append((f, ts_str, parse_dji_timestamp(ts_str)))

    bot_videos = []  # (file, timestamp_str, timestamp_int)
    for f in bot_files:
        m = clip_pattern.match(f.name)
        if m:
            ts_str = m.group(1)
            bot_videos.append((f, ts_str, parse_dji_timestamp(ts_str)))

    results, paired, skipped, missing_pairs, failed_pairs = [], 0, 0, [], []
    
    # 录制时间差值阈值（秒），超过此值认为是不同时间录制，不应融合
    TIMESTAMP_DIFF_THRESHOLD_SEC = 60

    # 已使用的视频集合
    used_tops = set()
    used_bots = set()

    # 纯按时间戳配对：遍历所有 top 视频，找时间最接近的未匹配 bottom 视频
    for top_video, top_ts_str, top_ts_int in top_videos:
        if top_video in used_tops:
            continue

        # 找时间最接近且未使用的 bottom 视频
        best_bot = None
        best_diff = float('inf')
        for bot_video, bot_ts_str, bot_ts_int in bot_videos:
            if bot_video in used_bots:
                continue
            diff = abs(top_ts_int - bot_ts_int)
            if diff < best_diff:
                best_diff = diff
                best_bot = (bot_video, bot_ts_str, bot_ts_int)

        if best_bot is None:
            missing_pairs.append((top_video, "没有可配对的 bottom 视频（全部已使用或不存在）"))
            continue

        bottom_video, bot_ts_str, bot_ts_int = best_bot
        used_tops.add(top_video)
        used_bots.add(bottom_video)

        # 检查时间差是否在阈值内
        if best_diff > TIMESTAMP_DIFF_THRESHOLD_SEC:
            failed_pairs.append((top_video, bottom_video,
                f"最接近的 bottom 视频录制时间差={best_diff}秒(>{TIMESTAMP_DIFF_THRESHOLD_SEC}秒)，top={top_ts_str} vs bot={bot_ts_str}，非同时间段录制，跳过融合"))
            # 标记为失败后，这两个视频都不能再用了
            # 注意：已加入 used_tops/bots，但仍记录失败
            continue

        pair_out = output_dir / f"{top_video.stem}__{bottom_video.stem}"
        report_file = pair_out / "alignment_report_v2.json"
        if report_file.exists():
            print(f"  ⏭️  跳过（已存在）: {report_file.name}  →  {pair_out.name}")
            skipped += 1
            continue
        pair_out.mkdir(parents=True, exist_ok=True)
        try:
            report = run_full_pipeline(top_video, bottom_video, pair_out, output_fps, max_width)
            report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            paired += 1
            results.append(report)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"   [跳过] {type(e).__name__}: {e}")
            failed_pairs.append((top_video, bottom_video, str(e)))
            skipped += 1

    # 检查有但没配对上的视频
    for top_video, top_ts_str, top_ts_int in top_videos:
        if top_video not in used_tops:
            # 检查是不是因为时间差太大被跳过了
            found = False
            for t, b, err in failed_pairs:
                if t == top_video:
                    found = True
                    break
            if not found:
                missing_pairs.append((top_video, "未配对成功"))

    for bot_video, bot_ts_str, bot_ts_int in bot_videos:
        if bot_video not in used_bots:
            # 检查是不是因为时间差太大被跳过了
            found = False
            for t, b, err in failed_pairs:
                if b == bot_video:
                    found = True
                    break
            if not found:
                missing_pairs.append((bot_video, "未配对成功（top 视频不足）"))

    print(f"\n{'='*60}")
    print(f"  批处理完成汇总")
    print(f"{'='*60}")
    print(f"  ✅ 已融合: {paired}")
    print(f"  ⏭️  已跳过: {skipped} (含 {len(failed_pairs)} 个失败)")
    print(f"  ❌ 缺失配对: {len(missing_pairs)}")

    if missing_pairs:
        print(f"\n  --- 缺失配对 ---")
        for video, reason in missing_pairs:
            print(f"    ❌ {video.name} → {reason}")

    if failed_pairs:
        print(f"\n  --- 融合失败 ---")
        for top_v, bot_v, err in failed_pairs:
            print(f"    ❌ {top_v.name} x {bot_v.name}")
            print(f"       原因: {err}")

    if not missing_pairs and not failed_pairs:
        print(f"\n  ✅ 所有视频均已正确配对并融合，无问题")

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="水上/水下游泳视频融合工具")
    parser.add_argument("--video-a", type=Path)
    parser.add_argument("--video-b", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    parser.add_argument("--batch-top", type=Path)
    parser.add_argument("--batch-bottom", type=Path)
    parser.add_argument("--batch-output", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--force-top-rotated", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--force-bot-rotated", type=str, default=None, choices=["true", "false"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.batch_top and args.batch_bottom and args.batch_output:
        run_batch_pipeline(args.batch_top, args.batch_bottom, args.batch_output, output_fps=args.fps, max_width=args.max_width)
    elif args.video_a and args.video_b:
        run_full_pipeline(args.video_a, args.video_b, args.output_dir, output_fps=args.fps, max_width=args.max_width,
                          force_top_rotated=args.force_top_rotated, force_bot_rotated=args.force_bot_rotated)