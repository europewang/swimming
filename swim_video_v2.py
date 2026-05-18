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
TOP_ZOOM_FACTOR = 1.2             # 水上视频全局放大倍率（拉近镜头，主体更大）
TOP_WATERLINE_OFFSET = 20         # 针对水上视频：将水线向下扩展的像素大小（展示更多人物上半身）
BOTTOM_WATERLINE_OFFSET = 20      # 针对水下视频：将水线向上扩展的像素大小（展示更多人物下半身）
FEATHER_PX = 20                    # 融合羽化宽度（在新拓展的重合带内进行过渡，0为硬切，不占有已有内容）
BLUE, GREEN, RED, YELLOW = (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)

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
# Step 1: 颠倒检测
# ============================================================
def is_top_camera_rotated(frames: list[np.ndarray]) -> bool:
    ub, lb = [], []
    for f in frames:
        h = f.shape[0]
        upper, lower = f[:h//2].astype(np.float64), f[h//2:].astype(np.float64)
        ub.append(float(np.mean(upper[:,:,2])) - float(np.mean(upper[:,:,0:2])/2.0))
        lb.append(float(np.mean(lower[:,:,2])) - float(np.mean(lower[:,:,0:2])/2.0))
    return bool(np.mean(ub) > np.mean(lb))

def is_underwater_camera_rotated(frames: list[np.ndarray], debug_suffix: str = "") -> bool:
    try:
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, PoseLandmark
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
    except ImportError: return False
    model_path = Path.home() / ".mediapipe" / "models" / "pose_landmarker_lite.task"
    if not model_path.exists():
        import urllib.request, os
        os.makedirs(str(model_path.parent), exist_ok=True)
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", str(model_path))

    options = PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)), running_mode=PoseLandmarkerOptions.running_mode.IMAGE)
    landmarker = PoseLandmarker.create_from_options(options)

    head_above_foot_count, foot_above_head_count = 0, 0
    for f in frames:
        img = Image(image_format=ImageFormat.SRGB, data=f)
        result = landmarker.detect(img)
        if not result.pose_landmarks: continue
        lm = result.pose_landmarks[0]
        shoulder = lm[PoseLandmark.LEFT_SHOULDER] if lm[PoseLandmark.LEFT_SHOULDER].visibility >= lm[PoseLandmark.RIGHT_SHOULDER].visibility else lm[PoseLandmark.RIGHT_SHOULDER]
        ankle = lm[PoseLandmark.LEFT_ANKLE] if lm[PoseLandmark.LEFT_ANKLE].visibility >= lm[PoseLandmark.RIGHT_ANKLE].visibility else lm[PoseLandmark.RIGHT_ANKLE]
        if shoulder.visibility < 0.3 or ankle.visibility < 0.3: continue
        if shoulder.y < ankle.y: head_above_foot_count += 1
        else: foot_above_head_count += 1

    landmarker.close()
    return foot_above_head_count > head_above_foot_count

def detect_rotations(video_a: Path, video_b: Path) -> tuple[bool, bool, bool, bool]:
    ma, mb = load_video_meta(video_a), load_video_meta(video_b)
    fa, fb = [], []
    for v, meta, lst in [(video_a, ma, fa), (video_b, mb, fb)]:
        reader = iio.get_reader(str(v))
        for t in np.linspace(1.0, max(1.0, meta.duration - 1.0), 6):
            if (fr := sample_frame_at(reader, t, meta.fps)) is not None: lst.append(fr)
        reader.close()
    if len(fa) < 3: fa = fb[:]
    if len(fb) < 3: fb = fa[:]
    return is_top_camera_rotated(fa), is_top_camera_rotated(fb), is_underwater_camera_rotated(fa, "_videoA"), is_underwater_camera_rotated(fb, "_videoB")

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
        if not wav_t or not wav_b: return 0.0, {}
        sr_t, raw_t = wavfile.read(wav_t)
        sr_b, raw_b = wavfile.read(wav_b)

    data_t = raw_t.astype(np.float32) / (np.max(np.abs(raw_t)) + 1e-9)
    data_b = raw_b.astype(np.float32) / (np.max(np.abs(raw_b)) + 1e-9)
    strategies = {}

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
    if sw_info and stft_info and abs(sw_info.get('histogram_peak',0) - stft_info.get('offset_sec',0)) <= 0.5:
        best_offset = stft_info.get('offset_sec', 0.0)

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

def compute_waterline_edge_profile(frame: np.ndarray, waterline_row: int, strip_height: int = 12) -> np.ndarray:
    h, w = frame.shape[:2]
    strip = frame[max(0, waterline_row - strip_height):min(h, waterline_row + strip_height), :]
    edge = np.abs(cv2.Sobel(cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY).astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)).mean(axis=0)
    return gaussian_filter(edge, sigma=1.5)

def find_horizontal_offset_by_edge_correlation(top_frame: np.ndarray, bot_frame: np.ndarray, top_wl: int, bot_wl: int) -> float:
    if top_frame.shape[1] != bot_frame.shape[1]: return 0.0
    prof_t = compute_waterline_edge_profile(top_frame, top_wl)
    prof_b = compute_waterline_edge_profile(bot_frame, bot_wl)
    if len(prof_t) < 10 or len(prof_b) < 10: return 0.0
    min_len = min(len(prof_t), len(prof_b))
    corr = fftconvolve(prof_b[:min_len], prof_t[:min_len][::-1], mode='full')
    mid = min_len - 1
    search_l, search_r = max(0, mid - 80), min(len(corr) - 1, mid + 80)
    if search_r <= search_l: return 0.0
    return float(int(np.argmax(corr[search_l:search_r + 1])) + search_l - mid)

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

def find_horizontal_offset_by_body_fusion(top_frame: np.ndarray, bot_frame: np.ndarray, top_wl: int, bot_wl: int, top_expand: int, bot_expand: int) -> float:
    scores = {}
    for offset in range(-80, 81, 4):
        # 寻找平移位点时采用 0 像素硬切，此时最锋利的边界最有助于寻找物理结构连续性
        fused = blend_waterline_fusion(top_frame, bot_frame, top_wl, bot_wl, offset, 0, top_expand, bot_expand)
        scores[offset] = score_fused_body_similarity(fused, top_wl + top_expand)
    return float(max(scores, key=lambda k: scores[k])) if scores else 0.0

def compute_spatial_alignment_10x4(
    top_vid: Path, bot_vid: Path, 
    time_offset: float, 
    top_rot: bool, bot_rot: bool, 
    top_zoom_factor: float = 1.0,
    top_expand: int = 20, bot_expand: int = 20
) -> tuple[int, int, int]:
    meta_t, meta_b = load_video_meta(top_vid), load_video_meta(bot_vid)
    ov_start, ov_end = 0.0, min(meta_t.duration, meta_b.duration - time_offset)
    if ov_end <= ov_start: raise ValueError("没有有效重叠时间")

    seg_dur = (ov_end - ov_start) / 10.0
    sample_times = [ov_start + (i + 0.5) * seg_dur + offset * (1.0 / meta_t.fps) for i in range(10) for offset in range(4)]

    model_dir = Path.home() / ".mediapipe" / "models"
    model_path = model_dir / "pose_landmarker_lite.task"
    options = PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)), running_mode=PoseLandmarkerOptions.running_mode.IMAGE)
    landmarker = PoseLandmarker.create_from_options(options)

    t_reader, b_reader = iio.get_reader(str(top_vid)), iio.get_reader(str(bot_vid))
    wl_tops, wl_bots, x_offsets_mediapipe, x_offsets_edge, x_offsets_body_fusion = [], [], [], [], []

    for t in sample_times:
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

        res_t = landmarker.detect(Image(image_format=ImageFormat.SRGB, data=tf))
        res_b = landmarker.detect(Image(image_format=ImageFormat.SRGB, data=bf))
        if res_t.pose_landmarks and res_b.pose_landmarks and \
           res_t.pose_landmarks[0][PoseLandmark.NOSE].visibility > 0.3 and res_b.pose_landmarks[0][PoseLandmark.NOSE].visibility > 0.3:
            x_offsets_mediapipe.append(res_t.pose_landmarks[0][PoseLandmark.NOSE].x * tf.shape[1] - res_b.pose_landmarks[0][PoseLandmark.NOSE].x * bf.shape[1])

        if wt is not None and wb is not None:
            if (e_off := find_horizontal_offset_by_edge_correlation(tf, bf, wt, wb)) != 0.0: x_offsets_edge.append(e_off)
            if (b_off := find_horizontal_offset_by_body_fusion(tf, bf, wt, wb, top_expand, bot_expand)) != 0.0: x_offsets_body_fusion.append(b_off)

    landmarker.close(); t_reader.close(); b_reader.close()
    if not wl_tops or not wl_bots: raise ValueError("未能识别到相机的水位线")
    
    available = [n for n, v in [('body_fusion', x_offsets_body_fusion), ('waterline_edge', x_offsets_edge), ('mediapipe_nose', x_offsets_mediapipe)] if v]
    if not available:
        x_offsets = _fallback_by_motion_centroid(top_vid, bot_vid, time_offset, top_rot, bot_rot, meta_t, meta_b, top_zoom_factor)
    else:
        all_medians = [float(np.median(v)) for n, v in [('body_fusion', x_offsets_body_fusion), ('waterline_edge', x_offsets_edge), ('mediapipe_nose', x_offsets_mediapipe)] if v]
        h_offset_raw = float(np.median(all_medians))
        if len(all_medians) >= 2:
            q1, q3 = np.percentile(sorted(all_medians), [25, 75])
            if clipped := [m for m in all_medians if abs(m - h_offset_raw) <= 1.5 * max(q3 - q1, 1.0) + 5]: h_offset_raw = float(np.median(clipped))
        x_offsets = [h_offset_raw]

    return int(np.round(np.median(x_offsets))), int(np.median(wl_tops)), int(np.median(wl_bots))

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
def run_full_pipeline(video_a: Path, video_b: Path, output_dir: Path, output_fps: float = 30.0, max_width: int = 1280, force_top_rotated: str = None, force_bot_rotated: str = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  处理: {video_a.name}  x  {video_b.name}\n{'='*60}")

    print("Step 1/4: 画面解析与身份辨别...")
    rot_a, rot_b, rot_a_ud, rot_b_ud = detect_rotations(video_a, video_b)
    meta_a, meta_b = load_video_meta(video_a), load_video_meta(video_b)
    
    score_a = np.mean([sky_water_separation_score(f) for t in np.linspace(1, min(meta_a.duration, 10), 6) if (f:=sample_frame_at((r:=iio.get_reader(str(video_a))), t, meta_a.fps)) is not None and not r.close()])
    score_b = np.mean([sky_water_separation_score(f) for t in np.linspace(1, min(meta_b.duration, 10), 6) if (f:=sample_frame_at((r:=iio.get_reader(str(video_b))), t, meta_b.fps)) is not None and not r.close()])

    if (score_a if not np.isnan(score_a) else 0) >= (score_b if not np.isnan(score_b) else 0):
        top_video, bottom_video, top_rot, bot_rot = video_a, video_b, rot_a, rot_b_ud
    else:
        top_video, bottom_video, top_rot, bot_rot = video_b, video_a, rot_b, rot_a_ud

    if force_top_rotated is not None: top_rot = force_top_rotated == "true"
    if force_bot_rotated is not None: bot_rot = force_bot_rotated == "true"

    print(f"Step 2/4: 音频波形对齐 (语谱图互相关 + 多策略) ...")
    time_offset, strategies = compute_audio_time_offset(top_video, bottom_video)
    print(f"   => 成功。水下相对水上时间偏移: {time_offset:.4f} 秒")

    print("Step 3/4: 10段x4帧空间定标 (环境物理水位线 + AI 头部追踪对齐) ...")
    h_offset, top_wl, bot_wl = compute_spatial_alignment_10x4(
        top_video, bottom_video, time_offset, top_rot, bot_rot, 
        TOP_ZOOM_FACTOR, TOP_WATERLINE_OFFSET, BOTTOM_WATERLINE_OFFSET
    )
    print(f"   => 放大倍率: {TOP_ZOOM_FACTOR}x")
    print(f"   => 横向偏移对齐: {h_offset} px")
    print(f"   => 物理水面上切线: {top_wl} px, 扩展后保留至 {top_wl + TOP_WATERLINE_OFFSET} px")
    print(f"   => 物理水面下切线: {bot_wl} px, 扩展后保留至 {bot_wl - BOTTOM_WATERLINE_OFFSET} px")

    print("Step 4/4: 全局静态切片物理重构并渲染...")
    cfg = FusionConfig(
        seam=0, feather_px=FEATHER_PX, horizontal_offset=h_offset, time_offset_sec=time_offset,
        top_rotate_180=top_rot, bottom_rotate_180=bot_rot, 
        raw_top_waterline=top_wl, raw_bottom_waterline=bot_wl, 
        top_waterline_offset=TOP_WATERLINE_OFFSET, bottom_waterline_offset=BOTTOM_WATERLINE_OFFSET,
        top_zoom_factor=TOP_ZOOM_FACTOR
    )

    write_fused_video_v2(top_video, bottom_video, cfg, output_dir / "fused_swim_v2.mp4", output_fps, max_width)

    report = {"time_offset_sec": time_offset, "horizontal_offset_px": h_offset, "top_wl": top_wl, "bot_wl": bot_wl}
    (output_dir / "alignment_report_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def run_batch_pipeline(top_dir: Path, bottom_dir: Path, output_dir: Path, output_fps: float = 30.0, max_width: int = 1280) -> list[dict]:
    import re
    top_dir, bottom_dir, output_dir = Path(top_dir), Path(bottom_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_files, bot_files = sorted(top_dir.glob("*.mp4")), sorted(bottom_dir.glob("*.mp4"))

    clip_pattern = re.compile(r'^DJI_\d+_(\d+)_D(_min)?\.mp4$', re.IGNORECASE)
    bot_by_clip = {m.group(1): f for f in bot_files if (m := clip_pattern.match(f.name))}

    results, paired, skipped = [], 0, 0
    for top_video in top_files:
        if not (clip_id := (m.group(1) if (m := clip_pattern.match(top_video.name)) else None)): continue
        if (bottom_video := bot_by_clip.get(clip_id)) is None: continue
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
        except Exception as e: print(e)
    print(f"\n配对完成: {paired} 已融合, {skipped} 已跳过（存在报告文件）")
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