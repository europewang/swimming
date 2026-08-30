"""
swim_video_v3.py
水上/水下游泳视频融合工具 - 工业级音视频对齐版 (音频同步 + AI头部打点 + 10x4抽样)

【基础操作方式】
1. 单组视频融合：
   python swim_video_v3.py --video-a 水上视频.mp4 --video-b 水下视频.mp4
   - 若水上视频是倒的，加上参数: --top-rotated
   - 若水下视频是倒的，加上参数: --bottom-rotated

2. 批量处理（按文件名排序一一对应）：
   python swim_video_v3.py --batch-top 水上文件夹路径 --batch-bottom 水下文件夹路径 --batch-output 输出文件夹路径
   - 同样可使用 --top-rotated 和 --bottom-rotated 来指定该批次视频的翻转情况。

核心架构：
Step 1: 手动指定视频目录及方向 (正反)。
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
TOP_ZOOM_FACTOR = 1.7             # 水上视频全局放大倍率（拉近镜头，主体更大）
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
    # 统一使用与水上视频相同的水位线识别方案：基于强烈水平纹理变化的边缘能量检测
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return find_best_waterline_row(gray)

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
    
    # 在调试目录下专门建一个子目录保存水位线的调试图像
    wl_debug_dir = None
    if debug_dir is not None:
        wl_debug_dir = debug_dir / "waterline_debug"
        wl_debug_dir.mkdir(exist_ok=True, parents=True)

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

        # 将中间过程每段的划线都输出到文件夹中
        if wl_debug_dir is not None:
            debug_frame_t = tf.copy()
            debug_frame_b = bf.copy()
            if wt is not None:
                cv2.line(debug_frame_t, (0, wt), (debug_frame_t.shape[1], wt), YELLOW, 2)
                cv2.putText(debug_frame_t, f"Waterline Y={wt}", (10, wt - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
                out_path_t = wl_debug_dir / f"segment_{frame_idx}_top_waterline.jpg"
                cv2.imwrite(str(out_path_t), cv2.cvtColor(debug_frame_t, cv2.COLOR_RGB2BGR))
                
            if wb is not None:
                cv2.line(debug_frame_b, (0, wb), (debug_frame_b.shape[1], wb), CYAN, 2)
                cv2.putText(debug_frame_b, f"Waterline Y={wb}", (10, wb - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
                out_path_b = wl_debug_dir / f"segment_{frame_idx}_bottom_waterline.jpg"
                cv2.imwrite(str(out_path_b), cv2.cvtColor(debug_frame_b, cv2.COLOR_RGB2BGR))

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
def run_full_pipeline(top_video: Path, bottom_video: Path, output_dir: Path, top_rotated: bool, bottom_rotated: bool, output_fps: float = 30.0, max_width: int = 1280) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / "debug_body_fusion"
    debug_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}\n  处理: {top_video.name}  x  {bottom_video.name}\n{'='*60}")

    print("Step 1/4: 使用用户指定的视频方向...")
    rot_top = top_rotated
    rot_bottom = bottom_rotated
    
    print(f"\n{'='*60}")
    print(f"  最终决定:")
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
            "top_rotated": rot_top,
            "bottom_rotated": rot_bottom
        }
    }
    return report

def run_batch_pipeline(top_dir: Path, bottom_dir: Path, output_dir: Path, top_rotated: bool, bottom_rotated: bool, output_fps: float = 30.0, max_width: int = 1280) -> list[dict]:
    import re
    from datetime import datetime
    
    top_dir, bottom_dir, output_dir = Path(top_dir), Path(bottom_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_files = sorted(top_dir.glob("*.mp4"))
    bot_files = sorted(bottom_dir.glob("*.mp4"))

    # DJI 文件名格式: DJI_<录制时间戳14位>_<剪辑序号4位>_D[_min].mp4
    # 例如: DJI_20260508195343_0001_D_min.mp4
    clip_pattern = re.compile(r'^DJI_(\d{14})_\d+_D(_min)?\.mp4$', re.IGNORECASE)

    def parse_dji_timestamp(ts_str: str) -> int:
        """将 DJI 时间戳字符串转换为 Unix 时间戳（秒）"""
        try:
            dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
            return int(dt.timestamp())
        except:
            return 0

    results, paired, skipped, failed_pairs = [], 0, 0, []
    
    # 智能配对逻辑：遍历所有的水上视频，在水下视频中寻找时间戳最接近且不超过 60 秒的视频
    used_bots = set()
    missing_tops = []
    
    for top_video in top_files:
        top_match = clip_pattern.match(top_video.name)
        if not top_match:
            failed_pairs.append((top_video, None, "水上视频文件名不符合 DJI 格式，无法解析时间戳"))
            continue
            
        top_ts = parse_dji_timestamp(top_match.group(1))
        
        best_bot = None
        best_diff = float('inf')
        
        for bot_video in bot_files:
            if bot_video in used_bots:
                continue
                
            bot_match = clip_pattern.match(bot_video.name)
            if not bot_match:
                continue
                
            bot_ts = parse_dji_timestamp(bot_match.group(1))
            diff_sec = abs(top_ts - bot_ts)
            
            if diff_sec < best_diff:
                best_diff = diff_sec
                best_bot = bot_video
                
        if best_bot is None or best_diff > 60:
            if best_bot is not None:
                print(f"  ❌  未找到匹配: {top_video.name} 最接近的水下视频是 {best_bot.name}，但时间差为 {best_diff} 秒 (>60秒)")
            else:
                print(f"  ❌  未找到匹配: {top_video.name} 没有可用的水下视频")
            missing_tops.append(top_video)
            continue
            
        # 成功找到匹配
        bottom_video = best_bot
        used_bots.add(bottom_video)
        
        pair_out = output_dir / f"{top_video.stem}__{bottom_video.stem}"
        report_file = pair_out / "alignment_report_v2.json"
        if report_file.exists():
            print(f"  ⏭️  跳过（已存在）: {report_file.name}  →  {pair_out.name}")
            skipped += 1
            continue
        pair_out.mkdir(parents=True, exist_ok=True)
        try:
            report = run_full_pipeline(top_video, bottom_video, pair_out, top_rotated, bottom_rotated, output_fps, max_width)
            report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            paired += 1
            results.append(report)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"   [跳过] {type(e).__name__}: {e}")
            failed_pairs.append((top_video, bottom_video, str(e)))
            skipped += 1

    print(f"\n{'='*60}")
    print(f"  批处理完成汇总")
    print(f"{'='*60}")
    print(f"  ✅ 已融合: {paired}")
    print(f"  ⏭️  已跳过: {skipped} (含 {len(failed_pairs)} 个失败)")
    
    unpaired_bot_count = len(bot_files) - len(used_bots)
    if missing_tops:
        print(f"  ❌ 缺失配对: {len(missing_tops)} 个水上视频没有找到匹配的水下视频 (时间差>60s)")
    if unpaired_bot_count > 0:
        print(f"  ❌ 冗余视频: {unpaired_bot_count} 个水下视频没有被使用")

    if failed_pairs:
        print(f"\n  --- 融合失败 ---")
        for top_v, bot_v, err in failed_pairs:
            bot_name = bot_v.name if bot_v else "无"
            print(f"    ❌ {top_v.name} x {bot_name}")
            print(f"       原因: {err}")

    if not missing_tops and unpaired_bot_count == 0 and not failed_pairs:
        print(f"\n  ✅ 所有视频均已正确配对并融合，无问题")

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="水上/水下游泳视频融合工具 (手动指定版)")
    parser.add_argument("--video-a", type=Path, help="水上视频路径")
    parser.add_argument("--video-b", type=Path, help="水下视频路径")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v3"))
    parser.add_argument("--batch-top", type=Path, help="批量处理的水上视频目录")
    parser.add_argument("--batch-bottom", type=Path, help="批量处理的水下视频目录")
    parser.add_argument("--batch-output", type=Path, help="批量处理的输出目录")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--top-rotated", action="store_true", help="指定水上视频是否需要翻转180度")
    parser.add_argument("--bottom-rotated", action="store_true", help="指定水下视频是否需要翻转180度")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.batch_top and args.batch_bottom and args.batch_output:
        run_batch_pipeline(args.batch_top, args.batch_bottom, args.batch_output, args.top_rotated, args.bottom_rotated, output_fps=args.fps, max_width=args.max_width)
    elif args.video_a and args.video_b:
        run_full_pipeline(args.video_a, args.video_b, args.output_dir, args.top_rotated, args.bottom_rotated, output_fps=args.fps, max_width=args.max_width)