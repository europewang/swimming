"""
swim_video_v2.py
水上/水下游泳视频融合工具 - 工业级音视频对齐版 (音频同步 + AI头部打点 + 10x4抽样)

核心架构：
Step 1: 视频配对与颠倒检测。
Step 2: 时间同步：提取两个视频的音频轨道，进行 FFT 互相关，实现毫秒级绝对时间对齐。
Step 3: 空间定标 (10段x4帧)：
        - 水位线：沿用最稳健的第一代物理环境特征算法 (上: Sobel梯度, 下: 镜像反射轴)。
        - 横向偏移：使用 MediaPipe 骨骼追踪，提取“鼻子”绝对 X 坐标，不受只露头/手的干扰。
        - 聚合：取 40 帧特征的中位数，得到绝对稳健的全局常数。
Step 4: 全局静态融合：沿物理水位线直接无缝拼接，绝对不抖动。
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
    # mediapipe 0.10.x renamed RunningMode -> VisionTaskRunningMode
    RunningMode = getattr(vrm, 'RunningMode', vrm.VisionTaskRunningMode)
    PoseLandmarkerResult = None  # placeholder, result type checked at runtime
except ImportError:
    print("请先安装 MediaPipe: pip install mediapipe")
    exit(1)

# ============================================================
# 融合参数配置
# ============================================================
TOP_ZOOM_FACTOR = 1.5       # 水上视频全局放大倍率（方便后续调整）
WATERLINE_OFFSET = 0       # 拼接缝向下微调量（像素），避开水面折射
FEATHER_PX = 20             # 融合羽化宽度
BLUE, GREEN, RED, YELLOW = (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)

@dataclass
class VideoMeta:
    path: Path; fps: float; width: int; height: int; duration: float; frame_count: int

@dataclass
class FusionConfig:
    seam: int; feather_px: int; horizontal_offset: int; time_offset_sec: float
    top_rotate_180: bool; bottom_rotate_180: bool
    raw_top_waterline: int; raw_bottom_waterline: int; waterline_offset: int
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

# ============================================================
# Step 1: 颠倒检测 (沿用原版)
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
    """
    水下相机旋转检测：用 MediaPipe 检测人体肩膀和脚踝的 Y 坐标。
    - 正常方向：肩膀 Y < 脚踝 Y（头在上方）
    - 翻转了：  肩膀 Y > 脚踝 Y（脚在上方）
    
    调试图：把第一帧的肩-踝连线画出来，保存到当前目录供核对。
    """
    try:
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, PoseLandmark
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
    except ImportError:
        print("MediaPipe 不可用，跳过水下旋转检测，默认不翻转")
        return False

    model_dir = Path.home() / ".mediapipe" / "models"
    model_path = model_dir / "pose_landmarker_lite.task"
    if not model_path.exists():
        import urllib.request, os
        os.makedirs(str(model_dir), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        print(f"下载 MediaPipe Pose 模型到 {model_path} ...")
        urllib.request.urlretrieve(url, str(model_path))

    options = PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                                    running_mode=PoseLandmarkerOptions.running_mode.IMAGE)
    landmarker = PoseLandmarker.create_from_options(options)

    head_above_foot_count = 0  # 肩膀在脚上方 = 正常
    foot_above_head_count = 0  # 脚在肩膀上方 = 倒

    first_frame = None
    first_shoulder_y = first_ankle_y = None

    for f in frames:
        img = Image(image_format=ImageFormat.SRGB, data=f)
        result = landmarker.detect(img)
        if not result.pose_landmarks:
            continue

        lm = result.pose_landmarks[0]
        # 用肩膀中点代替鼻子（水下更容易稳定检测）
        left_shoulder = lm[PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[PoseLandmark.RIGHT_SHOULDER]
        left_ankle = lm[PoseLandmark.LEFT_ANKLE]
        right_ankle = lm[PoseLandmark.RIGHT_ANKLE]

        if (left_shoulder.visibility < 0.3 and right_shoulder.visibility < 0.3) or \
           (left_ankle.visibility < 0.3 and right_ankle.visibility < 0.3):
            continue

        shoulder = left_shoulder if left_shoulder.visibility >= right_shoulder.visibility else right_shoulder
        ankle = left_ankle if left_ankle.visibility >= right_ankle.visibility else right_ankle
        shoulder_y, ankle_y = shoulder.y * f.shape[0], ankle.y * f.shape[0]

        if first_frame is None:
            first_frame = f.copy()
            first_shoulder_y, first_ankle_y = shoulder_y, ankle_y

        if shoulder_y < ankle_y:
            head_above_foot_count += 1
        else:
            foot_above_head_count += 1

    landmarker.close()

    if first_frame is not None:
        dbg = first_frame.copy()
        h, w = dbg.shape[:2]
        ny, ay = int(first_shoulder_y), int(first_ankle_y)
        cx = w // 2
        color = (0, 255, 0) if head_above_foot_count >= foot_above_head_count else (0, 0, 255)
        label = "正常" if head_above_foot_count >= foot_above_head_count else "翻转!"
        cv2.circle(dbg, (cx, ny), 8, (0, 255, 0), -1)
        cv2.circle(dbg, (cx, ay), 8, (255, 0, 0), -1)
        cv2.line(dbg, (cx, ny), (cx, ay), color, 3)
        cv2.putText(dbg, f"肩 y={ny} 踝 y={ay}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(dbg, f"判定: {label}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        debug_path = Path(f"underwater_pose_debug{debug_suffix}.png")
        cv2.imwrite(str(debug_path), cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
        print(f"   [调试图] {debug_path}  (绿点=肩膀, 蓝点=脚踝)")

    total = head_above_foot_count + foot_above_head_count
    if total == 0:
        print("   水下旋转检测: 未检测到有效人体关键点，默认不翻转")
        return False

    print(f"   水下旋转检测: 头在上={head_above_foot_count}帧 脚在上={foot_above_head_count}帧 ({total}帧有效)")
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
    # 修正: is_top_camera_rotated 适用于水上画面(天空vs水面亮度差),
    #       is_underwater_camera_rotated 适用于水下画面(上下梯度特征).
    #       水上视频的帧应送 is_top_camera_rotated，水下视频送 is_underwater_camera_rotated.
    rot_a = is_top_camera_rotated(fa)        # video_a 水上?  (用于水上视频)
    rot_b = is_top_camera_rotated(fb)        # video_b 水上?  (用于水下视频，但方法不适用水下，仅作备用)
    rot_a_ud = is_underwater_camera_rotated(fa, debug_suffix="_videoA")  # video_a 水下?
    rot_b_ud = is_underwater_camera_rotated(fb, debug_suffix="_videoB")  # video_b 水下?
    return rot_a, rot_b, rot_a_ud, rot_b_ud

def sky_water_separation_score(frame: np.ndarray) -> float:
    small = cv2.resize(frame, (320, 180))
    energy = np.abs(cv2.Sobel(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 0, 1, ksize=3)).mean(axis=1)
    b = small[:,:,2].astype(np.float32) / 255.0
    rg = ((small[:,:,0] + small[:,:,1]) / 2).astype(np.float32) / 255.0
    return float(energy[int(180*0.4):int(180*0.65)].max()) * 10.0 + (float(np.mean(b[90:] - rg[90:])) - float(np.mean(b[:90] - rg[:90]))) * 3.0

# ============================================================
# Step 2: 极速时间对齐 (Audio Cross-Correlation)
# ============================================================
# ============================================================
# Step 2: 多策略音频时间对齐
# ============================================================
def _extract_wav(vid_path: Path, tmpdir: Path, name: str) -> Path | None:
    """把视频里的音频流提取为 8000Hz 单声道 WAV"""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out = tmpdir / f"{name}.wav"
    cmd = [ffmpeg_exe, "-y", "-i", str(vid_path), "-vn",
           "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1", str(out)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out if out.exists() else None


def compute_audio_time_offset(video_top: Path, video_bot: Path) -> tuple[float, dict]:
    """
    多策略音频对齐，核心是语谱图互相关（和肉眼对齐方式一致）。
    返回 (best_offset_sec, strategy_info_dict)
    """
    from scipy.signal import butter, sosfilt, spectrogram
    from scipy.ndimage import gaussian_filter
        
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_t = _extract_wav(video_top, Path(tmpdir), "top")
        wav_b = _extract_wav(video_bot, Path(tmpdir), "bot")
        if not wav_t or not wav_b:
            print("警告: 无法提取音频！退回 0 秒偏移。")
            return 0.0, {}

        sr_t, raw_t = wavfile.read(wav_t)
        sr_b, raw_b = wavfile.read(wav_b)

    data_t = raw_t.astype(np.float32) / (np.max(np.abs(raw_t)) + 1e-9)
    data_b = raw_b.astype(np.float32) / (np.max(np.abs(raw_b)) + 1e-9)

    strategies = {}

    # =============================================================
    # 策略 A: 语谱图互相关 (STFT Cross-Correlation)
    # 这是和肉眼对齐完全相同的方式 — 对齐频谱能量包络
    # =============================================================
    nperseg = 1024        # 窗口大小: 1024/8000 = 128ms
    noverlap = int(nperseg * 0.85)  # 85% 重叠，高分辨率
    f_t, t_t, S_t = spectrogram(data_t, fs=sr_t, nperseg=nperseg, noverlap=noverlap)
    f_b, t_b, S_b = spectrogram(data_b, fs=sr_b, nperseg=nperseg, noverlap=noverlap)

    # 转为对数功率谱，用高斯平滑减少噪声（让特征更明显）
    S_t_log = np.log10(S_t + 1e-8)
    S_b_log = np.log10(S_b + 1e-8)
    S_t_log = gaussian_filter(S_t_log, sigma=1.2)
    S_b_log = gaussian_filter(S_b_log, sigma=1.2)

    # 对每条频率 bin 做互相关，取峰值（和肉眼对齐完全一致）
    n_bins = S_t_log.shape[0]  # 频率 bin 数量
    lag_samples_per_bin = []

    for fb_idx in range(n_bins):
        vec_t = S_t_log[fb_idx, :]
        vec_b = S_b_log[fb_idx, :]
        if np.std(vec_t) < 1e-6 or np.std(vec_b) < 1e-6:
            continue
        corr_1d = fftconvolve(vec_b, vec_t[::-1], mode='full')
        lag = int(np.argmax(corr_1d) - (len(vec_t) - 1))
        lag_samples_per_bin.append(lag)

    if lag_samples_per_bin:
        lag_median = float(np.median(lag_samples_per_bin))
        lag_sec_stft = lag_median * float(t_t[1] - t_t[0]) if len(t_t) > 1 else 0.0
        strategies['stft_spectrogram'] = {
            'offset_sec': float(lag_sec_stft),
            'lag_samples_per_bin': [int(x) for x in lag_samples_per_bin],
            'n_bins_used': len(lag_samples_per_bin),
        }

    # =============================================================
    # 策略 B: 分帧局部互相关（鲁棒，防止长音频相位失配）
    # =============================================================
    frame_len = int(0.25 * sr_t)   # 0.25s 短帧
    step = int(0.1 * sr_t)          # 每 0.1s 移动一次
    offsets_per_frame = []
    for start in range(0, len(data_t) - frame_len * 3, step):
        seg_t = data_t[start:start + frame_len]
        # 在 bot 里搜索 ±5s 范围
        search_start = max(0, start - int(5 * sr_t))
        search_end = min(len(data_b), start + frame_len + int(5 * sr_t))
        seg_b = data_b[search_start:search_end]
        if len(seg_b) < len(seg_t):
            continue
        c = fftconvolve(seg_b, seg_t[::-1], mode='valid')
        if len(c) > 0:
            local_lag = (np.argmax(c) - (len(seg_t) - 1)) / float(sr_t)
            offsets_per_frame.append(local_lag + (search_start - start) / float(sr_t))

    if offsets_per_frame:
        offsets_arr = np.array(offsets_per_frame)
        # 用直方图峰值（而非简单中位数），避免极端值干扰）
        counts, bin_edges = np.histogram(offsets_arr, bins=100)
        peak_bin = np.argmax(counts)
        lag_sec_sliding = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
        strategies['sliding_window'] = {
            'offset_sec': float(lag_sec_sliding),
            'all_offsets': offsets_arr.tolist(),
            'histogram_peak': float(lag_sec_sliding),
        }

    # =============================================================
    # 策略 C: 高通滤波 + 互相关（去除水中低频哄鸣）
    # =============================================================
    sos_hp = butter(4, 300 / (sr_t * 0.5), btype='high', output='sos')
    data_t_hp = sosfilt(sos_hp, data_t)
    data_b_hp = sosfilt(sos_hp, data_b)
    corr_hp = fftconvolve(data_b_hp, data_t_hp[::-1], mode='full')
    lag_hp = (np.argmax(corr_hp) - (len(data_t_hp) - 1)) / float(sr_t)
    strategies['highpass_300hz'] = {
        'offset_sec': lag_hp,
        'corr_normalized': float(corr_hp.max() / (np.linalg.norm(data_t_hp) * np.linalg.norm(data_b_hp) + 1e-9)),
    }

    # =============================================================
    # 综合决策: 滑动窗口直方图峰值作为首选
    # （STFT 语谱图互相关对水中声学变形效果差，直方图峰值更鲁棒）
    # =============================================================
    sw_info = strategies.get('sliding_window', {})
    hp_info = strategies.get('highpass_300hz', {})
    stft_info = strategies.get('stft_spectrogram', {})

    # 优先用滑动窗口直方图峰值
    if sw_info and stft_info:
        sw_peak = sw_info.get('histogram_peak', 0.0)
        stft_offset = stft_info.get('offset_sec', 0.0)
        # 如果两者差太大，用直方图峰值（水下直方图分布可能不集中）
        if abs(sw_peak - stft_offset) > 0.5:
            best_offset = sw_peak
            best_strategy = 'sliding_window'
        else:
            best_offset = stft_offset
            best_strategy = 'stft_spectrogram'
    elif sw_info:
        best_offset = sw_info.get('histogram_peak', 0.0)
        best_strategy = 'sliding_window'
    elif hp_info:
        best_offset = hp_info.get('offset_sec', 0.0)
        best_strategy = 'highpass_300hz'
    else:
        best_offset = 0.0
        best_strategy = 'unknown'

    print(f"\n   音频对齐策略对比:")
    for name, info in strategies.items():
        o = info.get('offset_sec', 0.0)
        marker = f' <-- 首选({best_strategy})' if name == best_strategy else ''
        print(f"   [{name}] offset={o:+.4f}s{marker}")
    print(f"   => 最终选择: {best_strategy} = {best_offset:+.4f}s")

    return best_offset, strategies


def visualize_audio_sync(time_offset: float, strategies: dict,
                          output_dir: Path, video_top: Path, video_bot: Path) -> None:
    """生成语谱图对齐调试图 —— 和用户肉眼对齐方式完全一致"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    tmpdir = output_dir / "audio_debug_tmp"
    tmpdir.mkdir(exist_ok=True)

    wav_t_path = tmpdir / "top_audio.wav"
    wav_b_path = tmpdir / "bot_audio.wav"
    subprocess.run([ffmpeg_exe, "-y", "-i", str(video_top), "-vn",
                    "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1", str(wav_t_path)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([ffmpeg_exe, "-y", "-i", str(video_bot), "-vn",
                    "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1", str(wav_b_path)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not wav_t_path.exists() or not wav_b_path.exists():
        return

    sr, data_t = wavfile.read(wav_t_path)
    _, data_b = wavfile.read(wav_b_path)
    data_t = data_t.astype(np.float32) / (np.max(np.abs(data_t)) + 1e-9)
    data_b = data_b.astype(np.float32) / (np.max(np.abs(data_b)) + 1e-9)

    nperseg = 1024
    noverlap = int(nperseg * 0.85)
    f_t, t_t, S_t = spectrogram(data_t, fs=sr, nperseg=nperseg, noverlap=noverlap)
    f_b, t_b, S_b = spectrogram(data_b, fs=sr, nperseg=nperseg, noverlap=noverlap)
    S_t_db = 10 * np.log10(S_t + 1e-8)
    S_b_db = 10 * np.log10(S_b + 1e-8)

    bot_t_samples = int(round(time_offset * sr))
    t_b_aligned = t_b + time_offset

    # ---- 图1: 语谱图直接叠加对比 (最核心的图) ----
    fig, axes = plt.subplots(3, 1, figsize=(20, 16), facecolor='#0d1117')

    # 对齐前 (时间对齐 = 0)
    t_b_raw = t_b
    vmin, vmax = min(S_t_db.min(), S_b_db.min()), max(S_t_db.max(), S_b_db.max())

    ax0 = axes[0]
    ax0.set_facecolor('#0d1117')
    im0a = ax0.pcolormesh(t_t, f_t, S_t_db, shading='auto', cmap='inferno',
                            vmin=vmin, vmax=vmax, alpha=1.0)
    im0b = ax0.pcolormesh(t_b_raw, f_b, S_b_db, shading='auto', cmap='viridis',
                            vmin=vmin, vmax=vmax, alpha=0.75)
    ax0.set_ylabel('Frequency (Hz)', color='white', fontsize=11)
    ax0.set_title('BEFORE Alignment — top (red/orange) vs bottom (green/blue) raw spectrograms',
                   color='#ff7b72', fontsize=12)
    ax0.tick_params(colors='white')
    ax0.set_ylim(0, 4000)

    # 对齐后 (使用检测到的偏移)
    ax1 = axes[1]
    ax1.set_facecolor('#0d1117')
    im1a = ax1.pcolormesh(t_t, f_t, S_t_db, shading='auto', cmap='inferno',
                            vmin=vmin, vmax=vmax, alpha=1.0)
    im1b = ax1.pcolormesh(t_b_aligned, f_b, S_b_db, shading='auto', cmap='viridis',
                            vmin=vmin, vmax=vmax, alpha=0.75)
    ax1.set_ylabel('Frequency (Hz)', color='white', fontsize=11)
    ax1.set_title(f'AFTER Alignment (offset={time_offset:+.4f}s) — spectrograms overlaid',
                   color='#7ee787', fontsize=12)
    ax1.tick_params(colors='white')
    ax1.set_ylim(0, 4000)

    # 水平拼接视图（上下排列在同一时间轴）
    ax2 = axes[2]
    ax2.set_facecolor('#0d1117')
    ax2.pcolormesh(t_t, f_t, S_t_db, shading='auto', cmap='inferno',
                    vmin=vmin, vmax=vmax)
    ax2.set_ylabel('Frequency (Hz)', color='white', fontsize=10)
    ax2.set_title('Aligned Spectrogram: TOP (air)', color='#ffa657', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.set_ylim(0, 4000)
    ax2_bot = ax2.twinx()
    ax2_bot.set_facecolor('#0d1117')
    ax2_bot.pcolormesh(t_b_aligned, f_b, S_b_db, shading='auto', cmap='viridis',
                         vmin=vmin, vmax=vmax)
    ax2_bot.set_ylabel('BOTTOM (underwater) freq', color='#79c0ff', fontsize=10)
    ax2_bot.tick_params(colors='white')
    ax2_bot.set_ylim(0, 4000)
    ax2.set_xlabel('Time (seconds)', color='white', fontsize=11)

    for ax in [ax0, ax1, ax2, ax2_bot]:
        if ax is not ax2_bot:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(output_dir / "audio_debug" / "spectrogram_alignment.png"),
                dpi=120, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)

    # ---- 图2: 各策略互相关曲线 ----
    fig2, axs2 = plt.subplots(2, 1, figsize=(18, 10), facecolor='#0d1117')

    ax2a = axs2[0]
    ax2a.set_facecolor('#0d1117')

    # STFT: 绘制每条频率 bin 的 lag 直方图
    stft_info = strategies.get('stft_spectrogram', {})
    lag_bins = stft_info.get('lag_samples_per_bin', [])
    if lag_bins:
        t_res = (t_t[1] - t_t[0]) if len(t_t) > 1 else 1.0
        lag_secs = [l * t_res for l in lag_bins]
        ax2a.hist(lag_secs, bins=60, color='#58a6ff', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax2a.axvline(x=time_offset, color='#ff6b6b', linewidth=2.5, linestyle='--',
                      label=f'STFT best offset: {time_offset:+.4f}s')
        ax2a.axvline(x=0, color='#cccccc', linewidth=1, linestyle=':', alpha=0.5)
        ax2a.set_xlabel('Lag (seconds)', color='white', fontsize=11)
        ax2a.set_ylabel('Frequency bins count', color='white', fontsize=11)
        ax2a.set_title('STFT Per-Bin Lag Histogram — how many frequency bins vote for each lag',
                       color='white', fontsize=12)
        ax2a.tick_params(colors='white')
        ax2a.legend(facecolor='#161b22', edgecolor='white', labelcolor='white')
        ax2a.grid(True, alpha=0.1, color='white')

    # 滑动窗口直方图
    sw_info = strategies.get('sliding_window', {})
    offsets = sw_info.get('all_offsets', [])
    if offsets:
        ax2b = axs2[1]
        ax2b.set_facecolor('#0d1117')
        ax2b.hist(offsets, bins=80, color='#3fb950', alpha=0.65, edgecolor='white', linewidth=0.5)
        hp_info = strategies.get('highpass_300hz', {})
        ax2b.axvline(x=sw_info.get('histogram_peak', 0), color='#3fb950', linewidth=2,
                      label=f'Sliding hist peak: {sw_info.get("histogram_peak", 0):+.4f}s')
        if hp_info:
            ax2b.axvline(x=hp_info.get('offset_sec', 0), color='#ffa657', linewidth=2,
                          label=f'HP 300Hz: {hp_info.get("offset_sec", 0):+.4f}s')
        ax2b.axvline(x=time_offset, color='#ff6b6b', linewidth=2.5, linestyle='--',
                      label=f'Final offset: {time_offset:+.4f}s')
        ax2b.axvline(x=0, color='#cccccc', linewidth=1, linestyle=':', alpha=0.5)
        ax2b.set_xlabel('Lag (seconds)', color='white', fontsize=11)
        ax2b.set_ylabel('Frame count', color='white', fontsize=11)
        ax2b.set_title('Sliding Window Local Lag Histogram — histogram peak = most voted lag',
                       color='white', fontsize=12)
        ax2b.tick_params(colors='white')
        ax2b.legend(facecolor='#161b22', edgecolor='white', labelcolor='white')
        ax2b.grid(True, alpha=0.1, color='white')

    plt.tight_layout()
    fig2.savefig(str(output_dir / "audio_debug" / "audio_correlation.png"),
                 dpi=120, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig2)

    # ---- 图3: 高质量语谱图拼接对比（垂直排列） ----
    fig3, axs3 = plt.subplots(2, 1, figsize=(20, 12), facecolor='#0d1117')
    fig3.suptitle(f'Spectrogram Side-by-Side (aligned by {time_offset:+.4f}s)', color='white', fontsize=13)

    # ---- 图3: 高质量语谱图拼接对比（垂直排列，同显示时间轴）----
    fig3, axs3 = plt.subplots(2, 1, figsize=(20, 12), facecolor='#0d1117')
    fig3.suptitle(f'Spectrogram Side-by-Side (aligned by {time_offset:+.4f}s)', color='white', fontsize=13)

    axs3[0].set_facecolor('#0d1117')
    axs3[0].pcolormesh(t_t, f_t, S_t_db, shading='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axs3[0].set_ylabel('Frequency (Hz)', color='white', fontsize=11)
    axs3[0].set_title('TOP (air) — original time axis', color='#ffa657', fontsize=12)
    axs3[0].tick_params(colors='white')
    axs3[0].set_ylim(0, 4000)
    axs3[0].tick_params(labelbottom=False)

    axs3[1].set_facecolor('#0d1117')
    axs3[1].pcolormesh(t_b_aligned, f_b, S_b_db, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs3[1].set_ylabel('Frequency (Hz)', color='white', fontsize=11)
    axs3[1].set_xlabel('Time (seconds)', color='white', fontsize=11)
    axs3[1].set_title('BOTTOM (underwater) — shifted by offset', color='#79c0ff', fontsize=12)
    axs3[1].tick_params(colors='white')
    axs3[1].set_ylim(0, 4000)

    for ax in axs3:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig3.savefig(str(output_dir / "audio_debug" / "spectrogram_stacked.png"),
                 dpi=120, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig3)

    # ---- 保存音频 ----
    wavfile.write(str(output_dir / "audio_debug" / "top_raw_8k.wav"),
                  sr, (data_t * 32767).astype(np.int16))
    wavfile.write(str(output_dir / "audio_debug" / "bot_raw_8k.wav"),
                  sr, (data_b * 32767).astype(np.int16))

    top_out = output_dir / "audio_debug" / "top_original.wav"
    bot_out = output_dir / "audio_debug" / f"bottom_aligned_{time_offset:+.3f}s.wav"
    bot_offset_samples = int(round(time_offset * sr))
    if bot_offset_samples >= 0:
        seg = data_b[bot_offset_samples:bot_offset_samples + len(data_t)]
    else:
        seg = np.concatenate([np.zeros(-bot_offset_samples), data_b[:len(data_t) + bot_offset_samples]])
    if len(seg) < len(data_t):
        seg = np.pad(seg, (0, len(data_t) - len(seg)))
    wavfile.write(str(top_out), sr, (data_t * 32767).astype(np.int16))
    wavfile.write(str(bot_out), sr, (seg[:len(data_t)] * 32767).astype(np.int16))

    # 保存 summary
    summary = {
        "time_offset_sec": float(time_offset),
        "strategies": {k: {kk: vv for kk, vv in v.items()
                          if kk not in ('stft_S_t_log', 'stft_S_b_log', 'stft_t_axis', 'all_offsets', 'lag_samples_per_bin')}
                       for k, v in strategies.items()},
        "debug_files": {
            "spectrogram_alignment": "spectrogram_alignment.png",
            "audio_correlation": "audio_correlation.png",
            "spectrogram_stacked": "spectrogram_stacked.png",
            "top_original_wav": "top_original.wav",
            "bottom_aligned_wav": f"bottom_aligned_{time_offset:+.3f}s.wav",
            "top_raw_8k": "top_raw_8k.wav",
            "bot_raw_8k": "bot_raw_8k.wav",
        }
    }
    (output_dir / "audio_debug" / "audio_sync_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    import shutil as _shutil
    _shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n   Audio debug generated in: {output_dir / 'audio_debug'}")
    print(f"   => spectrogram_alignment.png  [KEY] before/after overlay — verify with your eye")
    print(f"   => audio_correlation.png      lag histograms per strategy")
    print(f"   => spectrogram_stacked.png    vertical stacked spectrograms (aligned)")
    print(f"   => top_original.wav / bottom_aligned_{time_offset:+.3f}s.wav")


# ============================================================
# Step 3: 原版水线算法 + AI 头部追踪 (10x4 空间标定)
# ============================================================
def find_best_waterline_row(gray: np.ndarray) -> int | None:
    """上相机：第一版原汁原味 Sobel 梯度寻水线"""
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
    """下相机：第一版原汁原味镜像反射对称寻水线"""
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
    """备用防抖方案：如果 MediaPipe 找不到人，退回到运动水花质心"""
    if frame.shape != bg.shape: bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
    diff = cv2.cvtColor(cv2.absdiff(frame, bg), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    M = cv2.moments(mask)
    return float(M["m10"] / M["m00"]) if M["m00"] > 1000 else None


def compute_waterline_edge_profile(frame: np.ndarray, waterline_row: int, strip_height: int = 12) -> np.ndarray:
    """提取水线附近一条垂直条纹（用于横向互相关对齐）"""
    h, w = frame.shape[:2]
    row = int(waterline_row)
    strip_top = max(0, row - strip_height)
    strip_bot = min(h, row + strip_height)
    strip = frame[strip_top:strip_bot, :]
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY).astype(np.float64)
    edge = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)).mean(axis=0)
    edge = gaussian_filter(edge, sigma=1.5)
    return edge


def find_horizontal_offset_by_edge_correlation(
    top_frame: np.ndarray, bot_frame: np.ndarray,
    top_wl: int, bot_wl: int,
    strip_height: int = 12,
    max_search: int = 80
) -> float:
    if top_frame.shape[1] != bot_frame.shape[1]:
        return 0.0

    prof_t = compute_waterline_edge_profile(top_frame, top_wl, strip_height)
    prof_b = compute_waterline_edge_profile(bot_frame, bot_wl, strip_height)

    if len(prof_t) < 10 or len(prof_b) < 10:
        return 0.0

    min_len = min(len(prof_t), len(prof_b))
    prof_t = prof_t[:min_len]
    prof_b = prof_b[:min_len]

    corr = fftconvolve(prof_b, prof_t[::-1], mode='full')
    mid = len(prof_t) - 1
    search_l = max(0, mid - max_search)
    search_r = min(len(corr) - 1, mid + max_search)
    if search_r <= search_l:
        return 0.0

    corr_window = corr[search_l:search_r + 1]
    best_lag = int(np.argmax(corr_window)) + search_l - mid
    return float(best_lag)


def extract_person_mask_simple(frame: np.ndarray) -> np.ndarray | None:
    """用 GrabCut 提取人物前景蒙版（无 ML 模型，纯图像分割）"""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 粗略矩形：假设人占画面中央偏下 60% 区域
    margin_x = int(w * 0.1)
    rect = (margin_x, int(h * 0.15), w - margin_x * 2, int(h * 0.7))
    try:
        cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        # 轻微膨胀，连接断裂的肢体
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask * 255
    except Exception:
        return None


def fuse_person_by_waterline(
    top_frame: np.ndarray, bot_frame: np.ndarray,
    top_wl: int, bot_wl: int,
    h_offset: int
) -> np.ndarray | None:
    """
    把上下两帧按水线 + 横向偏移拼接成一张"完整人体"图。
    - top_frame 水上部分从 y=0 到 y=top_wl
    - bot_frame 水下部分从 y=bot_wl 开始
    - bot 横向平移 h_offset px（正值=向右移）
    - 羽化过渡，输出 3 通道图
    """
    h_t, w = top_frame.shape[:2]
    h_b = bot_frame.shape[0]

    # 横向对齐下帧
    if h_offset != 0:
        M = np.float32([[1, 0, h_offset], [0, 1, 0]])
        bot_shifted = cv2.warpAffine(bot_frame, M, (w, h_b), borderMode=cv2.BORDER_REPLICATE)
    else:
        bot_shifted = bot_frame

    # 输出画布高度：水线之上（top部分） + 水线之下（bot部分在水下的可见高度）
    out_h = top_wl + (h_b - bot_wl)
    if out_h <= 0:
        return None

    # 分配画布
    canvas = np.zeros((out_h, w, 3), dtype=np.uint8)

    # 填入 top（水上部分，上方0到水线）
    top_rows = min(h_t, out_h)
    canvas[0:top_rows] = top_frame[0:top_rows]

    # 填入 bot（水下部分，从 bot_wl 开始，映射到画布 top_wl 处）
    bot_in_canvas_y = top_wl          # bot 顶部(y=0) 放在画布的 top_wl 处
    bot_src_start = 0
    bot_src_end = h_b

    # 截断到画布范围内
    if bot_in_canvas_y < 0:
        bot_src_start = -bot_in_canvas_y
        bot_in_canvas_y = 0
    if bot_in_canvas_y + (bot_src_end - bot_src_start) > out_h:
        bot_src_end = bot_src_start + (out_h - bot_in_canvas_y)

    src_len = bot_src_end - bot_src_start
    if src_len > 0 and bot_in_canvas_y < out_h:
        canvas[bot_in_canvas_y:bot_in_canvas_y + src_len] = bot_shifted[bot_src_start:bot_src_end]

    # 羽化：画布 y=top_wl 处做上下过渡
    feather = 15
    y_start = max(0, top_wl - feather)
    y_end = min(out_h, top_wl + feather)
    for y in range(y_start, y_end):
        alpha = (y_end - y) / (y_end - y_start + 1e-9)
        if y < h_t:
            canvas[y] = (canvas[y] * alpha + top_frame[y] * (1 - alpha)).astype(np.uint8)

    return canvas


def score_fused_body_similarity(fused: np.ndarray, seam_y: int, top_frame: np.ndarray, bot_frame: np.ndarray, bot_wl: int, h_offset: int, w: int) -> float:
    """
    拼接缝相似度评分：在拼接缝附近，提取上半部分和下半部分的边缘特征，
    通过互相关判断上下拼接是否自然对齐。
    
    当横向偏移正确时，上下两部分在水线处的边缘应该相互衔接，
    互相关峰值最高；偏移错误时，边缘无法衔接，峰值降低。
    
    seam_y: 拼接缝的 y 坐标（= top_wl）
    """
    h, fw = fused.shape[:2]
    strip = 25   # 考察拼接缝上下 ±25 行
    
    # 提取上半部分（从水线向上）和下半部分（从水线向下）
    top_region = fused[max(0, seam_y - strip):seam_y]          # 水线上方 strip 行
    bot_region = fused[seam_y:min(h, seam_y + strip)]          # 水线下方 strip 行
    
    if top_region.size == 0 or bot_region.size == 0:
        return 0.0
    
    # 如果 bot 被横向平移了，bot_region 的边缘位置也发生了偏移
    # 正确的 bot_region 应该和 top_region 在同一条水平线上能自然衔接
    # 即：top_region 的底部边缘和 bot_region 的顶部边缘形状应该相似
    
    top_gray = cv2.cvtColor(top_region, cv2.COLOR_RGB2GRAY).astype(np.float64)
    bot_gray = cv2.cvtColor(bot_region, cv2.COLOR_RGB2GRAY).astype(np.float64)
    
    # 只取每行做平均，得到 1D 水平边缘剖面
    top_profile = np.abs(cv2.Sobel(top_gray, cv2.CV_64F, 1, 0, ksize=3)).mean(axis=0)
    bot_profile = np.abs(cv2.Sobel(bot_gray, cv2.CV_64F, 1, 0, ksize=3)).mean(axis=0)
    
    # 归一化
    top_profile = (top_profile - top_profile.mean()) / (top_profile.std() + 1e-6)
    bot_profile = (bot_profile - bot_profile.mean()) / (bot_profile.std() + 1e-6)
    
    # 互相关（bot_profile 在 top_profile 上滑动）
    if len(top_profile) < 5 or len(bot_profile) < 5:
        return 0.0
    
    # 横向偏移 h 对 bot_profile 的影响：
    # bot_profile 的列 x 对应世界坐标 (x + h)
    # top_profile 的列 x 对应世界坐标 x
    # 所以对齐要求：top_profile[x] ≈ bot_profile[x - h]
    # 相关峰的位置就是 (h) 的估计
    
    corr = fftconvolve(top_profile, bot_profile[::-1], mode='full')
    mid = len(bot_profile) - 1
    # 只在 ±50 范围内搜索（合理偏移范围）
    search_l = max(0, mid - 50)
    search_r = min(len(corr) - 1, mid + 50)
    if search_r <= search_l:
        return 0.0
    
    corr_window = corr[search_l:search_r + 1]
    peak_corr = float(np.max(corr_window))
    
    # 额外惩罚：如果 bot 被平移太多（超出画面边界），给低分
    # h_offset > 0: bot 向右移，如果太大右边会被截断
    # h_offset < 0: bot 向左移，如果太大左边会被截断（通过 BORDER_REPLICATE 补边）
    # 由于 warpAffine 用 BORDER_REPLICATE，实际不会出现截断，只是边缘会有复制伪影
    # 这里我们额外检查拼接缝附近的颜色一致性
    seam_row = fused[seam_y] if seam_y < h else fused[-1]
    seam_std = float(np.std(seam_row))
    # 水线处颜色变化越小（std 低），说明拼接越自然
    color_consistency = 1.0 / (1.0 + seam_std / 10.0)
    
    # 综合得分：互相关峰值（最重要）+ 颜色一致性（辅助）
    score = peak_corr * 0.85 + color_consistency * 15.0
    return score


def find_horizontal_offset_by_body_fusion(
    top_frame: np.ndarray, bot_frame: np.ndarray,
    top_wl: int, bot_wl: int,
    debug_dir: Path | None = None,
    frame_id: str = "0"
) -> tuple[float, Path | None]:
    """
    人体合成偏移检测：
    在 [-80, +80] px 范围内尝试不同横向偏移，
    把上下两帧拼成完整人体，取拼接缝纹理最自然的偏移量。

    返回: (best_offset, debug_image_path or None)
    """
    w = top_frame.shape[1]
    search_range = range(-80, 81, 4)   # 每 4px 采样，减少计算量
    scores = {}

    for offset in search_range:
        fused = fuse_person_by_waterline(top_frame, bot_frame, top_wl, bot_wl, offset)
        if fused is None:
            continue
        scores[offset] = score_fused_body_similarity(
            fused, seam_y=top_wl,
            top_frame=top_frame, bot_frame=bot_frame,
            bot_wl=bot_wl, h_offset=offset, w=top_frame.shape[1]
        )

    if not scores:
        return 0.0, None

    # 取最高分的偏移
    best_offset = max(scores, key=lambda k: scores[k])
    best_score = scores[best_offset]

    # 生成调试图
    debug_path = None
    if debug_dir is not None:
        debug_path = _save_body_fusion_debug(
            debug_dir, top_frame, bot_frame, top_wl, bot_wl,
            scores, best_offset, best_score, frame_id
        )

    return float(best_offset), debug_path


def _save_body_fusion_debug(
    debug_dir: Path, top_frame: np.ndarray, bot_frame: np.ndarray,
    top_wl: int, bot_wl: int,
    scores: dict[int, float],
    best_offset: int, best_score: float,
    frame_id: str
) -> Path:
    """生成横向偏移调试图：多偏移量叠加对比 + 分数曲线"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 1. 分数曲线图 ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor='#0d1117')

    ax_score = axes[0]
    ax_score.set_facecolor('#0d1117')
    offsets_sorted = sorted(scores.keys())
    score_vals = [scores[o] for o in offsets_sorted]
    ax_score.plot(offsets_sorted, score_vals, color='#58a6ff', linewidth=2, marker='o', markersize=4)
    ax_score.axvline(x=best_offset, color='#ff6b6b', linewidth=2, linestyle='--',
                     label=f'Best offset={best_offset:+d}px, score={best_score:.4f}')
    ax_score.set_xlabel('Horizontal Offset (px)', color='white', fontsize=11)
    ax_score.set_ylabel('Fusion Quality Score', color='white', fontsize=11)
    ax_score.set_title('Body Fusion Score vs Horizontal Offset\n(higher = better seam continuity)', color='white', fontsize=12)
    ax_score.tick_params(colors='white')
    ax_score.legend(facecolor='#161b22', edgecolor='white', labelcolor='white')
    ax_score.grid(True, alpha=0.15, color='white')
    ax_score.set_facecolor('#0d1117')

    # --- 2. 关键偏移量叠加效果图（取几个代表性偏移）---
    ax_gallery = axes[1]
    ax_gallery.set_facecolor('#0d1117')
    ax_gallery.set_title('Top-4 Offset Fusion Previews (waterline seam at center)', color='white', fontsize=12)
    ax_gallery.axis('off')

    # 取分数最高的前4个偏移
    top_offsets = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:4]
    n_show = len(top_offsets)

    for col, off in enumerate(top_offsets):
        fused = fuse_person_by_waterline(top_frame, bot_frame, top_wl, bot_wl, off)
        if fused is None:
            continue
        fuse_small = cv2.resize(fused, (320, 180))
        fuse_rgb = cv2.cvtColor(fuse_small, cv2.COLOR_BGR2RGB)
        score_c = scores[off]
        marker = ' ★ BEST' if off == best_offset else ''
        ax_gallery.imshow(fuse_rgb, extent=[col * 0.25, (col + 1) * 0.25, 0.5, 1.0])
        ax_gallery.text(col * 0.25 + 0.125, 0.45, f'{off:+d}px\n{score_c:.3f}{marker}',
                        ha='center', va='top', color='#ffa657' if off == best_offset else '#cccccc',
                        fontsize=9, transform=ax_gallery.transAxes)

    # --- 3. 原始帧标注图（水线 + 横向偏移示意）---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0d1117')
    fig2.suptitle(f'Horizontal Offset Debug — Frame {frame_id}  (best={best_offset:+d}px, score={best_score:.4f})',
                  color='white', fontsize=13)

    def annotate_frame(ax, frame, wl, title, color='#ffa657'):
        h, w = frame.shape[:2]
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axhline(y=wl, color=color, linewidth=2, linestyle='--', label=f'Waterline y={wl}')
        ax.set_title(title, color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.legend(facecolor='#161b22', edgecolor='white', labelcolor='white', fontsize=8)

    annotate_frame(axes2[0, 0], top_frame, top_wl, 'TOP Camera (waterline)', '#ffa657')
    annotate_frame(axes2[0, 1], bot_frame, bot_wl, 'BOTTOM Camera (reflection axis)', '#79c0ff')

    # 横向偏移示意：把 bot 叠在 top 上（半透明）
    h_overlay = max(top_frame.shape[0], bot_frame.shape[0])
    w_overlay = max(top_frame.shape[1], bot_frame.shape[1])
    top_vis = cv2.resize(top_frame, (w_overlay, h_overlay))
    bot_vis = cv2.resize(bot_frame, (w_overlay, h_overlay))
    overlay = cv2.addWeighted(top_vis, 0.5, bot_vis, 0.5, 0)
    axes2[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes2[1, 0].set_title(f'Overlay (no correction) — red=top, blue=bot', color='white', fontsize=11)
    axes2[1, 0].tick_params(colors='white')

    # 正确偏移的叠加
    if best_offset != 0:
        M = np.float32([[1, 0, best_offset], [0, 1, 0]])
        bot_corrected = cv2.warpAffine(bot_vis, M, (w_overlay, h_overlay), borderMode=cv2.BORDER_REPLICATE)
    else:
        bot_corrected = bot_vis
    overlay_corrected = cv2.addWeighted(top_vis, 0.5, bot_corrected, 0.5, 0)
    axes2[1, 1].imshow(cv2.cvtColor(overlay_corrected, cv2.COLOR_BGR2RGB))
    axes2[1, 1].set_title(f'Corrected Overlay (shift={best_offset:+d}px)', color='#7ee787', fontsize=11)
    axes2[1, 1].tick_params(colors='white')

    debug_dir.mkdir(parents=True, exist_ok=True)
    path1 = debug_dir / f"body_fusion_scores_{frame_id}.png"
    path2 = debug_dir / f"frame_annotated_{frame_id}.png"
    fig.savefig(str(path1), dpi=100, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    fig2.savefig(str(path2), dpi=100, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig2)

    return path1


def find_horizontal_offset_by_template_matching(
    top_frame: np.ndarray, bot_frame: np.ndarray,
    top_wl: int, strip_height: int = 10
) -> float:
    """
    模板匹配法（已废弃，保留用于对比）：以上相机水线附近一条窄带为模板，
    在下相机整个宽度范围内做滑动互相关。
    """
    h, w = top_frame.shape[:2]
    row = int(top_wl)
    t_top = max(0, row - strip_height)
    t_bot = min(h, row + strip_height)
    template = top_frame[t_top:t_bot, :]
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY).astype(np.float64)

    bot_roi_h = min(strip_height * 2 + 20, bot_frame.shape[0])
    bot_roi = bot_frame[:bot_roi_h, :]
    bot_gray = cv2.cvtColor(bot_roi, cv2.COLOR_RGB2GRAY).astype(np.float64)

    if bot_gray.shape[1] < template_gray.shape[1]:
        return 0.0

    template_edge = np.abs(cv2.Sobel(template_gray, cv2.CV_64F, 1, 0, ksize=3)).mean(axis=0)
    best_offset, best_score = 0.0, -1e9
    for r in range(bot_gray.shape[0] - 1):
        search_edge = np.abs(cv2.Sobel(bot_gray[r:r+1, :], cv2.CV_64F, 1, 0, ksize=3)).flatten()
        if len(search_edge) < len(template_edge):
            continue
        corr = fftconvolve(search_edge, template_edge[::-1], mode='valid')
        if len(corr) == 0:
            continue
        peak = float(np.max(corr))
        if peak > best_score:
            best_score = peak
            best_offset = float(np.argmax(corr))

    return -best_offset


def compute_spatial_alignment_10x4(top_vid: Path, bot_vid: Path, time_offset: float, top_rot: bool, bot_rot: bool) -> tuple[int, int, int]:
    """10段 x 4帧的极速采样策略，提取全局空间属性的中位数
    
    横向偏移检测采用多策略融合：
    1. MediaPipe 鼻子 X 坐标（当游泳者清晰露出头部时最准）
    2. 水线边缘互相关（物理直接，依赖水位线检测）
    3. 人体合成偏移检测（按水线拼接上下半身，评分最高的偏移）
    最终取各策略结果的中位数融合。
    """
    meta_t, meta_b = load_video_meta(top_vid), load_video_meta(bot_vid)
    ov_start, ov_end = 0.0, min(meta_t.duration, meta_b.duration - time_offset)
    if ov_end <= ov_start: raise ValueError("没有有效重叠时间")

    # 构建抽样时间点（聚焦于水线区域更有代表性的时间段）
    seg_dur = (ov_end - ov_start) / 10.0
    sample_times = []
    for i in range(10):
        mid = ov_start + (i + 0.5) * seg_dur
        for offset in range(4):
            sample_times.append(mid + offset * (1.0 / meta_t.fps))

    # 模型路径解析
    model_dir = Path.home() / ".mediapipe" / "models"
    model_path = model_dir / "pose_landmarker_lite.task"
    if not model_path.exists():
        import urllib.request
        os.makedirs(str(model_dir), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        print(f"下载 MediaPipe Pose 模型到 {model_path} ...")
        urllib.request.urlretrieve(url, str(model_path))

    t_reader, b_reader = iio.get_reader(str(top_vid)), iio.get_reader(str(bot_vid))

    wl_tops, wl_bots = [], []
    x_offsets_mediapipe, x_offsets_edge, x_offsets_body_fusion = [], [], []

    # 调试输出目录：每 N 帧输出一个调试图
    debug_dir = Path("outputs_v2") / "horizontal_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_interval = 10   # 每 10 帧（约每段 1 次）输出调试图
    debug_paths = []

    # 初始化新版 MediaPipe PoseLandmarker (IMAGE 模式)
    options = PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=PoseLandmarkerOptions.running_mode.IMAGE
    )
    landmarker = PoseLandmarker.create_from_options(options)

    for idx, t in enumerate(sample_times):
        tf = sample_frame_at(t_reader, t, meta_t.fps)
        bf = sample_frame_at(b_reader, max(0.0, t + time_offset), meta_b.fps)
        if tf is None or bf is None:
            continue

        tf, bf = rotate_frame(tf, top_rot), rotate_frame(bf, bot_rot)
        if tf.shape != bf.shape:
            bf = cv2.resize(bf, (tf.shape[1], tf.shape[0]))

        # 1. 物理水位线计算
        wt = find_best_waterline_row(cv2.cvtColor(tf, cv2.COLOR_RGB2GRAY))
        wb = find_underwater_reflection_axis(bf)
        if wt is not None:
            wl_tops.append(wt)
        if wb is not None:
            wl_bots.append(wb)

        # 2. MediaPipe 鼻子 X 坐标（策略1）
        img_t = Image(image_format=ImageFormat.SRGB, data=tf)
        img_b = Image(image_format=ImageFormat.SRGB, data=bf)
        res_t = landmarker.detect(img_t)
        res_b = landmarker.detect(img_b)

        if (res_t.pose_landmarks and
            res_t.pose_landmarks[0][PoseLandmark.NOSE].visibility > 0.3 and
            res_b.pose_landmarks and
            res_b.pose_landmarks[0][PoseLandmark.NOSE].visibility > 0.3):
            xt = res_t.pose_landmarks[0][PoseLandmark.NOSE].x * tf.shape[1]
            xb = res_b.pose_landmarks[0][PoseLandmark.NOSE].x * bf.shape[1]
            x_offsets_mediapipe.append(xt - xb)

        # 3. 水线边缘互相关（策略2）- 仅当两个水位线都检测成功时
        if wt is not None and wb is not None:
            edge_offset = find_horizontal_offset_by_edge_correlation(tf, bf, wt, wb)
            if edge_offset != 0.0:
                x_offsets_edge.append(edge_offset)

        # 4. 人体合成偏移检测（策略3）- 核心新策略
        if wt is not None and wb is not None:
            body_offset, dbg_path = find_horizontal_offset_by_body_fusion(
                tf, bf, wt, wb,
                debug_dir=debug_dir if (idx % debug_interval == 0) else None,
                frame_id=f"t{idx:.1f}s_{int(t*100)/100:.2f}"
            )
            if body_offset != 0.0:
                x_offsets_body_fusion.append(body_offset)
            if dbg_path is not None:
                debug_paths.append(dbg_path)

    landmarker.close()
    t_reader.close()
    b_reader.close()
    
    # 数据校验与后处理
    if not wl_tops: raise ValueError("未能识别到上相机的水位线")
    if not wl_bots: raise ValueError("未能识别到下相机的水位线")
    
    # ============================================================
    # 多策略横向偏移综合决策
    # ============================================================
    all_valid_strategies = {
        'mediapipe_nose': x_offsets_mediapipe,
        'waterline_edge': x_offsets_edge,
        'body_fusion': x_offsets_body_fusion,
    }

    strategy_results = {}
    for name, values in all_valid_strategies.items():
        if values:
            median_val = float(np.median(values))
            strategy_results[name] = {
                'median': median_val,
                'count': len(values),
                'values': values,
            }

    print(f"\n   横向偏移多策略检测结果:")
    print(f"   {'策略':<20} {'有效样本':>8} {'中位数偏移':>12} {'样本列表（前5个）'}")
    for name, res in strategy_results.items():
        vals_str = ', '.join(f"{v:.1f}" for v in res['values'][:5])
        print(f"   {name:<20} {res['count']:>8} {res['median']:>+12.2f} px   [{vals_str}]")

    if debug_paths:
        print(f"\n   调试图像已保存到: {debug_dir}/")
        print(f"   共 {len(debug_paths)} 帧调试图（每隔 ~{debug_interval} 帧采样）")

    # 综合决策：优先用人体合成（直观物理），其次水线边缘，最后 MediaPipe
    priority_order = ['body_fusion', 'waterline_edge', 'mediapipe_nose']
    available = [name for name in priority_order if name in strategy_results]

    if not available:
        print("   警告: 所有横向偏移策略均失效，尝试水花质心 fallback...")
        x_offsets = _fallback_by_motion_centroid(top_vid, bot_vid, time_offset, top_rot, bot_rot, meta_t, meta_b)
    else:
        all_medians = [strategy_results[n]['median'] for n in available]
        h_offset_raw = float(np.median(all_medians))
        print(f"   => 综合决策: 融合 {len(available)} 个策略, 中位数偏移 = {h_offset_raw:+.2f} px")

        # IQR 异常值裁剪
        if len(all_medians) >= 2:
            q1, q3 = np.percentile(sorted(all_medians), [25, 75])
            iqr = max(q3 - q1, 1.0)
            clipped = [m for m in all_medians if abs(m - h_offset_raw) <= 1.5 * iqr + 5]
            if clipped:
                h_offset_raw = float(np.median(clipped))
                print(f"   => IQR裁剪后: {h_offset_raw:+.2f} px")

        x_offsets = [h_offset_raw]

    return int(np.round(np.median(x_offsets))), int(np.median(wl_tops)), int(np.median(wl_bots))


def _fallback_by_motion_centroid(
    top_vid: Path, bot_vid: Path, time_offset: float,
    top_rot: bool, bot_rot: bool,
    meta_t, meta_b
) -> list:
    """水花质心 fallback — 只在所有其他策略失效时使用"""
    try:
        tr = iio.get_reader(str(top_vid)); br = iio.get_reader(str(bot_vid))
        bg_t = np.median([rotate_frame(tr.get_data(i), top_rot) for i in range(10)], axis=0).astype(np.uint8)
        bg_b = np.median([rotate_frame(br.get_data(i), bot_rot) for i in range(10)], axis=0).astype(np.uint8)
        tr.close(); br.close()

        ov_end = min(meta_t.duration, meta_b.duration - time_offset)
        sample_times = list(np.linspace(1.0, max(1.0, ov_end - 1.0), 15))

        fallback_x = []
        tr, br = iio.get_reader(str(top_vid)), iio.get_reader(str(bot_vid))
        for t in sample_times:
            tf = sample_frame_at(tr, t, meta_t.fps)
            bf = sample_frame_at(br, max(0.0, t + time_offset), meta_b.fps)
            if tf is None or bf is None:
                continue
            tf = rotate_frame(tf, top_rot)
            bf = rotate_frame(bf, bot_rot)
            if tf.shape != bf.shape:
                bf = cv2.resize(bf, (tf.shape[1], tf.shape[0]))
            cx_t = get_motion_x_centroid(tf, bg_t)
            cx_b = get_motion_x_centroid(bf, bg_b)
            if cx_t is not None and cx_b is not None:
                fallback_x.append(cx_t - cx_b)
        tr.close(); br.close()
        if fallback_x:
            return fallback_x
    except Exception:
        pass
    return [0]

# ============================================================
# Step 4: 物理硬切拼接融合
# ============================================================
def blend_waterline_fusion(
    top_frame: np.ndarray, bottom_frame: np.ndarray, top_wl: int, bot_wl: int,
    horizontal_offset: int, feather_px: int, waterline_offset: int
) -> np.ndarray:
    h_top, w = top_frame.shape[:2]
    h_bot = bottom_frame.shape[0]

    # 1. 执行下半部的横向位移对齐
    if horizontal_offset != 0:
        bottom_frame = cv2.warpAffine(bottom_frame, np.float32([[1, 0, horizontal_offset], [0, 1, 0]]), (w, h_bot), borderMode=cv2.BORDER_REPLICATE)

    # 2. 算好天衣无缝的拼接高度
    out_h = top_wl + (h_bot - bot_wl)
    if out_h <= 0: return top_frame.copy()

    # 3. 画布坐标映射 (将两根线叠在一起)
    top_y_s, top_y_e = 0, min(out_h, h_top)
    bot_y_s, bot_y_e = top_wl - bot_wl, top_wl - bot_wl + h_bot
    bot_in_s, bot_in_e = 0, h_bot
    if bot_y_s < 0: bot_in_s, bot_y_s = -bot_y_s, 0
    if bot_y_e > out_h: bot_in_e, bot_y_e = h_bot - (bot_y_e - out_h), out_h

    # 4. 生成羽化过渡蒙版
    seam_y = max(0, min(out_h, top_wl + waterline_offset))
    feather = max(0, feather_px)
    top_mask = np.zeros((out_h, 1, 1), dtype=np.float32)
    y_start, y_end = max(0, seam_y - feather), min(out_h, seam_y + feather)
    
    if y_end > y_start:
        top_mask[0:y_start] = 1.0
        top_mask[y_start:y_end] = np.linspace(1.0, 0.0, y_end - y_start).reshape(-1, 1, 1)
    else: top_mask[0:seam_y] = 1.0

    # 5. 组合
    acc = np.zeros((out_h, w, 3), dtype=np.float32)
    if top_y_e > top_y_s: acc[top_y_s:top_y_e] += top_frame[0:top_y_e] * top_mask[top_y_s:top_y_e]
    if bot_y_e > bot_y_s: acc[bot_y_s:bot_y_e] += bottom_frame[bot_in_s:bot_in_e] * (1.0 - top_mask[bot_y_s:bot_y_e])

    return acc.astype(np.uint8)

def write_fused_video_v2(top_video: Path, bottom_video: Path, cfg: FusionConfig, output_path: Path, output_fps: float = 30.0, max_width: int = 1280) -> None:
    top_meta, bottom_meta = load_video_meta(top_video), load_video_meta(bottom_video)
    top_reader, bottom_reader = iio.get_reader(str(top_video)), iio.get_reader(str(bottom_video))
    
    # 重叠区间：top 从 0s 开始，bot 从 -offset 开始（领先）
    # top 覆盖: [0,              top_dur)
    # bot 覆盖: [-offset,       bottom_dur - offset)
    # 公共区间: [0,              min(top_dur, bottom_dur - offset)]
    overlap_start = 0.0
    overlap_end = min(top_meta.duration, bottom_meta.duration - cfg.time_offset_sec)

    writer, current, frame_count = None, overlap_start, 0
    while current < overlap_end:
        top_frame = sample_frame_at(top_reader, current, top_meta.fps)
        # bot 的时间戳 = top 时间戳 + time_offset
        # （若 bottom 相机比 top 早开 2.194s，则 top T 时刻的画面 = bottom T+2.194 时刻的画面）
        bot_ts = current + cfg.time_offset_sec
        bottom_frame = sample_frame_at(bottom_reader, max(0.0, bot_ts), bottom_meta.fps)
        if top_frame is None or bottom_frame is None:
            current += 1.0 / output_fps
            continue

        top_frame, bottom_frame = rotate_frame(top_frame, cfg.top_rotate_180), rotate_frame(bottom_frame, cfg.bottom_rotate_180)

        # 全局分辨率缩放（把视频统一到 max_width）
        scale = max_width / top_frame.shape[1] if top_frame.shape[1] > max_width else 1.0
        if scale != 1.0:
            top_frame = cv2.resize(top_frame, (max_width, int(top_frame.shape[0] * scale)))
            bottom_frame = cv2.resize(bottom_frame, (max_width, int(bottom_frame.shape[0] * scale)))

        # 水上视频在最终分辨率上再放大（zoom 的效果就是减小视野、突出水上部分）
        if cfg.top_zoom_factor != 1.0:
            new_h = int(top_frame.shape[0] * cfg.top_zoom_factor)
            new_w = int(top_frame.shape[1] * cfg.top_zoom_factor)
            top_frame = cv2.resize(top_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if top_frame.shape != bottom_frame.shape:
                bottom_frame = cv2.resize(bottom_frame, (top_frame.shape[1], top_frame.shape[0]))

        fused = blend_waterline_fusion(
            top_frame, bottom_frame,
            int(round(cfg.raw_top_waterline * scale * cfg.top_zoom_factor)),
            int(round(cfg.raw_bottom_waterline * scale)),
            int(round(cfg.horizontal_offset * cfg.top_zoom_factor)),
            int(round(cfg.feather_px * scale * cfg.top_zoom_factor)),
            int(round(cfg.waterline_offset * scale * cfg.top_zoom_factor)),
        )

        # 确保高度是偶数（libx264 要求）
        if fused.shape[0] % 2 != 0:
            fused = fused[:-1, :] if fused.shape[0] > 1 else fused

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
def run_full_pipeline(video_a: Path, video_b: Path, output_dir: Path,
                     output_fps: float = 30.0, max_width: int = 1280,
                     force_top_rotated: str = None,
                     force_bot_rotated: str = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  处理: {video_a.name}  x  {video_b.name}")
    print(f"{'='*60}")

    print("Step 1/4: 画面解析与身份辨别...")
    rot_a, rot_b, rot_a_ud, rot_b_ud = detect_rotations(video_a, video_b)
    meta_a, meta_b = load_video_meta(video_a), load_video_meta(video_b)
    
    score_a = np.mean([sky_water_separation_score(f) for t in np.linspace(1, min(meta_a.duration, 10), 6) if (f:=sample_frame_at((r:=iio.get_reader(str(video_a))), t, meta_a.fps)) is not None and not r.close()])
    score_b = np.mean([sky_water_separation_score(f) for t in np.linspace(1, min(meta_b.duration, 10), 6) if (f:=sample_frame_at((r:=iio.get_reader(str(video_b))), t, meta_b.fps)) is not None and not r.close()])

    if (score_a if not np.isnan(score_a) else 0) >= (score_b if not np.isnan(score_b) else 0):
        top_video, bottom_video = video_a, video_b
        # video_a → 水上：其 rot_a（is_top_camera_rotated）有效；video_b → 水下：其 rot_b_ud 有效
        top_rot, bot_rot = rot_a, rot_b_ud
    else:
        top_video, bottom_video = video_b, video_a
        # video_b → 水上：其 rot_b（is_top_camera_rotated）有效；video_a → 水下：其 rot_a_ud 有效
        top_rot, bot_rot = rot_b, rot_a_ud

    print(f"   视频身份判定: video_a={'水上' if top_video == video_a else '水下'}, video_b={'水上' if top_video == video_a else '水下'}")
    print(f"   旋转检测: top_rot={top_rot}, bot_rot={bot_rot}  (True=需翻转180°)")

    if force_top_rotated is not None:
        top_rot = force_top_rotated == "true"
        print(f"   ⚠  手动覆盖 top_rot={top_rot}")
    if force_bot_rotated is not None:
        bot_rot = force_bot_rotated == "true"
        print(f"   ⚠  手动覆盖 bot_rot={bot_rot}")

    print(f"Step 2/4: 音频波形对齐 (语谱图互相关 + 多策略) ...")
    time_offset, strategies = compute_audio_time_offset(top_video, bottom_video)
    print(f"   => 成功。水下相对水上时间偏移: {time_offset:.4f} 秒")

    print(f"Step 2b/4: 生成音频同步调试图...")
    (output_dir / "audio_debug").mkdir(exist_ok=True)
    visualize_audio_sync(time_offset, strategies, output_dir, top_video, bottom_video)

    print("Step 3/4: 10段x4帧空间定标 (环境物理水位线 + AI 头部追踪对齐) ...")
    h_offset, top_wl, bot_wl = compute_spatial_alignment_10x4(top_video, bottom_video, time_offset, top_rot, bot_rot)
    print(f"   => 横向偏移对齐: {h_offset} px")
    print(f"   => 物理水面上切线: {top_wl} px")
    print(f"   => 物理水面下切线: {bot_wl} px")

    print("Step 4/4: 全局静态切片物理融合并渲染...")
    cfg = FusionConfig(
        seam=0, feather_px=FEATHER_PX, horizontal_offset=h_offset, time_offset_sec=time_offset,
        top_rotate_180=top_rot, bottom_rotate_180=bot_rot, 
        raw_top_waterline=top_wl, raw_bottom_waterline=bot_wl, waterline_offset=WATERLINE_OFFSET,
        top_zoom_factor=TOP_ZOOM_FACTOR
    )

    write_fused_video_v2(top_video, bottom_video, cfg, output_dir / "fused_swim_v2.mp4", output_fps, max_width)

    report = {"time_offset_sec": time_offset, "horizontal_offset_px": h_offset, "top_wl": top_wl, "bot_wl": bot_wl}
    (output_dir / "alignment_report_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def run_batch_pipeline(top_dir: Path, bottom_dir: Path, output_dir: Path,
                       output_fps: float = 30.0, max_width: int = 1280) -> list[dict]:
    """
    批量处理模式：自动配对两个文件夹中同名视频，逐对融合。
    配对规则：按镜头编号 (_XXXX_) 匹配，水上水下视频的该部分相同即为配对。
    文件名格式: DJI_YYYYMMDDHHMMSS_XXXX_D_min.mp4
      - YYYYMMDDHHMMSS: 录制时间戳，两路可能相差数秒
      - XXXX:           镜头编号，同一场景的水上水下视频共用此编号
    """
    import re

    top_dir, bottom_dir, output_dir = Path(top_dir), Path(bottom_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_files = sorted(top_dir.glob("*.mp4"))
    bot_files = sorted(bottom_dir.glob("*.mp4"))

    # ── 按镜头编号匹配 ──────────────────────────────────────────────
    # DJI_20260513180601_0002_D_min.mp4  →  clip_id = "0002"
    clip_pattern = re.compile(r'^DJI_\d+_(\d+)_D(_min)?\.mp4$', re.IGNORECASE)

    def get_clip_id(path: Path) -> str | None:
        m = clip_pattern.match(path.name)
        return m.group(1) if m else None

    # 建立 clip_id → 文件路径 的倒排索引（水下）
    bot_by_clip: dict[str, Path] = {}
    for f in bot_files:
        cid = get_clip_id(f)
        if cid:
            bot_by_clip[cid] = f

    print(f"\n{'='*60}")
    print(f"  批量融合模式")
    print(f"{'='*60}")
    print(f"  水上文件夹:   {top_dir}")
    print(f"  水下文件夹:   {bottom_dir}")
    print(f"  输出文件夹:   {output_dir}")
    print(f"  水上视频:     {len(top_files)} 个")
    print(f"  水下视频:     {len(bot_files)} 个")
    print(f"{'='*60}\n")

    results = []
    paired = 0
    skipped = 0

    for top_video in top_files:
        clip_id = get_clip_id(top_video)
        if clip_id is None:
            print(f"  ⚠️  无法解析镜头编号，跳过: {top_video.name}")
            skipped += 1
            continue

        bottom_video = bot_by_clip.get(clip_id)

        if bottom_video is None:
            print(f"  ⏭  未找到配对 (clip={clip_id}): {top_video.stem}")
            skipped += 1
            continue

        # 输出目录：每个视频对创建一个子文件夹
        pair_name = f"{top_video.stem}__{bottom_video.stem}"
        pair_out = output_dir / pair_name
        pair_out.mkdir(parents=True, exist_ok=True)

        # 以 alignment_report.json 是否存在判断是否已处理过
        if (pair_out / "alignment_report.json").exists():
            print(f"\n  ⏭  已存在，跳过: {pair_name}/")
            skipped += 1
            continue

        print(f"\n{'─'*60}")
        print(f"  [{paired + 1}] {top_video.stem}  (clip={clip_id})")
        print(f"{'─'*60}")

        try:
            report = run_full_pipeline(top_video, bottom_video, pair_out, output_fps, max_width)
            # 同时保存一份对齐报告到输出根目录
            report_path = pair_out / "alignment_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append({**report, "pair": pair_name, "top": str(top_video), "bottom": str(bottom_video)})
            paired += 1
            print(f"\n  ✅ 完成 [{paired}]: {pair_name}/")
        except Exception as e:
            print(f"\n  ❌ 失败 [{pair_name}]: {e}")
            import traceback
            traceback.print_exc()
            results.append({"pair": pair_name, "top": str(top_video), "bottom": str(bottom_video), "error": str(e)})

    # 保存汇总报告
    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(json.dumps({
        "total": len(top_files),
        "paired": paired,
        "skipped": skipped,
        "results": results
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  批量处理完成")
    print(f"  配对成功: {paired} 对")
    print(f"  跳过:     {skipped} 个（无配对或已存在）")
    print(f"  汇总:     {summary_path}")
    print(f"{'='*60}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="水上/水下游泳视频融合工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  单对模式:
    python swim_video_v2.py --video-a video1.mp4 --video-b video2.mp4 --output-dir outputs

  批量模式（推荐）:
    python swim_video_v2.py --batch-top video_merge/video1_min --batch-bottom video_merge/video2_min --batch-output video_merge/融合
        """
    )
    # 单对模式
    parser.add_argument("--video-a", type=Path, help="水上视频路径")
    parser.add_argument("--video-b", type=Path, help="水下视频路径")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v2"))
    # 批量模式
    parser.add_argument("--batch-top", type=Path, help="水上视频文件夹")
    parser.add_argument("--batch-bottom", type=Path, help="水下视频文件夹")
    parser.add_argument("--batch-output", type=Path, help="批量输出文件夹")
    # 共用参数
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-width", type=int, default=1280)
    # 手动覆盖旋转检测（当自动检测结果错误时使用）
    # --force-top-rotated: 强制指定水上视频是否需要翻转180°
    # --force-bot-rotated: 强制指定水下视频是否需要翻转180°
    parser.add_argument("--force-top-rotated", type=str, default=None,
                        choices=["true", "false"],
                        help="强制水上视频是否翻转180°，不指定则自动检测")
    parser.add_argument("--force-bot-rotated", type=str, default=None,
                        choices=["true", "false"],
                        help="强制水下视频是否翻转180°，不指定则自动检测")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 批量模式
    if args.batch_top and args.batch_bottom and args.batch_output:
        run_batch_pipeline(
            args.batch_top, args.batch_bottom, args.batch_output,
            output_fps=args.fps, max_width=args.max_width
        )
    # 单对模式
    elif args.video_a and args.video_b:
        run_full_pipeline(args.video_a, args.video_b, args.output_dir,
                         output_fps=args.fps, max_width=args.max_width,
                         force_top_rotated=args.force_top_rotated,
                         force_bot_rotated=args.force_bot_rotated)
    else:
        print("错误: 请指定 --video-a 和 --video-b（单对模式），或指定 --batch-top / --batch-bottom / --batch-output（批量模式）。")
        exit(1)