"""
quick_audio_check.py — 快速音频对齐检查工具
只做：提取音频 → 画 dB 语谱图 → 输出对齐前后对比图

用法：
  python quick_audio_check.py ^
    --top  "D:\...\DJI_20260515183640_0012_D_min.mp4" ^
    --bot  "D:\...\DJI_20260515183635_0012_D_min.mp4" ^
    --offset 2.44 ^
    --output outputs_v2\audio_debug
"""
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def extract_wav(video_path: Path, tmpdir: Path, name: str) -> Path | None:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out = tmpdir / f"{name}.wav"
    cmd = [ffmpeg_exe, "-y", "-i", str(video_path), "-vn",
           "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1", str(out)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out if out.exists() else None


def main():
    parser = argparse.ArgumentParser(description='快速音频对齐检查')
    parser.add_argument('--top', type=Path, required=True, help='水上视频路径')
    parser.add_argument('--bot', type=Path, required=True, help='水下视频路径')
    parser.add_argument('--offset', type=float, default=None,
                        help='时间偏移秒数（秒），例如 2.44。留空则自动检测。')
    parser.add_argument('--output', type=Path, default=Path('outputs_v2/audio_debug'),
                        help='输出目录')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"提取音频: {args.top.name}")
    print(f"提取音频: {args.bot.name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_t = extract_wav(args.top, Path(tmpdir), "top")
        wav_b = extract_wav(args.bot, Path(tmpdir), "bot")

        if not wav_t or not wav_b:
            print("错误: 音频提取失败"); return

        sr, data_t = wavfile.read(wav_t)
        _, data_b = wavfile.read(wav_b)

    data_t = data_t.astype(np.float32) / (np.max(np.abs(data_t)) + 1e-9)
    data_b = data_b.astype(np.float32) / (np.max(np.abs(data_b)) + 1e-9)

    # ---- 自动检测偏移（STFT 分帧直方图法）----
    if args.offset is None:
        print("自动检测偏移中...")
        from scipy.signal import fftconvolve
        frame_len = int(0.25 * sr)
        step = int(0.1 * sr)
        offsets = []
        for start in range(0, len(data_t) - frame_len * 3, step):
            seg_t = data_t[start:start + frame_len]
            search_start = max(0, start - int(5 * sr))
            search_end = min(len(data_b), start + frame_len + int(5 * sr))
            seg_b = data_b[search_start:search_end]
            if len(seg_b) < len(seg_t): continue
            c = fftconvolve(seg_b, seg_t[::-1], mode='valid')
            if len(c) > 0:
                lag = (np.argmax(c) - (len(seg_t) - 1)) / float(sr)
                offsets.append(lag + (search_start - start) / float(sr))
        offsets_arr = np.array(offsets)
        counts, bin_edges = np.histogram(offsets_arr, bins=100)
        peak_bin = np.argmax(counts)
        args.offset = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
        print(f"自动检测偏移: {args.offset:+.4f}s (直方图峰值)")

    print(f"\n生成对齐检查图 (offset = {args.offset:+.4f}s)...")

    # ---- 生成语谱图 ----
    nperseg = 1024
    noverlap = int(nperseg * 0.85)
    f_t, t_t, S_t = spectrogram(data_t, fs=sr, nperseg=nperseg, noverlap=noverlap)
    f_b, t_b, S_b = spectrogram(data_b, fs=sr, nperseg=nperseg, noverlap=noverlap)
    S_t_db = 10 * np.log10(S_t + 1e-8)
    S_b_db = 10 * np.log10(S_b + 1e-8)

    t_b_aligned = t_b + args.offset
    vmin = min(S_t_db.min(), S_b_db.min())
    vmax = max(S_t_db.max(), S_b_db.max())

    # ---- 图1: 上下语谱图对齐前后对比 (核心图) ----
    fig, axes = plt.subplots(2, 1, figsize=(24, 12), facecolor='#0d1117')
    fig.suptitle(
        f'Audio Spectrogram Alignment Check  |  offset = {args.offset:+.4f}s  |  '
        f'TOP (above water) vs BOTTOM (underwater)',
        color='white', fontsize=14, y=0.98
    )

    # 对齐前
    ax0 = axes[0]
    ax0.set_facecolor('#0d1117')
    ax0.pcolormesh(t_t, f_t, S_t_db, shading='auto', cmap='inferno',
                    vmin=vmin, vmax=vmax, alpha=1.0)
    ax0.pcolormesh(t_b, f_b, S_b_db, shading='auto', cmap='viridis',
                    vmin=vmin, vmax=vmax, alpha=0.75)
    ax0.set_ylabel('Frequency (Hz)', color='white', fontsize=11)
    ax0.set_title(f'BEFORE Alignment — raw overlay (offset=0s)',
                   color='#ff7b72', fontsize=12)
    ax0.tick_params(colors='white')
    ax0.set_ylim(0, 4000)

    # 对齐后
    ax1 = axes[1]
    ax1.set_facecolor('#0d1117')
    ax1.pcolormesh(t_t, f_t, S_t_db, shading='auto', cmap='inferno',
                    vmin=vmin, vmax=vmax, alpha=1.0)
    ax1.pcolormesh(t_b_aligned, f_b, S_b_db, shading='auto', cmap='viridis',
                    vmin=vmin, vmax=vmax, alpha=0.75)
    ax1.set_ylabel('Frequency (Hz)', color='white', fontsize=11)
    ax1.set_xlabel('Time (seconds)', color='white', fontsize=11)
    ax1.set_title(f'AFTER Alignment — bottom shifted by {args.offset:+.4f}s',
                   color='#7ee787', fontsize=12)
    ax1.tick_params(colors='white')
    ax1.set_ylim(0, 4000)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = args.output / f'quick_check_{args.offset:+.3f}s.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    print(f"  保存: {out_path}")

    # ---- 图2: 垂直拼接语谱图（同一时间轴）----
    fig2, axs2 = plt.subplots(2, 1, figsize=(24, 12), facecolor='#0d1117')
    fig2.suptitle(
        f'Aligned Spectrograms  |  offset = {args.offset:+.4f}s',
        color='white', fontsize=13
    )

    axs2[0].set_facecolor('#0d1117')
    axs2[0].pcolormesh(t_t, f_t, S_t_db, shading='auto', cmap='magma',
                         vmin=vmin, vmax=vmax)
    axs2[0].set_ylabel('Freq (Hz)', color='white')
    axs2[0].set_title('TOP (above water)', color='#ffa657')
    axs2[0].tick_params(colors='white')
    axs2[0].set_ylim(0, 4000)
    axs2[0].tick_params(labelbottom=False)

    axs2[1].set_facecolor('#0d1117')
    axs2[1].pcolormesh(t_b_aligned, f_b, S_b_db, shading='auto', cmap='viridis',
                         vmin=vmin, vmax=vmax)
    axs2[1].set_ylabel('Freq (Hz)', color='white')
    axs2[1].set_xlabel('Time (seconds) — aligned', color='white')
    axs2[1].set_title('BOTTOM (underwater)', color='#79c0ff')
    axs2[1].tick_params(colors='white')
    axs2[1].set_ylim(0, 4000)

    for ax in axs2:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out2 = args.output / f'quick_check_stacked_{args.offset:+.3f}s.png'
    fig2.savefig(str(out2), dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig2)
    print(f"  保存: {out2}")

    # ---- 保存对齐后的 WAV 片段 ----
    offset_samples = int(round(args.offset * sr))
    if offset_samples >= 0:
        seg = data_b[offset_samples:offset_samples + len(data_t)]
    else:
        seg = np.concatenate([np.zeros(-offset_samples), data_b[:len(data_t) + offset_samples]])
    seg = np.pad(seg, (0, max(0, len(data_t) - len(seg))))[:len(data_t)]

    wavfile.write(str(args.output / f'bottom_aligned_{args.offset:+.3f}s.wav'),
                  sr, (seg * 32767).astype(np.int16))

    print(f"\n完成！")
    print(f"  打开 {out_path} 人工确认对齐效果")
    print(f"  如果上下特征峰对齐良好，当前偏移 {args.offset:+.4f}s 即为正确值")
    print(f"  如果不对，手动尝试其他偏移值后重新运行，例如: --offset 2.5")
    print(f"  然后把正确的偏移值填入 alignment_report_v2.json 的 time_offset_sec")


if __name__ == '__main__':
    main()
