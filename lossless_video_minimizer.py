from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
import re

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".m2ts"}

def parse_args():
    parser = argparse.ArgumentParser(description="纯净极速视频压缩")
    parser.add_argument("input_dir", type=Path, help="视频文件夹")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg", type=Path)
    return parser.parse_args()

def run_ffmpeg_with_progress(cmd):
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="ignore"
    )
    time_pat = re.compile(r"time=(\d+):(\d+):(\d+)\.")
    while proc.poll() is None:
        line = proc.stderr.readline()
        if line:
            print(f"\r    编码中...", end="", flush=True)
    print()
    if proc.returncode != 0:
        raise RuntimeError("压缩失败")

def format_bytes(size):
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

def format_duration(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}分{s}秒" if m else f"{s}秒"

def resolve_ffmpeg(manual):
    if manual and manual.exists():
        return manual
    if shutil.which("ffmpeg"):
        return Path(shutil.which("ffmpeg"))
    if imageio_ffmpeg:
        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    raise RuntimeError("未找到 ffmpeg")

def iter_video_files(root, recursive, exts):
    it = root.rglob("*") if recursive else root.glob("*")
    for p in it:
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def prepare_output(src, input_dir, output_dir, overwrite):
    rel = src.parent.relative_to(input_dir)
    out_dir = output_dir / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_min.mp4"
    if dst.exists() and not overwrite:
        raise RuntimeError("文件已存在，加 --overwrite 覆盖")
    return dst

def main():
    total_start = time.time()
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir or input_dir.parent / f"{input_dir.name}_min"
    output_dir.mkdir(exist_ok=True)
    ffmpeg_exe = resolve_ffmpeg(args.ffmpeg)
    videos = list(iter_video_files(input_dir, args.recursive, VIDEO_EXTENSIONS))
    total = len(videos)

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"视频数量: {total}")
    print(f"编码: H.264 速度: ultrafast 压缩率: CRF30")
    print("="*50)

    for idx, src in enumerate(videos, 1):
        print(f"\n[{idx}/{total}] 处理: {src.name}")
        try:
            dst = prepare_output(src, input_dir, output_dir, args.overwrite)
            orig_size = src.stat().st_size
            t0 = time.time()

            cmd = [
                str(ffmpeg_exe), "-y", "-hide_banner",
                "-i", str(src),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "30",
                "-c:a", "aac",      # 音频压缩，更快更小
                "-b:a", "96k",
                "-tag:v", "avc1"
            ]
            cmd.append(str(dst))
            run_ffmpeg_with_progress(cmd)

            new_size = dst.stat().st_size
            print(f"  原大小: {format_bytes(orig_size)}")
            print(f"  新大小: {format_bytes(new_size)}")
            print(f"  耗时: {format_duration(time.time() - t0)}")

        except Exception as e:
            print(f"  失败: {e}")

    print("\n" + "="*50)
    print(f"✅ 全部完成！总耗时：{format_duration(time.time() - total_start)}")
    print("="*50)

if __name__ == "__main__":
    main()