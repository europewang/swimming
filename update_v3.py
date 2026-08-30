import sys
import re

file_path = r"d:\myflie\ALL_CODE\swimming\swim_video_v3.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace header comment
content = content.replace("Step 1: 视频配对与颠倒检测。", "Step 1: 接受手动指定的视频目录及旋转方向。")

# Remove lines 116 to 880 (from `# ============================================================` before `analyze_waterline_color_features` to before `def sky_water_separation_score`)
start_marker = "# ============================================================\n# Step 1: 颠倒检测 - 基于水位线视觉特征\n# ============================================================"
end_marker = "def sky_water_separation_score(frame: np.ndarray) -> float:"

if start_marker in content and end_marker in content:
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    content = content[:start_idx] + end_marker + content[end_idx + len(end_marker):]

# Modify run_full_pipeline signature and content
old_run_full = "def run_full_pipeline(top_video: Path, bottom_video: Path, output_dir: Path, output_fps: float = 30.0, max_width: int = 1280, force_top_rotated: str = None, force_bot_rotated: str = None) -> dict:"
new_run_full = "def run_full_pipeline(top_video: Path, bottom_video: Path, output_dir: Path, top_rotated: bool, bottom_rotated: bool, output_fps: float = 30.0, max_width: int = 1280) -> dict:"

if old_run_full in content:
    content = content.replace(old_run_full, new_run_full)

# Replace the rotation detection part in run_full_pipeline
old_detect_part = """    print("Step 1/4: 画面解析与身份辨别...")
    rot_top, rot_bottom, rotation_details = detect_rotations(top_video, bottom_video, debug_dir)
    
    if force_top_rotated is not None: rot_top = force_top_rotated == "true"
    if force_bot_rotated is not None: rot_bottom = force_bot_rotated == "true"
    
    # 打印详细的旋转判断依据
    print(f"\\n{'='*60}")
    print(f"  旋转检测详细分析")
    print(f"{'='*60}")
    print(f"\\n  【水上视频: {top_video.name}】")
    print(f"  {rotation_details['top_video']['explanation']}")
    print(f"\\n  【水下视频: {bottom_video.name}】")
    print(f"  {rotation_details['bottom_video']['explanation']}")
    
    print(f"\\n  最终决定:")
    print(f"    水上视频 ({top_video.name}): {'翻转180°' if rot_top else '正常方向'}")
    print(f"    水下视频 ({bottom_video.name}): {'翻转180°' if rot_bottom else '正常方向'}")
    print(f"{'='*60}\\n")"""

new_detect_part = """    print("Step 1/4: 使用用户指定的视频方向...")
    rot_top = top_rotated
    rot_bottom = bottom_rotated
    
    print(f"\\n{'='*60}")
    print(f"  最终决定:")
    print(f"    水上视频 ({top_video.name}): {'翻转180°' if rot_top else '正常方向'}")
    print(f"    水下视频 ({bottom_video.name}): {'翻转180°' if rot_bottom else '正常方向'}")
    print(f"{'='*60}\\n")"""

content = content.replace(old_detect_part, new_detect_part)

# Also fix the report dictionary in run_full_pipeline
old_report_dict = """        "rotation_detection": {
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
        }"""
new_report_dict = """        "rotation_detection": {
            "top_rotated": rot_top,
            "bottom_rotated": rot_bottom
        }"""
content = content.replace(old_report_dict, new_report_dict)

# Modify run_batch_pipeline signature
old_batch = "def run_batch_pipeline(top_dir: Path, bottom_dir: Path, output_dir: Path, output_fps: float = 30.0, max_width: int = 1280) -> list[dict]:"
new_batch = "def run_batch_pipeline(top_dir: Path, bottom_dir: Path, output_dir: Path, top_rotated: bool, bottom_rotated: bool, output_fps: float = 30.0, max_width: int = 1280) -> list[dict]:"
content = content.replace(old_batch, new_batch)

# Update run_full_pipeline call in run_batch_pipeline
old_call = "report = run_full_pipeline(top_video, bottom_video, pair_out, output_fps, max_width)"
new_call = "report = run_full_pipeline(top_video, bottom_video, pair_out, top_rotated, bottom_rotated, output_fps, max_width)"
content = content.replace(old_call, new_call)

# Now, we also need to address how the user wants to specify folders and match videos.
# The user said: "我来指定哪个文件夹里的所有视频是水上还是水下" - this is already handled by --batch-top and --batch-bottom.
# The matching logic is currently by DJI timestamp. If they want to simplify it further, maybe just match by filename or sequentially?
# The prompt says "我可以去掉，由我来指定哪个文件夹里的所有视频是水上还是水下，并且水上视频是正还是反了，水下视频是正还是反。"
# The current pairing logic is DJI specific. If they just have generic video1, video2, maybe just sort and zip? Let's simplify the pairing to just match by alphabetical order!
# Let's replace the pairing logic in run_batch_pipeline.
old_pairing = """    # DJI 文件名格式: DJI_<录制时间戳14位>_<剪辑序号4位>_D[_min].mp4
    # 例如: DJI_20260508195343_0001_D_min.mp4
    clip_pattern = re.compile(r'^DJI_(\\d{14})_\\d+_D(_min)?\\.mp4$', re.IGNORECASE)

    def parse_dji_timestamp(ts_str: str) -> int:
        \"\"\"将 DJI 时间戳字符串转换为 Unix 时间戳（秒）\"\"\"
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

        pair_out = output_dir / f"{top_video.stem}__{bottom_video.stem}\"\"\"

new_pairing = """    results, paired, skipped, failed_pairs = [], 0, 0, []
    
    # 简化配对逻辑：按文件名排序，一一对应配对
    for top_video, bottom_video in zip(top_files, bot_files):
        pair_out = output_dir / f"{top_video.stem}__{bottom_video.stem}\"\"\"

content = content.replace(old_pairing, new_pairing)

# Also remove the checking logic for missing pairs
old_missing_checks = """    # 检查有但没配对上的视频
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
                missing_pairs.append((bot_video, "未配对成功（top 视频不足）"))"""

content = content.replace(old_missing_checks, "")

# And update the summary prints
old_summary = """    print(f"\\n{'='*60}")
    print(f"  批处理完成汇总")
    print(f"{'='*60}")
    print(f"  ✅ 已融合: {paired}")
    print(f"  ⏭️  已跳过: {skipped} (含 {len(failed_pairs)} 个失败)")
    print(f"  ❌ 缺失配对: {len(missing_pairs)}")

    if missing_pairs:
        print(f"\\n  --- 缺失配对 ---")
        for video, reason in missing_pairs:
            print(f"    ❌ {video.name} → {reason}")

    if failed_pairs:
        print(f"\\n  --- 融合失败 ---")
        for top_v, bot_v, err in failed_pairs:
            print(f"    ❌ {top_v.name} x {bot_v.name}")
            print(f"       原因: {err}")

    if not missing_pairs and not failed_pairs:
        print(f"\\n  ✅ 所有视频均已正确配对并融合，无问题")"""

new_summary = """    print(f"\\n{'='*60}")
    print(f"  批处理完成汇总")
    print(f"{'='*60}")
    print(f"  ✅ 已融合: {paired}")
    print(f"  ⏭️  已跳过: {skipped} (含 {len(failed_pairs)} 个失败)")
    
    unpaired_top = len(top_files) - len(bot_files)
    if unpaired_top > 0:
        print(f"  ❌ 缺失配对: {unpaired_top} 个水上视频没有对应的水下视频")
    elif unpaired_top < 0:
        print(f"  ❌ 缺失配对: {-unpaired_top} 个水下视频没有对应的水上视频")

    if failed_pairs:
        print(f"\\n  --- 融合失败 ---")
        for top_v, bot_v, err in failed_pairs:
            print(f"    ❌ {top_v.name} x {bot_v.name}")
            print(f"       原因: {err}")

    if unpaired_top == 0 and not failed_pairs:
        print(f"\\n  ✅ 所有视频均已正确配对并融合，无问题")"""

content = content.replace(old_summary, new_summary)


# Update parse_args and main execution
old_main = """def parse_args():
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
                          force_top_rotated=args.force_top_rotated, force_bot_rotated=args.force_bot_rotated)"""

new_main = """def parse_args():
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
        run_full_pipeline(args.video_a, args.video_b, args.output_dir, args.top_rotated, args.bottom_rotated, output_fps=args.fps, max_width=args.max_width)"""

content = content.replace(old_main, new_main)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated successfully")
