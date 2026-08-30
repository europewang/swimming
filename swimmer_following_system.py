from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from swim_video_sync import (
    PersonSegmenter,
    analyze_usable_segment,
    blend_frames,
    build_background_model,
    detect_video_roles,
    estimate_person_mask,
    global_offset_from_signatures,
    load_video_meta,
    refine_alignment_points,
    resize_to_max_width,
    sample_frame_at,
    smooth_offsets,
    build_signature_timeline,
)


@dataclass
class TargetObservation:
    time_sec: float
    source: str
    center_x: float
    center_y: float
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class FollowCommand:
    time_sec: float
    center_x: float
    center_y: float
    normalized_error: float
    steering: float
    forward_speed: float
    action: str
    source: str
    confidence: float


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def load_face_detector() -> cv2.CascadeClassifier | None:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    return detector if not detector.empty() else None


def detect_face_observation(
    frame: np.ndarray,
    person_mask: np.ndarray | None,
    detector: cv2.CascadeClassifier | None,
    time_sec: float,
) -> TargetObservation | None:
    if detector is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    roi = gray[: max(1, int(h * 0.55)), :]
    offset_x = 0
    offset_y = 0

    if person_mask is not None and np.count_nonzero(person_mask) > 0:
        bbox = mask_bbox(person_mask)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            head_bottom = min(h, max(y1 + 24, int(y1 + (y2 - y1) * 0.45)))
            if head_bottom > y1 and x2 > x1:
                roi = gray[y1:head_bottom, x1:x2]
                offset_x = x1
                offset_y = y1

    faces = detector.detectMultiScale(
        roi,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(24, 24),
    )
    if len(faces) == 0:
        return None

    areas = [fw * fh for (_, _, fw, fh) in faces]
    best_idx = int(np.argmax(areas))
    fx, fy, fw, fh = [int(v) for v in faces[best_idx]]
    full_bbox = (fx + offset_x, fy + offset_y, fx + offset_x + fw, fy + offset_y + fh)
    cx, cy = bbox_center(full_bbox)
    confidence = min(1.0, (fw * fh) / float(max(1, w * h)) * 40.0 + 0.35)
    return TargetObservation(
        time_sec=time_sec,
        source="face",
        center_x=cx,
        center_y=cy,
        bbox=full_bbox,
        confidence=float(confidence),
    )


def detect_mask_observation(
    mask: np.ndarray,
    time_sec: float,
    source: str,
    favor_upper_part: bool,
) -> TargetObservation | None:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    center_x = (x1 + x2) / 2.0
    if favor_upper_part:
        center_y = y1 + height * 0.22
    else:
        center_y = y1 + height * 0.62
    frame_area = float(mask.shape[0] * mask.shape[1])
    confidence = min(0.95, (width * height) / max(1.0, frame_area) * 8.0 + 0.2)
    return TargetObservation(
        time_sec=time_sec,
        source=source,
        center_x=float(center_x),
        center_y=float(center_y),
        bbox=bbox,
        confidence=float(confidence),
    )


def merge_observations(
    top_obs: TargetObservation | None,
    bottom_obs: TargetObservation | None,
    frame_shape: tuple[int, int, int],
    previous: TargetObservation | None,
) -> TargetObservation | None:
    candidates = [obs for obs in (top_obs, bottom_obs) if obs is not None]
    if not candidates:
        return previous
    if len(candidates) == 1:
        return candidates[0]

    h, _ = frame_shape[:2]
    total_weight = sum(obs.confidence for obs in candidates)
    cx = sum(obs.center_x * obs.confidence for obs in candidates) / max(total_weight, 1e-6)
    cy = sum(obs.center_y * obs.confidence for obs in candidates) / max(total_weight, 1e-6)
    x1 = min(obs.bbox[0] for obs in candidates)
    y1 = min(obs.bbox[1] for obs in candidates)
    x2 = max(obs.bbox[2] for obs in candidates)
    y2 = max(obs.bbox[3] for obs in candidates)
    if previous is not None:
        cx = 0.7 * previous.center_x + 0.3 * cx
        cy = 0.7 * previous.center_y + 0.3 * cy
    cy = float(np.clip(cy, 0, h - 1))
    return TargetObservation(
        time_sec=candidates[0].time_sec,
        source="+".join(obs.source for obs in candidates),
        center_x=float(cx),
        center_y=float(cy),
        bbox=(x1, y1, x2, y2),
        confidence=float(min(1.0, total_weight / len(candidates))),
    )


def build_follow_command(
    target: TargetObservation | None,
    frame_shape: tuple[int, int, int],
    cruise_speed: float,
    max_steering: float,
) -> FollowCommand | None:
    if target is None:
        return None
    h, w = frame_shape[:2]
    error_px = target.center_x - (w / 2.0)
    normalized_error = float(error_px / max(1.0, w / 2.0))
    steering = float(np.clip(normalized_error * 1.35, -max_steering, max_steering))

    abs_err = abs(normalized_error)
    if abs_err > 0.35:
        action = "turn_hard"
        speed = cruise_speed * 0.35
    elif abs_err > 0.18:
        action = "turn"
        speed = cruise_speed * 0.6
    else:
        action = "forward"
        speed = cruise_speed

    if target.confidence < 0.25:
        action = "search"
        speed = 0.0
        steering = 0.0

    return FollowCommand(
        time_sec=target.time_sec,
        center_x=float(target.center_x),
        center_y=float(target.center_y),
        normalized_error=normalized_error,
        steering=steering,
        forward_speed=float(speed),
        action=action,
        source=target.source,
        confidence=float(target.confidence),
    )


def draw_overlay(
    frame: np.ndarray,
    target: TargetObservation | None,
    command: FollowCommand | None,
) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    center_x = w // 2
    cv2.line(canvas, (center_x, 0), (center_x, h), (255, 255, 0), 2)
    cv2.line(canvas, (0, h // 2), (w, h // 2), (255, 255, 0), 1)

    if target is not None:
        x1, y1, x2, y2 = target.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(canvas, (int(target.center_x), int(target.center_y)), 6, (255, 80, 80), -1)
        cv2.putText(
            canvas,
            f"target={target.source} conf={target.confidence:.2f}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    if command is not None:
        cv2.putText(
            canvas,
            f"cmd={command.action} speed={command.forward_speed:.2f} steer={command.steering:.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"err={command.normalized_error:.3f}",
            (20, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    return canvas


def save_commands_csv(output_path: Path, commands: list[FollowCommand]) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_sec",
                "center_x",
                "center_y",
                "normalized_error",
                "steering",
                "forward_speed",
                "action",
                "source",
                "confidence",
            ],
        )
        writer.writeheader()
        for command in commands:
            writer.writerow(asdict(command))


def align_videos(top_video: Path, bottom_video: Path) -> tuple[np.ndarray, np.ndarray, object, object]:
    top_preprocess = analyze_usable_segment(top_video)
    bottom_preprocess = analyze_usable_segment(bottom_video)
    top_meta = load_video_meta(top_video)
    bottom_meta = load_video_meta(bottom_video)
    top_ts, top_sig = build_signature_timeline(top_video, sample_fps=6.0, underwater=False, preprocess=top_preprocess)
    bottom_ts, bottom_sig = build_signature_timeline(bottom_video, sample_fps=6.0, underwater=True, preprocess=bottom_preprocess)
    global_offset = global_offset_from_signatures(top_ts, top_sig, bottom_ts, bottom_sig, sample_fps=6.0)
    anchors = refine_alignment_points(
        ts_a=top_ts,
        sig_a=top_sig,
        ts_b=bottom_ts,
        sig_b=bottom_sig,
        base_offset_sec=global_offset,
        anchor_step_sec=4.0,
        window_sec=2.5,
        search_radius_sec=1.5,
        sample_fps=6.0,
    )
    total_duration = min(top_preprocess.usable_duration, bottom_preprocess.usable_duration)
    curve_times, curve_offsets = smooth_offsets(anchors, total_duration=total_duration)
    return curve_times, curve_offsets, top_preprocess, bottom_preprocess


def run_follow_system(
    top_video: Path,
    bottom_video: Path,
    output_dir: Path,
    output_fps: float,
    max_output_width: int,
    waterline_ratio: float,
    blend_px: int,
    cruise_speed: float,
    max_steering: float,
    segmentation_device: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_times, curve_offsets, top_preprocess, bottom_preprocess = align_videos(top_video, bottom_video)
    offset_interp = np.interp

    top_meta = load_video_meta(top_video)
    bottom_meta = load_video_meta(bottom_video)
    top_reader = iio.get_reader(str(top_video))
    bottom_reader = iio.get_reader(str(bottom_video))
    top_background = build_background_model(top_video, target_width=min(max_output_width, top_meta.width), preprocess=top_preprocess)
    bottom_background = build_background_model(bottom_video, target_width=min(max_output_width, bottom_meta.width), preprocess=bottom_preprocess)
    segmenter = PersonSegmenter(enabled=True, device=segmentation_device)
    face_detector = load_face_detector()

    overlap_start = max(0.0, -float(np.min(curve_offsets)))
    overlap_end = min(top_preprocess.usable_duration, bottom_preprocess.usable_duration - float(np.max(curve_offsets)))
    preview_path = output_dir / "follow_preview.mp4"
    writer = iio.get_writer(str(preview_path), fps=output_fps, codec="libx264", quality=8)

    current = overlap_start
    prev_top_frame = None
    prev_bottom_frame = None
    prev_top_mask = None
    prev_bottom_mask = None
    prev_target = None
    commands: list[FollowCommand] = []
    frame_idx = 0

    while current < overlap_end:
        bottom_time = current + float(offset_interp(current, curve_times, curve_offsets))
        top_frame = sample_frame_at(top_reader, current, top_meta.fps, preprocess=top_preprocess)
        bottom_frame = sample_frame_at(bottom_reader, bottom_time, bottom_meta.fps, preprocess=bottom_preprocess)
        if top_frame is None or bottom_frame is None:
            current += 1.0 / output_fps
            continue

        if top_frame.shape[:2] != bottom_frame.shape[:2]:
            bottom_frame = cv2.resize(bottom_frame, (top_frame.shape[1], top_frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        top_frame = resize_to_max_width(top_frame, max_output_width)
        bottom_frame = resize_to_max_width(bottom_frame, max_output_width)

        run_yolo = frame_idx % 2 == 0 or prev_top_mask is None or prev_bottom_mask is None
        top_mask = estimate_person_mask(
            top_frame,
            underwater=False,
            background=top_background,
            prev_frame=prev_top_frame,
            segmenter=segmenter if run_yolo else None,
            use_roi_segmentation=True,
            seed_mask=prev_top_mask if not run_yolo else None,
        )
        bottom_mask = estimate_person_mask(
            bottom_frame,
            underwater=True,
            background=bottom_background,
            prev_frame=prev_bottom_frame,
            segmenter=segmenter if run_yolo else None,
            use_roi_segmentation=True,
            seed_mask=prev_bottom_mask if not run_yolo else None,
        )

        fused = blend_frames(
            top_frame=top_frame,
            bottom_frame=bottom_frame,
            top_mask=top_mask,
            bottom_mask=bottom_mask,
            waterline_ratio=waterline_ratio,
            blend_px=blend_px,
        )

        top_face = detect_face_observation(top_frame, top_mask, face_detector, current)
        top_body = detect_mask_observation(top_mask, current, "top_body", favor_upper_part=True)
        bottom_body = detect_mask_observation(bottom_mask, current, "underwater_body", favor_upper_part=False)
        top_obs = top_face if top_face is not None else top_body
        target = merge_observations(top_obs, bottom_body, fused.shape, prev_target)
        command = build_follow_command(target, fused.shape, cruise_speed, max_steering)
        preview = draw_overlay(fused, target, command)
        writer.append_data(preview)

        if command is not None:
            commands.append(command)
        prev_target = target
        prev_top_frame = top_frame
        prev_bottom_frame = bottom_frame
        prev_top_mask = top_mask
        prev_bottom_mask = bottom_mask
        frame_idx += 1
        current += 1.0 / output_fps

    writer.close()
    top_reader.close()
    bottom_reader.close()

    command_csv = output_dir / "follow_commands.csv"
    save_commands_csv(command_csv, commands)

    report = {
        "top_video": str(top_video),
        "bottom_video": str(bottom_video),
        "preview_video": str(preview_path),
        "command_csv": str(command_csv),
        "command_count": len(commands),
        "overlap_start_sec": overlap_start,
        "overlap_end_sec": overlap_end,
        "waterline_ratio": waterline_ratio,
        "blend_px": blend_px,
        "cruise_speed": cruise_speed,
        "max_steering": max_steering,
        "segmentation_device": segmentation_device,
    }
    report_path = output_dir / "follow_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="游泳双机位拼接 + 池边小车跟随原型")
    parser.add_argument("--video-a", type=Path, help="输入视频 A（自动识别上下机位时使用）")
    parser.add_argument("--video-b", type=Path, help="输入视频 B（自动识别上下机位时使用）")
    parser.add_argument("--top-video", type=Path, help="显式指定水上视频")
    parser.add_argument("--bottom-video", type=Path, help="显式指定水下视频")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_follow"), help="输出目录")
    parser.add_argument("--output-fps", type=float, default=12.0, help="输出预览视频帧率")
    parser.add_argument("--max-output-width", type=int, default=1280, help="最大输出宽度")
    parser.add_argument("--waterline-ratio", type=float, default=0.5, help="默认水线比例")
    parser.add_argument("--blend-px", type=int, default=36, help="水线上下羽化像素")
    parser.add_argument("--cruise-speed", type=float, default=0.75, help="小车正常巡航速度")
    parser.add_argument("--max-steering", type=float, default=1.0, help="最大转向量")
    parser.add_argument("--segmentation-device", default="auto", choices=["auto", "cpu", "cuda:0"], help="YOLO 设备")
    args = parser.parse_args()

    explicit_mode = args.top_video is not None or args.bottom_video is not None
    auto_mode = args.video_a is not None or args.video_b is not None
    if explicit_mode and auto_mode:
        raise SystemExit("请二选一：要么用 --top-video/--bottom-video，要么用 --video-a/--video-b。")
    if explicit_mode:
        if args.top_video is None or args.bottom_video is None:
            raise SystemExit("显式模式下必须同时提供 --top-video 和 --bottom-video。")
        top_video = args.top_video
        bottom_video = args.bottom_video
        role_info: dict[str, object] = {
            "mode": "explicit",
            "top_path": str(top_video),
            "underwater_path": str(bottom_video),
        }
    else:
        if args.video_a is None or args.video_b is None:
            raise SystemExit("自动模式下必须同时提供 --video-a 和 --video-b。")
        preprocess_a = analyze_usable_segment(args.video_a)
        preprocess_b = analyze_usable_segment(args.video_b)
        role = detect_video_roles(args.video_a, args.video_b, preprocess_a=preprocess_a, preprocess_b=preprocess_b)
        top_video = Path(role.top_path)
        bottom_video = Path(role.underwater_path)
        role_info = asdict(role)

    report = run_follow_system(
        top_video=top_video,
        bottom_video=bottom_video,
        output_dir=args.output_dir,
        output_fps=args.output_fps,
        max_output_width=args.max_output_width,
        waterline_ratio=args.waterline_ratio,
        blend_px=args.blend_px,
        cruise_speed=args.cruise_speed,
        max_steering=args.max_steering,
        segmentation_device=args.segmentation_device,
    )
    report["role_detection"] = role_info
    report_path = Path(report["report_path"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
