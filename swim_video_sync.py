from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import imageio.v2 as iio
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate, savgol_filter

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


@dataclass
class VideoMeta:
    path: Path
    fps: float
    width: int
    height: int
    duration: float
    frame_count: int | None


@dataclass
class AlignmentPoint:
    time_sec: float
    offset_sec: float
    score: float


@dataclass
class RoleDetection:
    video_a_path: str
    video_b_path: str
    underwater_path: str
    top_path: str
    score_a: float
    score_b: float
    reason: str


class PersonSegmenter:
    def __init__(self, enabled: bool = True, model_name: str = "yolov8n-seg.pt") -> None:
        self.enabled = enabled and YOLO is not None
        self.model_name = model_name
        self._model = None
        self.available = False
        if self.enabled:
            try:
                self._model = YOLO(model_name)
                self.available = True
            except Exception:
                self.available = False

    def segment_person(self, frame: np.ndarray) -> np.ndarray | None:
        if not self.available or self._model is None:
            return None
        try:
            result = self._model.predict(
                source=frame,
                verbose=False,
                conf=0.15,
                iou=0.45,
                retina_masks=False,
                imgsz=640,
                max_det=5,
                device="cpu",
            )[0]
        except Exception:
            return None

        if result.masks is None or result.boxes is None:
            return None

        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()
        masks = result.masks.data.detach().cpu().numpy()
        best_idx = None
        best_score = -1.0
        for idx, class_id in enumerate(classes):
            if class_id != 0:
                continue
            area = float(masks[idx].sum())
            score = area * float(confidences[idx])
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            return None
        mask = (masks[best_idx] > 0.35).astype(np.uint8) * 255
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask


def load_video_meta(path: Path) -> VideoMeta:
    reader = iio.get_reader(str(path))
    meta = reader.get_meta_data()
    reader.close()
    fps = float(meta.get("fps", 30.0))
    width, height = meta.get("size", (0, 0))
    duration = float(meta.get("duration", 0.0))
    raw_frames = meta.get("nframes")
    frame_count = None if raw_frames in (None, float("inf")) else int(raw_frames)
    if frame_count is None and duration > 0 and fps > 0:
        frame_count = int(round(duration * fps))
    return VideoMeta(
        path=path,
        fps=fps,
        width=int(width),
        height=int(height),
        duration=duration,
        frame_count=frame_count,
    )


def evenly_spaced_frames(path: Path, max_samples: int = 12, max_seconds: float = 3.0) -> list[np.ndarray]:
    reader = iio.get_reader(str(path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    duration = float(meta.get("duration", 0.0))
    frame_total = int(max(1, round(min(duration, max_seconds) * fps)))
    sample_count = min(max_samples, frame_total)
    indices = np.linspace(0, max(0, frame_total - 1), num=sample_count, dtype=int)
    frames: list[np.ndarray] = []
    for frame_idx in indices:
        try:
            frames.append(reader.get_data(int(frame_idx)))
        except Exception:
            continue
    reader.close()
    return frames


def underwater_likelihood(path: Path) -> tuple[float, dict[str, float]]:
    frames = evenly_spaced_frames(path)
    if not frames:
        raise ValueError(f"无法采样视频内容: {path}")

    blue_scores = []
    top_blue_scores = []
    top_warm_scores = []
    edge_densities = []
    sharpness_scores = []

    for frame in frames:
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        small_f = small.astype(np.float32) / 255.0
        r = small_f[:, :, 0]
        g = small_f[:, :, 1]
        # imageio delivers RGB arrays
        b = small_f[:, :, 2]
        top = small_f[:90]
        top_r = top[:, :, 0]
        top_g = top[:, :, 1]
        top_b = top[:, :, 2]

        blue_scores.append(float(np.mean(b - 0.5 * (r + g))))
        top_blue_scores.append(float(np.mean(top_b - 0.5 * (top_r + top_g))))
        top_warm_scores.append(float(np.mean(top_r - top_b)))

        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edge_densities.append(float(np.mean(edges > 0)))
        sharpness_scores.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))

    metrics = {
        "blue_dominance": float(np.mean(blue_scores)),
        "top_blue_dominance": float(np.mean(top_blue_scores)),
        "top_warm_dominance": float(np.mean(top_warm_scores)),
        "edge_density": float(np.mean(edge_densities)),
        "sharpness": float(np.mean(sharpness_scores)),
    }
    score = (
        metrics["blue_dominance"] * 7.0
        + metrics["top_blue_dominance"] * 5.0
        - metrics["top_warm_dominance"] * 6.0
        - metrics["edge_density"] * 1.5
        - metrics["sharpness"] / 1200.0
    )
    return float(score), metrics


def detect_video_roles(video_a: Path, video_b: Path) -> RoleDetection:
    score_a, metrics_a = underwater_likelihood(video_a)
    score_b, metrics_b = underwater_likelihood(video_b)
    if score_a >= score_b:
        underwater_path = video_a
        top_path = video_b
    else:
        underwater_path = video_b
        top_path = video_a
    reason = (
        "更高的水下分数通常意味着整幅画面更偏蓝、上半区更像水体反射、纹理更柔和、边缘更少。"
        f" A={score_a:.3f}, B={score_b:.3f}。"
        f" A指标={json.dumps(metrics_a, ensure_ascii=False)};"
        f" B指标={json.dumps(metrics_b, ensure_ascii=False)}"
    )
    return RoleDetection(
        video_a_path=str(video_a),
        video_b_path=str(video_b),
        underwater_path=str(underwater_path),
        top_path=str(top_path),
        score_a=float(score_a),
        score_b=float(score_b),
        reason=reason,
    )


def iter_sampled_frames(
    path: Path,
    sample_fps: float,
    max_frames: int | None = None,
) -> Iterable[tuple[float, np.ndarray]]:
    reader = iio.get_reader(str(path))
    meta = reader.get_meta_data()
    native_fps = float(meta.get("fps", 30.0))
    step = max(int(round(native_fps / sample_fps)), 1)
    yielded = 0
    for frame_idx, frame in enumerate(reader):
        if frame_idx % step != 0:
            continue
        time_sec = frame_idx / native_fps
        yield time_sec, frame
        yielded += 1
        if max_frames is not None and yielded >= max_frames:
            break
    reader.close()


def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    if frame.shape[1] == target_width:
        return frame
    scale = target_width / frame.shape[1]
    target_height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def normalize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    resized = resize_frame(frame, target_width=target_width)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def motion_descriptor(
    gray: np.ndarray,
    prev_gray: np.ndarray | None,
    underwater: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if prev_gray is None:
        diff = np.zeros_like(gray)
    else:
        diff = cv2.absdiff(gray, prev_gray)
    if underwater:
        # Underwater reflections often create a mirrored body in the upper half.
        h = diff.shape[0]
        upper = diff[: h // 2]
        lower = diff[h // 2 :]
        upper = (upper * 0.45).astype(np.uint8)
        diff = np.vstack([upper, lower])
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return diff, mask


def keep_primary_blob(mask: np.ndarray, prefer_lower_half: bool) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    h = mask.shape[0]
    best_label = 0
    best_score = -1.0
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 40:
            continue
        cy = float(centroids[label][1])
        lower_bonus = 1.35 if prefer_lower_half and cy > h * 0.5 else 1.0
        score = area * lower_bonus
        if score > best_score:
            best_score = score
            best_label = label
    if best_label == 0:
        return np.zeros_like(mask)
    output = np.zeros_like(mask)
    output[labels == best_label] = 255
    return output


def frame_signature(frame: np.ndarray, prev_gray: np.ndarray | None, underwater: bool) -> tuple[np.ndarray, np.ndarray]:
    gray = normalize_frame(frame, target_width=320)
    _, motion_mask = motion_descriptor(gray, prev_gray, underwater=underwater)
    primary = keep_primary_blob(motion_mask, prefer_lower_half=underwater)
    x_profile = primary.sum(axis=0).astype(np.float32)
    y_profile = primary.sum(axis=1).astype(np.float32)
    energy = float(primary.mean()) / 255.0
    moments = cv2.moments(primary)
    cx = 0.5
    cy = 0.5
    if moments["m00"] > 0:
        cx = float(moments["m10"] / moments["m00"]) / primary.shape[1]
        cy = float(moments["m01"] / moments["m00"]) / primary.shape[0]
    features = np.concatenate(
        [
            cv2.resize(x_profile[None, :], (32, 1), interpolation=cv2.INTER_AREA).ravel(),
            cv2.resize(y_profile[:, None], (1, 24), interpolation=cv2.INTER_AREA).ravel(),
            np.array([energy, cx, cy], dtype=np.float32),
        ]
    )
    norm = np.linalg.norm(features)
    if norm > 1e-6:
        features = features / norm
    return features, gray


def build_signature_timeline(path: Path, sample_fps: float, underwater: bool) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    features: list[np.ndarray] = []
    prev_gray = None
    for time_sec, frame in iter_sampled_frames(path, sample_fps=sample_fps):
        signature, prev_gray = frame_signature(frame, prev_gray=prev_gray, underwater=underwater)
        timestamps.append(time_sec)
        features.append(signature)
    if not features:
        raise ValueError(f"无法从视频读取采样帧: {path}")
    return np.array(timestamps, dtype=np.float32), np.stack(features).astype(np.float32)


def global_offset_from_signatures(
    ts_a: np.ndarray,
    sig_a: np.ndarray,
    ts_b: np.ndarray,
    sig_b: np.ndarray,
    sample_fps: float,
) -> float:
    signal_a = sig_a[:, -3]
    signal_b = sig_b[:, -3]
    signal_a = signal_a - signal_a.mean()
    signal_b = signal_b - signal_b.mean()
    corr = correlate(signal_b, signal_a, mode="full")
    lag = int(np.argmax(corr) - (len(signal_a) - 1))
    return lag / sample_fps


def local_window_score(
    window_a: np.ndarray,
    window_b: np.ndarray,
) -> float:
    if len(window_a) == 0 or len(window_b) == 0:
        return -1e9
    size = min(len(window_a), len(window_b))
    if size < 3:
        return -1e9
    a = window_a[:size]
    b = window_b[:size]
    sims = np.sum(a * b, axis=1)
    return float(np.mean(sims))


def refine_alignment_points(
    ts_a: np.ndarray,
    sig_a: np.ndarray,
    ts_b: np.ndarray,
    sig_b: np.ndarray,
    base_offset_sec: float,
    anchor_step_sec: float,
    window_sec: float,
    search_radius_sec: float,
    sample_fps: float,
) -> list[AlignmentPoint]:
    anchors: list[AlignmentPoint] = []
    max_time = min(ts_a[-1], ts_b[-1] - base_offset_sec)
    current = window_sec
    while current < max_time - window_sec:
        center_idx_a = int(round(current * sample_fps))
        half_window = int(round(window_sec * sample_fps / 2))
        start_a = max(0, center_idx_a - half_window)
        end_a = min(len(sig_a), center_idx_a + half_window)
        window_a = sig_a[start_a:end_a]
        best_offset = base_offset_sec
        best_score = -1e9
        for candidate in np.linspace(
            base_offset_sec - search_radius_sec,
            base_offset_sec + search_radius_sec,
            num=25,
        ):
            center_b = current + candidate
            center_idx_b = int(round(center_b * sample_fps))
            start_b = max(0, center_idx_b - half_window)
            end_b = min(len(sig_b), center_idx_b + half_window)
            window_b = sig_b[start_b:end_b]
            score = local_window_score(window_a, window_b)
            if score > best_score:
                best_score = score
                best_offset = float(candidate)
        anchors.append(AlignmentPoint(time_sec=float(current), offset_sec=best_offset, score=best_score))
        current += anchor_step_sec
    if not anchors:
        anchors.append(AlignmentPoint(time_sec=0.0, offset_sec=base_offset_sec, score=0.0))
    return anchors


def smooth_offsets(points: list[AlignmentPoint], total_duration: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.array([p.time_sec for p in points], dtype=np.float32)
    offsets = np.array([p.offset_sec for p in points], dtype=np.float32)
    if len(offsets) >= 5:
        window = min(len(offsets) if len(offsets) % 2 == 1 else len(offsets) - 1, 7)
        if window >= 5:
            offsets = savgol_filter(offsets, window_length=window, polyorder=2, mode="interp")
    if len(times) == 1:
        times = np.array([0.0, max(total_duration, 0.1)], dtype=np.float32)
        offsets = np.array([offsets[0], offsets[0]], dtype=np.float32)
    else:
        times = np.concatenate([[0.0], times, [total_duration]])
        offsets = np.concatenate([[offsets[0]], offsets, [offsets[-1]]])
    return times, offsets


def sample_frame_at(reader, time_sec: float, fps: float) -> np.ndarray | None:
    if time_sec < 0:
        return None
    frame_idx = int(round(time_sec * fps))
    try:
        return reader.get_data(frame_idx)
    except Exception:
        return None


def build_background_model(path: Path, target_width: int, max_samples: int = 21) -> np.ndarray:
    reader = iio.get_reader(str(path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    duration = float(meta.get("duration", 0.0))
    total_frames = max(1, int(round(duration * fps))) if duration > 0 else max_samples
    sample_indices = np.linspace(0, max(0, total_frames - 1), num=min(max_samples, total_frames), dtype=int)
    frames: list[np.ndarray] = []
    for idx in sample_indices:
        try:
            frame = reader.get_data(int(idx))
        except Exception:
            continue
        frames.append(resize_frame(frame, target_width=target_width))
    reader.close()
    if not frames:
        raise ValueError(f"无法建立背景模型: {path}")
    stack = np.stack(frames).astype(np.float32)
    return np.median(stack, axis=0).astype(np.uint8)


def estimate_person_mask(
    frame: np.ndarray,
    underwater: bool,
    background: np.ndarray | None = None,
    prev_frame: np.ndarray | None = None,
    segmenter: PersonSegmenter | None = None,
) -> np.ndarray:
    yolo_mask = None
    if segmenter is not None:
        yolo_mask = segmenter.segment_person(frame)
        if yolo_mask is not None and np.count_nonzero(yolo_mask) > 500:
            kernel = np.ones((7, 7), np.uint8)
            yolo_mask = cv2.morphologyEx(yolo_mask, cv2.MORPH_CLOSE, kernel)
            yolo_mask = keep_primary_blob(yolo_mask, prefer_lower_half=underwater)

    small = resize_frame(frame, target_width=320)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    bg_mask = np.zeros(gray.shape, dtype=np.uint8)
    if background is not None:
        bg_small = resize_frame(background, target_width=small.shape[1])
        diff_rgb = cv2.absdiff(small, bg_small)
        diff_gray = cv2.cvtColor(diff_rgb, cv2.COLOR_RGB2GRAY)
        color_strength = diff_rgb.max(axis=2).astype(np.uint8)
        combined = cv2.addWeighted(diff_gray, 0.7, color_strength, 0.3, 0.0)
        thresh = 22 if underwater else 18
        _, bg_mask = cv2.threshold(combined, thresh, 255, cv2.THRESH_BINARY)

    motion_mask = np.zeros(gray.shape, dtype=np.uint8)
    if prev_frame is not None:
        prev_small = resize_frame(prev_frame, target_width=small.shape[1])
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY)
        motion = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(motion, 16 if underwater else 12, 255, cv2.THRESH_BINARY)

    mask = bg_mask if background is not None else motion_mask
    if background is not None and prev_frame is not None:
        mask = cv2.bitwise_or(bg_mask, motion_mask)

    if underwater:
        h = mask.shape[0]
        mask[: int(h * 0.42)] = (mask[: int(h * 0.42)] * 0.35).astype(np.uint8)

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = keep_primary_blob(mask, prefer_lower_half=underwater)

    if np.count_nonzero(mask) > 0:
        # Expand around the selected body blob so arms/legs don't get clipped too aggressively.
        dilate_kernel = np.ones((11, 11), np.uint8)
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    full = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    if yolo_mask is not None and np.count_nonzero(yolo_mask) > 0:
        ys, xs = np.where(yolo_mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            pad_x = max(30, int((x2 - x1 + 1) * 0.35))
            pad_y = max(30, int((y2 - y1 + 1) * 0.45))
            rx1 = max(0, x1 - pad_x)
            rx2 = min(frame.shape[1], x2 + pad_x)
            ry1 = max(0, y1 - pad_y)
            ry2 = min(frame.shape[0], y2 + pad_y)
            roi = np.zeros_like(full)
            roi[ry1:ry2, rx1:rx2] = 255
            constrained_full = cv2.bitwise_and(full, roi)
            merged = cv2.bitwise_or(yolo_mask, constrained_full)
            merged = keep_primary_blob(merged, prefer_lower_half=underwater)
            if np.count_nonzero(merged) > 0:
                dilate_kernel = np.ones((15, 15), np.uint8)
                merged = cv2.dilate(merged, dilate_kernel, iterations=1)
                return merged

    return full


def blend_frames(
    top_frame: np.ndarray,
    bottom_frame: np.ndarray,
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    waterline_ratio: float,
    blend_px: int,
) -> np.ndarray:
    h, w = top_frame.shape[:2]
    seam = int(h * waterline_ratio)
    seam = max(0, min(h, seam))
    output = bottom_frame.copy()
    output[:seam] = top_frame[:seam]

    alpha = np.zeros((h, w), dtype=np.float32)
    top_zone_start = max(0, seam - blend_px)
    top_zone_end = min(h, seam + blend_px)
    if top_zone_end > top_zone_start:
        grad = np.linspace(1.0, 0.0, top_zone_end - top_zone_start, dtype=np.float32)
        alpha[top_zone_start:top_zone_end, :] = grad[:, None]
    alpha[:top_zone_start, :] = 1.0
    output = (top_frame * alpha[..., None] + bottom_frame * (1.0 - alpha[..., None])).astype(np.uint8)

    top_person = cv2.bitwise_and(top_frame, top_frame, mask=top_mask)
    bottom_person = cv2.bitwise_and(bottom_frame, bottom_frame, mask=bottom_mask)
    person_mix = output.copy()

    top_region = np.zeros_like(top_mask)
    top_region[: seam + blend_px] = 255
    bottom_region = np.zeros_like(bottom_mask)
    bottom_region[max(0, seam - blend_px) :] = 255

    top_person_mask = cv2.bitwise_and(top_mask, top_region)
    bottom_person_mask = cv2.bitwise_and(bottom_mask, bottom_region)

    person_mix[top_person_mask > 0] = top_person[top_person_mask > 0]
    person_mix[bottom_person_mask > 0] = bottom_person[bottom_person_mask > 0]
    return person_mix


def resize_to_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    target_height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (max_width, target_height), interpolation=cv2.INTER_AREA)


def write_fused_video(
    top_path: Path,
    bottom_path: Path,
    top_meta: VideoMeta,
    bottom_meta: VideoMeta,
    output_path: Path,
    curve_times: np.ndarray,
    curve_offsets: np.ndarray,
    waterline_ratio: float,
    blend_px: int,
    output_fps: float,
    max_output_width: int,
    use_person_segmentation: bool,
) -> None:
    interp = interp1d(curve_times, curve_offsets, kind="linear", fill_value="extrapolate")
    top_reader = iio.get_reader(str(top_path))
    bottom_reader = iio.get_reader(str(bottom_path))
    top_background = build_background_model(top_path, target_width=min(max_output_width, top_meta.width))
    bottom_background = build_background_model(bottom_path, target_width=min(max_output_width, bottom_meta.width))
    segmenter = PersonSegmenter(enabled=use_person_segmentation)
    overlap_start = max(0.0, -float(curve_offsets.min()))
    overlap_end = min(top_meta.duration, bottom_meta.duration - float(curve_offsets.max()))
    if overlap_end <= overlap_start:
        raise ValueError("两个视频没有有效重叠区间，无法输出融合视频。")
    writer = iio.get_writer(str(output_path), fps=output_fps, codec="libx264", quality=8)
    current = overlap_start
    prev_top_frame = None
    prev_bottom_frame = None
    while current < overlap_end:
        bottom_time = current + float(interp(current))
        top_frame = sample_frame_at(top_reader, current, top_meta.fps)
        bottom_frame = sample_frame_at(bottom_reader, bottom_time, bottom_meta.fps)
        if top_frame is None or bottom_frame is None:
            current += 1.0 / output_fps
            continue
        if top_frame.shape[:2] != bottom_frame.shape[:2]:
            bottom_frame = cv2.resize(
                bottom_frame,
                (top_frame.shape[1], top_frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        top_frame = resize_to_max_width(top_frame, max_width=max_output_width)
        bottom_frame = resize_to_max_width(bottom_frame, max_width=max_output_width)
        top_mask = estimate_person_mask(
            top_frame,
            underwater=False,
            background=top_background,
            prev_frame=prev_top_frame,
            segmenter=segmenter,
        )
        bottom_mask = estimate_person_mask(
            bottom_frame,
            underwater=True,
            background=bottom_background,
            prev_frame=prev_bottom_frame,
            segmenter=segmenter,
        )
        fused = blend_frames(
            top_frame=top_frame,
            bottom_frame=bottom_frame,
            top_mask=top_mask,
            bottom_mask=bottom_mask,
            waterline_ratio=waterline_ratio,
            blend_px=blend_px,
        )
        writer.append_data(fused)
        prev_top_frame = top_frame
        prev_bottom_frame = bottom_frame
        current += 1.0 / output_fps
    writer.close()
    top_reader.close()
    bottom_reader.close()


def save_alignment_report(
    output_path: Path,
    top_meta: VideoMeta,
    bottom_meta: VideoMeta,
    global_offset: float,
    anchors: list[AlignmentPoint],
    curve_times: np.ndarray,
    curve_offsets: np.ndarray,
    role_detection: RoleDetection | None = None,
) -> None:
    top_data = asdict(top_meta)
    bottom_data = asdict(bottom_meta)
    top_data["path"] = str(top_meta.path)
    bottom_data["path"] = str(bottom_meta.path)
    data = {
        "top_video": top_data,
        "bottom_video": bottom_data,
        "global_offset_sec": global_offset,
        "anchors": [asdict(point) for point in anchors],
        "curve_times_sec": curve_times.tolist(),
        "curve_offsets_sec": curve_offsets.tolist(),
    }
    if role_detection is not None:
        data["role_detection"] = asdict(role_detection)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对齐水上/水下游泳视频，并输出平滑偏移曲线与融合视频。"
    )
    parser.add_argument("--top-video", type=Path, help="显式指定水上视角视频路径")
    parser.add_argument("--bottom-video", type=Path, help="显式指定水下视角视频路径")
    parser.add_argument("--video-a", type=Path, help="未区分机位的第一个视频，脚本会自动判断")
    parser.add_argument("--video-b", type=Path, help="未区分机位的第二个视频，脚本会自动判断")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="输出目录")
    parser.add_argument("--sample-fps", type=float, default=6.0, help="对齐分析采样帧率")
    parser.add_argument("--anchor-step-sec", type=float, default=5.0, help="局部微调锚点间隔")
    parser.add_argument("--window-sec", type=float, default=2.0, help="局部对齐窗口长度")
    parser.add_argument("--search-radius-sec", type=float, default=1.2, help="局部搜索半径")
    parser.add_argument("--waterline-ratio", type=float, default=0.44, help="拼接水线高度比例")
    parser.add_argument("--blend-px", type=int, default=90, help="水线上下融合像素宽度")
    parser.add_argument("--output-fps", type=float, default=30.0, help="输出视频帧率")
    parser.add_argument("--max-output-width", type=int, default=1280, help="融合输出最大宽度，默认会缩小 4K 视频以加速")
    parser.add_argument("--disable-person-segmentation", action="store_true", help="禁用 YOLO 人体分割，仅使用传统图像法")
    parser.add_argument(
        "--skip-fuse",
        action="store_true",
        help="只生成时间对齐报告，不输出融合视频",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    role_detection = None
    if args.video_a and args.video_b:
        role_detection = detect_video_roles(args.video_a, args.video_b)
        top_video = Path(role_detection.top_path)
        bottom_video = Path(role_detection.underwater_path)
    elif args.top_video and args.bottom_video:
        top_video = args.top_video
        bottom_video = args.bottom_video
    else:
        raise ValueError("请提供 --video-a/--video-b 让脚本自动识别，或提供 --top-video/--bottom-video 显式指定。")

    top_meta = load_video_meta(top_video)
    bottom_meta = load_video_meta(bottom_video)

    top_ts, top_sig = build_signature_timeline(top_video, sample_fps=args.sample_fps, underwater=False)
    bottom_ts, bottom_sig = build_signature_timeline(bottom_video, sample_fps=args.sample_fps, underwater=True)

    global_offset = global_offset_from_signatures(
        ts_a=top_ts,
        sig_a=top_sig,
        ts_b=bottom_ts,
        sig_b=bottom_sig,
        sample_fps=args.sample_fps,
    )
    anchors = refine_alignment_points(
        ts_a=top_ts,
        sig_a=top_sig,
        ts_b=bottom_ts,
        sig_b=bottom_sig,
        base_offset_sec=global_offset,
        anchor_step_sec=args.anchor_step_sec,
        window_sec=args.window_sec,
        search_radius_sec=args.search_radius_sec,
        sample_fps=args.sample_fps,
    )
    total_duration = min(top_meta.duration, bottom_meta.duration)
    curve_times, curve_offsets = smooth_offsets(anchors, total_duration=total_duration)

    report_path = args.output_dir / "alignment_report.json"
    save_alignment_report(
        output_path=report_path,
        top_meta=top_meta,
        bottom_meta=bottom_meta,
        global_offset=global_offset,
        anchors=anchors,
        curve_times=curve_times,
        curve_offsets=curve_offsets,
        role_detection=role_detection,
    )

    if not args.skip_fuse:
        output_video_path = args.output_dir / "fused_swim.mp4"
        write_fused_video(
            top_path=top_video,
            bottom_path=bottom_video,
            top_meta=top_meta,
            bottom_meta=bottom_meta,
            output_path=output_video_path,
            curve_times=curve_times,
            curve_offsets=curve_offsets,
            waterline_ratio=args.waterline_ratio,
            blend_px=args.blend_px,
            output_fps=args.output_fps,
            max_output_width=args.max_output_width,
            use_person_segmentation=not args.disable_person_segmentation,
        )

    print(json.dumps(
        {
            "top_video_path": str(top_video),
            "bottom_video_path": str(bottom_video),
            "global_offset_sec": global_offset,
            "anchor_count": len(anchors),
            "report_path": str(report_path),
            "fused_video_path": str(args.output_dir / "fused_swim.mp4") if not args.skip_fuse else None,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
