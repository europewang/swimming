from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import imageio.v2 as iio
import imageio_ffmpeg
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


@dataclass
class VideoCandidate:
    path: Path
    timestamp: datetime
    stem: str


@dataclass
class VideoPair:
    video_a: Path
    video_b: Path
    delta_minutes: float
    pair_name: str


@dataclass
class VideoPreprocess:
    path: Path
    trim_start_sec: float
    trim_end_sec: float
    rotate_180: bool = False

    @property
    def usable_duration(self) -> float:
        return max(0.0, self.trim_end_sec - self.trim_start_sec)


def torch_cuda_available() -> bool:
    if YOLO is None:
        return False
    try:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect_best_video_codec(prefer_gpu_encoder: bool) -> str:
    if not prefer_gpu_encoder:
        return "libx264"
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        res = subprocess.run(
            [exe, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            errors="ignore",
            timeout=20,
        )
        stdout = res.stdout.lower()
        if "h264_nvenc" in stdout:
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


def rotate_frame_if_needed(frame: np.ndarray, rotate_180: bool) -> np.ndarray:
    if not rotate_180:
        return frame
    return cv2.rotate(frame, cv2.ROTATE_180)


class PersonSegmenter:
    def __init__(self, enabled: bool = True, model_name: str = "yolov8n-seg.pt", device: str = "auto") -> None:
        self.enabled = enabled and YOLO is not None
        self.model_name = model_name
        self._model = None
        self.available = False
        self.device = "cpu"
        if self.enabled:
            try:
                if device == "auto":
                    self.device = "cuda:0" if torch_cuda_available() else "cpu"
                else:
                    self.device = device
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
                device=self.device,
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


def frame_pool_metrics(frame: np.ndarray) -> dict[str, float]:
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    small_f = small.astype(np.float32) / 255.0
    r = small_f[:, :, 0]
    g = small_f[:, :, 1]
    b = small_f[:, :, 2]
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_u8 = (gray * 255.0).astype(np.uint8)
    upper = slice(0, 90)
    lower = slice(90, 180)
    center_start = 75
    center_end = 105
    vertical_grad = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    center_edge = float(np.mean(np.abs(vertical_grad[center_start:center_end, :]))) / 255.0
    return {
        "blue": float(np.mean(b - 0.5 * (r + g))),
        "sat": float(np.mean(hsv[:, :, 1])) / 255.0,
        "upper_blue": float(np.mean(b[upper] - 0.5 * (r[upper] + g[upper]))),
        "lower_blue": float(np.mean(b[lower] - 0.5 * (r[lower] + g[lower]))),
        "upper_brightness": float(np.mean(gray[upper])),
        "lower_brightness": float(np.mean(gray[lower])),
        "upper_darkness": float(1.0 - np.mean(gray[upper])),
        "lower_darkness": float(1.0 - np.mean(gray[lower])),
        "center_edge": center_edge,
    }


def pool_presence_score(metrics: dict[str, float]) -> float:
    return (
        metrics["blue"] * 5.0
        + metrics["sat"] * 1.4
        + max(metrics["upper_blue"], metrics["lower_blue"]) * 2.2
    )


def analyze_usable_segment(path: Path, sample_step_sec: float = 1.0) -> VideoPreprocess:
    meta = load_video_meta(path)
    reader = iio.get_reader(str(path))
    scores: list[tuple[float, bool]] = []
    current = 0.0
    while current < max(meta.duration, sample_step_sec):
        frame_idx = int(round(current * meta.fps))
        try:
            frame = reader.get_data(frame_idx)
        except Exception:
            break
        metrics = frame_pool_metrics(frame)
        score = pool_presence_score(metrics)
        usable = (metrics["blue"] > 0.07 and metrics["sat"] > 0.22) or score > 1.0
        scores.append((current, usable))
        current += sample_step_sec
    reader.close()

    valid_times = [t for t, usable in scores if usable]
    if not valid_times:
        return VideoPreprocess(path=path, trim_start_sec=0.0, trim_end_sec=meta.duration, rotate_180=False)

    trim_start = max(0.0, valid_times[0] - sample_step_sec)
    trim_end = min(meta.duration, valid_times[-1] + sample_step_sec)
    if trim_end <= trim_start:
        trim_start = 0.0
        trim_end = meta.duration
    return VideoPreprocess(path=path, trim_start_sec=trim_start, trim_end_sec=trim_end, rotate_180=False)


def iter_analysis_frames(
    path: Path,
    preprocess: VideoPreprocess,
    max_samples: int = 8,
    rotate_180: bool = False,
) -> Iterable[np.ndarray]:
    reader = iio.get_reader(str(path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    start = preprocess.trim_start_sec
    end = preprocess.trim_end_sec if preprocess.trim_end_sec > start else float(meta.get("duration", 0.0))
    duration = max(0.1, end - start)
    timestamps = np.linspace(start, end - min(1.0 / max(fps, 1.0), duration / max_samples), num=max_samples)
    for ts in timestamps:
        frame_idx = int(round(ts * fps))
        try:
            frame = reader.get_data(frame_idx)
        except Exception:
            continue
        yield rotate_frame_if_needed(frame, rotate_180=rotate_180)
    reader.close()


def orientation_score_for_role(path: Path, preprocess: VideoPreprocess, role: str, rotate_180: bool) -> float:
    scores: list[float] = []
    for frame in iter_analysis_frames(path, preprocess=preprocess, max_samples=8, rotate_180=rotate_180):
        m = frame_pool_metrics(frame)
        if role == "top":
            score = (
                (m["lower_blue"] - m["upper_blue"]) * 6.0
                + (m["lower_brightness"] - m["upper_brightness"]) * 3.0
                + m["upper_darkness"] * 2.5
                + m["center_edge"] * 5.0
            )
        else:
            score = (
                (m["upper_blue"] + m["lower_blue"]) * 4.5
                - abs(m["lower_blue"] - m["upper_blue"]) * 4.0
                - m["upper_darkness"] * 1.5
                - m["center_edge"] * 4.0
                + (m["lower_brightness"] - m["upper_brightness"]) * 2.5
            )
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def choose_video_roles_and_orientation(
    video_a: Path,
    video_b: Path,
    preprocess_a: VideoPreprocess,
    preprocess_b: VideoPreprocess,
) -> tuple[RoleDetection, VideoPreprocess, VideoPreprocess]:
    role_scores: dict[tuple[str, bool, str], float] = {}
    for video_name, video_path, preprocess in (
        ("A", video_a, preprocess_a),
        ("B", video_b, preprocess_b),
    ):
        for rotate_180 in (False, True):
            role_scores[(video_name, rotate_180, "top")] = orientation_score_for_role(
                video_path,
                preprocess=preprocess,
                role="top",
                rotate_180=rotate_180,
            )
            role_scores[(video_name, rotate_180, "underwater")] = orientation_score_for_role(
                video_path,
                preprocess=preprocess,
                role="underwater",
                rotate_180=rotate_180,
            )

    candidates: list[tuple[float, str, bool, bool]] = []
    for rotate_a in (False, True):
        for rotate_b in (False, True):
            score_a_top = role_scores[("A", rotate_a, "top")]
            score_a_under = role_scores[("A", rotate_a, "underwater")]
            score_b_top = role_scores[("B", rotate_b, "top")]
            score_b_under = role_scores[("B", rotate_b, "underwater")]
            candidates.append((score_a_top + score_b_under, "A_top", rotate_a, rotate_b))
            candidates.append((score_b_top + score_a_under, "B_top", rotate_a, rotate_b))

    best_score, assignment, best_rotate_a, best_rotate_b = max(candidates, key=lambda item: item[0])
    final_preprocess_a = VideoPreprocess(
        path=preprocess_a.path,
        trim_start_sec=preprocess_a.trim_start_sec,
        trim_end_sec=preprocess_a.trim_end_sec,
        rotate_180=best_rotate_a,
    )
    final_preprocess_b = VideoPreprocess(
        path=preprocess_b.path,
        trim_start_sec=preprocess_b.trim_start_sec,
        trim_end_sec=preprocess_b.trim_end_sec,
        rotate_180=best_rotate_b,
    )

    if assignment == "A_top":
        top_path = video_a
        underwater_path = video_b
        score_a = role_scores[("A", best_rotate_a, "top")]
        score_b = role_scores[("B", best_rotate_b, "underwater")]
    else:
        top_path = video_b
        underwater_path = video_a
        score_a = role_scores[("A", best_rotate_a, "underwater")]
        score_b = role_scores[("B", best_rotate_b, "top")]

    reason = (
        f"联合评估机位角色与180度方向。最佳组合={assignment}, "
        f"A.rotate_180={best_rotate_a}, B.rotate_180={best_rotate_b}, total_score={best_score:.3f}。"
        f" A[top/under]={{正:{role_scores[('A', False, 'top')]:.3f}/{role_scores[('A', False, 'underwater')]:.3f},"
        f" 反:{role_scores[('A', True, 'top')]:.3f}/{role_scores[('A', True, 'underwater')]:.3f}}};"
        f" B[top/under]={{正:{role_scores[('B', False, 'top')]:.3f}/{role_scores[('B', False, 'underwater')]:.3f},"
        f" 反:{role_scores[('B', True, 'top')]:.3f}/{role_scores[('B', True, 'underwater')]:.3f}}}"
    )
    role_detection = RoleDetection(
        video_a_path=str(video_a),
        video_b_path=str(video_b),
        underwater_path=str(underwater_path),
        top_path=str(top_path),
        score_a=float(score_a),
        score_b=float(score_b),
        reason=reason,
    )
    return role_detection, final_preprocess_a, final_preprocess_b


def evenly_spaced_frames(path: Path, max_samples: int = 12, preprocess: VideoPreprocess | None = None) -> list[np.ndarray]:
    reader = iio.get_reader(str(path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    duration = float(meta.get("duration", 0.0))
    start_sec = preprocess.trim_start_sec if preprocess is not None else 0.0
    end_sec = preprocess.trim_end_sec if preprocess is not None else duration
    end_sec = min(end_sec, duration)
    if end_sec <= start_sec:
        start_sec = 0.0
        end_sec = duration
    sample_count = max(1, max_samples)
    sample_times = np.linspace(start_sec, max(start_sec, end_sec - 1.0 / max(fps, 1.0)), num=sample_count)
    frames: list[np.ndarray] = []
    for ts in sample_times:
        frame_idx = int(round(ts * fps))
        try:
            frame = reader.get_data(int(frame_idx))
            if preprocess is not None:
                frame = rotate_frame_if_needed(frame, rotate_180=preprocess.rotate_180)
            frames.append(frame)
        except Exception:
            continue
    reader.close()
    return frames


def underwater_likelihood(path: Path, preprocess: VideoPreprocess | None = None) -> tuple[float, dict[str, float]]:
    frames = evenly_spaced_frames(path, preprocess=preprocess)
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


def detect_video_roles(
    video_a: Path,
    video_b: Path,
    preprocess_a: VideoPreprocess | None = None,
    preprocess_b: VideoPreprocess | None = None,
) -> RoleDetection:
    score_a, metrics_a = underwater_likelihood(video_a, preprocess=preprocess_a)
    score_b, metrics_b = underwater_likelihood(video_b, preprocess=preprocess_b)
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


def parse_video_timestamp(path: Path) -> datetime:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0].startswith("DJI") and len(parts[1]) == 14:
        try:
            return datetime.strptime(parts[1], "%Y%m%d%H%M%S")
        except ValueError:
            pass
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d_%H%M"):
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法从文件名解析时间，期望类似 20210627_1231.mp4: {path.name}")


def collect_video_candidates(folder: Path) -> list[VideoCandidate]:
    candidates: list[VideoCandidate] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".mkv"}:
            continue
        candidates.append(
            VideoCandidate(
                path=path,
                timestamp=parse_video_timestamp(path),
                stem=path.stem,
            )
        )
    if not candidates:
        raise ValueError(f"目录下没有可处理视频: {folder}")
    return candidates


def pair_videos_by_nearest_time(
    camera_a_dir: Path,
    camera_b_dir: Path,
    max_delta_minutes: float,
) -> list[VideoPair]:
    candidates_a = collect_video_candidates(camera_a_dir)
    candidates_b = collect_video_candidates(camera_b_dir)
    remaining_b = candidates_b.copy()
    pairs: list[VideoPair] = []

    for item_a in candidates_a:
        if not remaining_b:
            break
        best = min(
            remaining_b,
            key=lambda item_b: abs((item_a.timestamp - item_b.timestamp).total_seconds()),
        )
        delta_minutes = abs((item_a.timestamp - best.timestamp).total_seconds()) / 60.0
        if delta_minutes > max_delta_minutes:
            continue
        remaining_b.remove(best)
        pair_name = f"{item_a.stem}__{best.stem}"
        pairs.append(
            VideoPair(
                video_a=item_a.path,
                video_b=best.path,
                delta_minutes=delta_minutes,
                pair_name=pair_name,
            )
        )
    return pairs


def iter_sampled_frames(
    path: Path,
    sample_fps: float,
    preprocess: VideoPreprocess | None = None,
    max_frames: int | None = None,
) -> Iterable[tuple[float, np.ndarray]]:
    reader = iio.get_reader(str(path))
    meta = reader.get_meta_data()
    native_fps = float(meta.get("fps", 30.0))
    step = max(int(round(native_fps / sample_fps)), 1)
    start_frame = int(round((preprocess.trim_start_sec if preprocess is not None else 0.0) * native_fps))
    end_sec = preprocess.trim_end_sec if preprocess is not None else float(meta.get("duration", 0.0))
    end_frame = int(round(end_sec * native_fps)) if end_sec > 0 else None
    yielded = 0
    for frame_idx, frame in enumerate(reader):
        if frame_idx < start_frame:
            continue
        if end_frame is not None and frame_idx > end_frame:
            break
        if frame_idx % step != 0:
            continue
        if preprocess is not None:
            frame = rotate_frame_if_needed(frame, rotate_180=preprocess.rotate_180)
        time_sec = frame_idx / native_fps - (preprocess.trim_start_sec if preprocess is not None else 0.0)
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


def build_signature_timeline(path: Path, sample_fps: float, underwater: bool, preprocess: VideoPreprocess | None = None) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    features: list[np.ndarray] = []
    prev_gray = None
    for time_sec, frame in iter_sampled_frames(path, sample_fps=sample_fps, preprocess=preprocess):
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


def sample_frame_at(reader, time_sec: float, fps: float, preprocess: VideoPreprocess | None = None) -> np.ndarray | None:
    if time_sec < 0:
        return None
    actual_time_sec = time_sec + (preprocess.trim_start_sec if preprocess is not None else 0.0)
    frame_idx = int(round(actual_time_sec * fps))
    try:
        frame = reader.get_data(frame_idx)
        if preprocess is not None:
            frame = rotate_frame_if_needed(frame, rotate_180=preprocess.rotate_180)
        return frame
    except Exception:
        return None


def build_background_model(path: Path, target_width: int, preprocess: VideoPreprocess | None = None, max_samples: int = 21) -> np.ndarray:
    reader = iio.get_reader(str(path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    duration = float(meta.get("duration", 0.0))
    start_sec = preprocess.trim_start_sec if preprocess is not None else 0.0
    end_sec = preprocess.trim_end_sec if preprocess is not None else duration
    total_frames = max(1, int(round(duration * fps))) if duration > 0 else max_samples
    start_idx = int(round(start_sec * fps))
    end_idx = min(total_frames - 1, int(round(end_sec * fps))) if end_sec > 0 else total_frames - 1
    sample_indices = np.linspace(start_idx, max(start_idx, end_idx), num=min(max_samples, max(1, end_idx - start_idx + 1)), dtype=int)
    frames: list[np.ndarray] = []
    for idx in sample_indices:
        try:
            frame = reader.get_data(int(idx))
        except Exception:
            continue
        if preprocess is not None:
            frame = rotate_frame_if_needed(frame, rotate_180=preprocess.rotate_180)
        frames.append(resize_frame(frame, target_width=target_width))
    reader.close()
    if not frames:
        raise ValueError(f"无法建立背景模型: {path}")
    stack = np.stack(frames).astype(np.float32)
    return np.median(stack, axis=0).astype(np.uint8)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_bbox(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x + 1),
        min(h, y2 + pad_y + 1),
    )


def paste_mask(mask_roi: np.ndarray, roi_bbox: tuple[int, int, int, int], frame_shape: tuple[int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi_bbox
    full = np.zeros(frame_shape[:2], dtype=np.uint8)
    full[y1:y2, x1:x2] = mask_roi
    return full


def estimate_person_mask(
    frame: np.ndarray,
    underwater: bool,
    background: np.ndarray | None = None,
    prev_frame: np.ndarray | None = None,
    segmenter: PersonSegmenter | None = None,
    use_roi_segmentation: bool = True,
    seed_mask: np.ndarray | None = None,
) -> np.ndarray:
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

    full_coarse = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    coarse_bbox = mask_bbox(full_coarse)

    if seed_mask is not None and np.count_nonzero(seed_mask) > 0:
        seed_kernel = np.ones((15, 15), np.uint8)
        seed_mask = cv2.dilate(seed_mask, seed_kernel, iterations=1)
        merged_seed = cv2.bitwise_or(full_coarse, seed_mask)
        merged_seed = keep_primary_blob(merged_seed, prefer_lower_half=underwater)
        if np.count_nonzero(merged_seed) > 0:
            full_coarse = merged_seed
            coarse_bbox = mask_bbox(full_coarse)

    # ROI optimization: use the cheap coarse mask to localize the swimmer,
    # then run the expensive segmentation only inside that neighborhood.
    if segmenter is not None and coarse_bbox is not None and use_roi_segmentation:
        x1, y1, x2, y2 = coarse_bbox
        bbox_w = max(1, x2 - x1 + 1)
        bbox_h = max(1, y2 - y1 + 1)
        roi_bbox = expand_bbox(
            coarse_bbox,
            frame.shape,
            pad_x=max(48, int(bbox_w * 0.4)),
            pad_y=max(48, int(bbox_h * 0.5)),
        )
        rx1, ry1, rx2, ry2 = roi_bbox
        frame_roi = frame[ry1:ry2, rx1:rx2]
        roi_mask = segmenter.segment_person(frame_roi)
        if roi_mask is not None and np.count_nonzero(roi_mask) > 300:
            kernel = np.ones((7, 7), np.uint8)
            roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
            roi_mask = keep_primary_blob(roi_mask, prefer_lower_half=underwater)
            if np.count_nonzero(roi_mask) > 0:
                coarse_roi = full_coarse[ry1:ry2, rx1:rx2]
                merged_roi = cv2.bitwise_or(roi_mask, coarse_roi)
                merged_roi = keep_primary_blob(merged_roi, prefer_lower_half=underwater)
                if np.count_nonzero(merged_roi) > 0:
                    dilate_kernel = np.ones((15, 15), np.uint8)
                    merged_roi = cv2.dilate(merged_roi, dilate_kernel, iterations=1)
                    return paste_mask(merged_roi, roi_bbox, frame.shape)

    # Fallback path: if ROI is disabled, or coarse localization failed,
    # run full-frame segmentation as a slower but more robust option.
    if segmenter is not None and ((not use_roi_segmentation) or coarse_bbox is None):
        yolo_mask = segmenter.segment_person(frame)
        if yolo_mask is not None and np.count_nonzero(yolo_mask) > 500:
            kernel = np.ones((7, 7), np.uint8)
            yolo_mask = cv2.morphologyEx(yolo_mask, cv2.MORPH_CLOSE, kernel)
            yolo_mask = keep_primary_blob(yolo_mask, prefer_lower_half=underwater)
            if np.count_nonzero(yolo_mask) > 0:
                merged = cv2.bitwise_or(yolo_mask, full_coarse)
                merged = keep_primary_blob(merged, prefer_lower_half=underwater)
                if np.count_nonzero(merged) > 0:
                    dilate_kernel = np.ones((15, 15), np.uint8)
                    merged = cv2.dilate(merged, dilate_kernel, iterations=1)
                    return merged

    return full_coarse


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
    top_preprocess: VideoPreprocess,
    bottom_preprocess: VideoPreprocess,
    output_path: Path,
    curve_times: np.ndarray,
    curve_offsets: np.ndarray,
    waterline_ratio: float,
    blend_px: int,
    output_fps: float,
    max_output_width: int,
    use_person_segmentation: bool,
    segmentation_device: str,
    video_codec: str,
    use_roi_segmentation: bool,
    yolo_interval: int,
) -> None:
    interp = interp1d(curve_times, curve_offsets, kind="linear", fill_value="extrapolate")
    top_reader = iio.get_reader(str(top_path))
    bottom_reader = iio.get_reader(str(bottom_path))
    top_background = build_background_model(top_path, target_width=min(max_output_width, top_meta.width), preprocess=top_preprocess)
    bottom_background = build_background_model(bottom_path, target_width=min(max_output_width, bottom_meta.width), preprocess=bottom_preprocess)
    segmenter = PersonSegmenter(enabled=use_person_segmentation, device=segmentation_device)
    overlap_start = max(0.0, -float(curve_offsets.min()))
    overlap_end = min(top_preprocess.usable_duration, bottom_preprocess.usable_duration - float(curve_offsets.max()))
    if overlap_end <= overlap_start:
        raise ValueError("两个视频没有有效重叠区间，无法输出融合视频。")
    writer = iio.get_writer(str(output_path), fps=output_fps, codec=video_codec, quality=8)
    current = overlap_start
    prev_top_frame = None
    prev_bottom_frame = None
    prev_top_mask = None
    prev_bottom_mask = None
    frame_counter = 0
    while current < overlap_end:
        bottom_time = current + float(interp(current))
        top_frame = sample_frame_at(top_reader, current, top_meta.fps, preprocess=top_preprocess)
        bottom_frame = sample_frame_at(bottom_reader, bottom_time, bottom_meta.fps, preprocess=bottom_preprocess)
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
        run_segmentation_this_frame = (
            use_person_segmentation
            and (frame_counter % max(yolo_interval, 1) == 0 or prev_top_mask is None or prev_bottom_mask is None)
        )
        top_mask = estimate_person_mask(
            top_frame,
            underwater=False,
            background=top_background,
            prev_frame=prev_top_frame,
            segmenter=segmenter if run_segmentation_this_frame else None,
            use_roi_segmentation=use_roi_segmentation,
            seed_mask=prev_top_mask if not run_segmentation_this_frame else None,
        )
        bottom_mask = estimate_person_mask(
            bottom_frame,
            underwater=True,
            background=bottom_background,
            prev_frame=prev_bottom_frame,
            segmenter=segmenter if run_segmentation_this_frame else None,
            use_roi_segmentation=use_roi_segmentation,
            seed_mask=prev_bottom_mask if not run_segmentation_this_frame else None,
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
        prev_top_mask = top_mask
        prev_bottom_mask = bottom_mask
        frame_counter += 1
        current += 1.0 / output_fps
    writer.close()
    top_reader.close()
    bottom_reader.close()


def run_alignment_and_fusion(
    *,
    top_video: Path,
    bottom_video: Path,
    top_preprocess: VideoPreprocess,
    bottom_preprocess: VideoPreprocess,
    output_dir: Path,
    sample_fps: float,
    anchor_step_sec: float,
    window_sec: float,
    search_radius_sec: float,
    waterline_ratio: float,
    blend_px: int,
    output_fps: float,
    max_output_width: int,
    use_person_segmentation: bool,
    skip_fuse: bool,
    segmentation_device: str,
    video_codec: str,
    use_roi_segmentation: bool,
    yolo_interval: int,
    role_detection: RoleDetection | None = None,
    report_filename: str = "alignment_report.json",
    fused_video_filename: str = "fused_swim.mp4",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    top_meta = load_video_meta(top_video)
    bottom_meta = load_video_meta(bottom_video)

    top_ts, top_sig = build_signature_timeline(top_video, sample_fps=sample_fps, underwater=False, preprocess=top_preprocess)
    bottom_ts, bottom_sig = build_signature_timeline(bottom_video, sample_fps=sample_fps, underwater=True, preprocess=bottom_preprocess)

    global_offset = global_offset_from_signatures(
        ts_a=top_ts,
        sig_a=top_sig,
        ts_b=bottom_ts,
        sig_b=bottom_sig,
        sample_fps=sample_fps,
    )
    anchors = refine_alignment_points(
        ts_a=top_ts,
        sig_a=top_sig,
        ts_b=bottom_ts,
        sig_b=bottom_sig,
        base_offset_sec=global_offset,
        anchor_step_sec=anchor_step_sec,
        window_sec=window_sec,
        search_radius_sec=search_radius_sec,
        sample_fps=sample_fps,
    )
    total_duration = min(top_preprocess.usable_duration, bottom_preprocess.usable_duration)
    curve_times, curve_offsets = smooth_offsets(anchors, total_duration=total_duration)

    report_path = output_dir / report_filename
    save_alignment_report(
        output_path=report_path,
        top_meta=top_meta,
        bottom_meta=bottom_meta,
        global_offset=global_offset,
        anchors=anchors,
        curve_times=curve_times,
        curve_offsets=curve_offsets,
        top_preprocess=top_preprocess,
        bottom_preprocess=bottom_preprocess,
        role_detection=role_detection,
    )

    output_video_path = output_dir / fused_video_filename
    if not skip_fuse:
        write_fused_video(
            top_path=top_video,
            bottom_path=bottom_video,
            top_meta=top_meta,
            bottom_meta=bottom_meta,
            top_preprocess=top_preprocess,
            bottom_preprocess=bottom_preprocess,
            output_path=output_video_path,
            curve_times=curve_times,
            curve_offsets=curve_offsets,
            waterline_ratio=waterline_ratio,
            blend_px=blend_px,
            output_fps=output_fps,
            max_output_width=max_output_width,
            use_person_segmentation=use_person_segmentation,
            segmentation_device=segmentation_device,
            video_codec=video_codec,
            use_roi_segmentation=use_roi_segmentation,
            yolo_interval=yolo_interval,
        )

    return {
        "top_video_path": str(top_video),
        "bottom_video_path": str(bottom_video),
        "global_offset_sec": global_offset,
        "anchor_count": len(anchors),
        "report_path": str(report_path),
        "fused_video_path": str(output_video_path) if not skip_fuse else None,
        "segmentation_device": segmentation_device,
        "video_codec": video_codec,
        "use_roi_segmentation": use_roi_segmentation,
        "yolo_interval": yolo_interval,
        "top_trim_start_sec": top_preprocess.trim_start_sec,
        "top_trim_end_sec": top_preprocess.trim_end_sec,
        "bottom_trim_start_sec": bottom_preprocess.trim_start_sec,
        "bottom_trim_end_sec": bottom_preprocess.trim_end_sec,
        "top_rotate_180": top_preprocess.rotate_180,
        "bottom_rotate_180": bottom_preprocess.rotate_180,
    }


def process_batch_pair(task: dict[str, object]) -> dict[str, object]:
    video_a = Path(str(task["camera_a_video"]))
    video_b = Path(str(task["camera_b_video"]))
    pair_name = str(task["pair_name"])
    fusion_root = Path(str(task["fusion_root"]))
    report_path = fusion_root / f"{pair_name}_alignment_report.json"
    fused_video_path = fusion_root / f"{pair_name}.mp4"
    skip_fuse = bool(task["skip_fuse"])
    overwrite_existing = bool(task["overwrite_existing"])

    if not overwrite_existing:
        report_exists = report_path.exists()
        fused_exists = fused_video_path.exists()
        if (skip_fuse and report_exists) or ((not skip_fuse) and report_exists and fused_exists):
            return {
                "pair_name": pair_name,
                "delta_minutes": float(task["delta_minutes"]),
                "camera_a_video": str(video_a),
                "camera_b_video": str(video_b),
                "top_video_path": None,
                "bottom_video_path": None,
                "global_offset_sec": None,
                "anchor_count": None,
                "report_path": str(report_path) if report_exists else None,
                "fused_video_path": str(fused_video_path) if fused_exists else None,
                "segmentation_device": str(task["segmentation_device"]),
                "video_codec": str(task["video_codec"]),
                "skipped_existing": True,
                "yolo_interval": int(task["yolo_interval"]),
            }

    preprocess_a = analyze_usable_segment(video_a)
    preprocess_b = analyze_usable_segment(video_b)
    role_detection, final_preprocess_a, final_preprocess_b = choose_video_roles_and_orientation(
        video_a,
        video_b,
        preprocess_a,
        preprocess_b,
    )
    if str(role_detection.top_path) == str(video_a):
        top_preprocess = final_preprocess_a
        bottom_preprocess = final_preprocess_b
    else:
        top_preprocess = final_preprocess_b
        bottom_preprocess = final_preprocess_a
    top_video = Path(role_detection.top_path)
    bottom_video = Path(role_detection.underwater_path)
    result = run_alignment_and_fusion(
        top_video=top_video,
        bottom_video=bottom_video,
        top_preprocess=top_preprocess,
        bottom_preprocess=bottom_preprocess,
        output_dir=fusion_root,
        sample_fps=float(task["sample_fps"]),
        anchor_step_sec=float(task["anchor_step_sec"]),
        window_sec=float(task["window_sec"]),
        search_radius_sec=float(task["search_radius_sec"]),
        waterline_ratio=float(task["waterline_ratio"]),
        blend_px=int(task["blend_px"]),
        output_fps=float(task["output_fps"]),
        max_output_width=int(task["max_output_width"]),
        use_person_segmentation=bool(task["use_person_segmentation"]),
        skip_fuse=skip_fuse,
        segmentation_device=str(task["segmentation_device"]),
        video_codec=str(task["video_codec"]),
        use_roi_segmentation=bool(task["use_roi_segmentation"]),
        yolo_interval=int(task["yolo_interval"]),
        role_detection=role_detection,
        report_filename=report_path.name,
        fused_video_filename=fused_video_path.name,
    )
    return {
        "pair_name": pair_name,
        "delta_minutes": float(task["delta_minutes"]),
        "camera_a_video": str(video_a),
        "camera_b_video": str(video_b),
        **result,
        "skipped_existing": False,
    }


def run_batch_processing(args: argparse.Namespace) -> dict[str, object]:
    if not args.batch_video_root:
        raise ValueError("批处理模式缺少 --batch-video-root")

    root = args.batch_video_root
    camera_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name != args.batch_output_dir_name])
    if len(camera_dirs) != 2:
        raise ValueError(f"批处理模式要求根目录下恰好有两个相机子目录，当前找到: {[p.name for p in camera_dirs]}")

    pairs = pair_videos_by_nearest_time(
        camera_a_dir=camera_dirs[0],
        camera_b_dir=camera_dirs[1],
        max_delta_minutes=args.max_pair_delta_minutes,
    )
    if not pairs:
        raise ValueError("没有找到可配对的视频，请检查文件名时间格式或放宽最大配对时间差。")

    fusion_root = root / args.batch_output_dir_name
    fusion_root.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, args.batch_workers)
    codec = detect_best_video_codec(prefer_gpu_encoder=not args.disable_gpu_encode)
    segmentation_device = args.segmentation_device
    if segmentation_device == "auto":
        segmentation_device = "cuda:0" if torch_cuda_available() else "cpu"

    tasks: list[dict[str, object]] = []
    for pair in pairs:
        tasks.append(
            {
                "pair_name": pair.pair_name,
                "delta_minutes": pair.delta_minutes,
                "camera_a_video": str(pair.video_a),
                "camera_b_video": str(pair.video_b),
                "fusion_root": str(fusion_root),
                "sample_fps": args.sample_fps,
                "anchor_step_sec": args.anchor_step_sec,
                "window_sec": args.window_sec,
                "search_radius_sec": args.search_radius_sec,
                "waterline_ratio": args.waterline_ratio,
                "blend_px": args.blend_px,
                "output_fps": args.output_fps,
                "max_output_width": args.max_output_width,
                "use_person_segmentation": not args.disable_person_segmentation,
                "skip_fuse": args.skip_fuse,
                "overwrite_existing": args.overwrite_existing,
                "segmentation_device": segmentation_device,
                "video_codec": codec,
                "use_roi_segmentation": not args.disable_roi_segmentation,
                "yolo_interval": args.yolo_interval,
            }
        )

    if segmentation_device.startswith("cuda") and worker_count > 1:
        worker_count = 1

    results: list[dict[str, object]] = []
    if worker_count == 1:
        for task in tasks:
            results.append(process_batch_pair(task))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(process_batch_pair, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda item: str(item["pair_name"]))

    summary_path = fusion_root / "batch_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "pair_count": len(results),
        "fusion_root": str(fusion_root),
        "summary_path": str(summary_path),
        "batch_workers": worker_count,
        "segmentation_device": segmentation_device,
        "video_codec": codec,
        "pairs": results,
    }


def save_alignment_report(
    output_path: Path,
    top_meta: VideoMeta,
    bottom_meta: VideoMeta,
    global_offset: float,
    anchors: list[AlignmentPoint],
    curve_times: np.ndarray,
    curve_offsets: np.ndarray,
    top_preprocess: VideoPreprocess,
    bottom_preprocess: VideoPreprocess,
    role_detection: RoleDetection | None = None,
) -> None:
    top_data = asdict(top_meta)
    bottom_data = asdict(bottom_meta)
    top_preprocess_data = asdict(top_preprocess)
    bottom_preprocess_data = asdict(bottom_preprocess)
    top_data["path"] = str(top_meta.path)
    bottom_data["path"] = str(bottom_meta.path)
    top_preprocess_data["path"] = str(top_preprocess.path)
    bottom_preprocess_data["path"] = str(bottom_preprocess.path)
    data = {
        "top_video": top_data,
        "bottom_video": bottom_data,
        "global_offset_sec": global_offset,
        "anchors": [asdict(point) for point in anchors],
        "curve_times_sec": curve_times.tolist(),
        "curve_offsets_sec": curve_offsets.tolist(),
        "top_preprocess": top_preprocess_data,
        "bottom_preprocess": bottom_preprocess_data,
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
    parser.add_argument("--force-top-video", choices=["a", "b"], help="单组模式下强制指定哪个输入是水上视频")
    parser.add_argument("--rotate-video-a-180", action="store_true", help="单组模式下强制将 video-a 旋转180度")
    parser.add_argument("--rotate-video-b-180", action="store_true", help="单组模式下强制将 video-b 旋转180度")
    parser.add_argument("--rotate-top-video-180", action="store_true", help="显式 top/bottom 模式下强制将 top-video 旋转180度")
    parser.add_argument("--rotate-bottom-video-180", action="store_true", help="显式 top/bottom 模式下强制将 bottom-video 旋转180度")
    parser.add_argument("--batch-video-root", type=Path, help="批处理根目录，内部应包含两个相机子目录")
    parser.add_argument("--batch-output-dir-name", type=str, default="融合", help="批处理输出目录名，默认在根目录下创建“融合”文件夹")
    parser.add_argument("--max-pair-delta-minutes", type=float, default=10.0, help="批处理时允许配对的最大时间差（分钟）")
    parser.add_argument("--batch-workers", type=int, default=1, help="批处理并行 worker 数；若使用 CUDA 分割会自动退回 1")
    parser.add_argument("--overwrite-existing", action="store_true", help="批处理时即使已存在输出也重新生成")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="输出目录")
    parser.add_argument("--sample-fps", type=float, default=6.0, help="对齐分析采样帧率")
    parser.add_argument("--anchor-step-sec", type=float, default=5.0, help="局部微调锚点间隔")
    parser.add_argument("--window-sec", type=float, default=2.0, help="局部对齐窗口长度")
    parser.add_argument("--search-radius-sec", type=float, default=1.2, help="局部搜索半径")
    parser.add_argument("--waterline-ratio", type=float, default=0.44, help="拼接水线高度比例")
    parser.add_argument("--blend-px", type=int, default=90, help="水线上下融合像素宽度")
    parser.add_argument("--output-fps", type=float, default=30.0, help="输出视频帧率")
    parser.add_argument("--max-output-width", type=int, default=1280, help="融合输出最大宽度，默认会缩小 4K 视频以加速")
    parser.add_argument("--yolo-interval", type=int, default=1, help="YOLO 调用间隔；1 表示每帧都调用，2 表示每隔 1 帧调用一次")
    parser.add_argument("--segmentation-device", type=str, default="auto", help="人体分割设备：auto、cpu、cuda:0")
    parser.add_argument("--disable-gpu-encode", action="store_true", help="禁用 NVENC/QSV 等硬件编码，强制使用 libx264")
    parser.add_argument("--disable-person-segmentation", action="store_true", help="禁用 YOLO 人体分割，仅使用传统图像法")
    parser.add_argument("--disable-roi-segmentation", action="store_true", help="禁用人体 ROI 优化，回退为整帧分割路径")
    parser.add_argument(
        "--skip-fuse",
        action="store_true",
        help="只生成时间对齐报告，不输出融合视频",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_video_root:
        result = run_batch_processing(args)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        role_detection = None
        if args.video_a and args.video_b:
            preprocess_a = analyze_usable_segment(args.video_a)
            preprocess_b = analyze_usable_segment(args.video_b)
            role_detection, final_preprocess_a, final_preprocess_b = choose_video_roles_and_orientation(
                args.video_a,
                args.video_b,
                preprocess_a,
                preprocess_b,
            )
            if args.force_top_video == "a":
                role_detection = RoleDetection(
                    video_a_path=str(args.video_a),
                    video_b_path=str(args.video_b),
                    underwater_path=str(args.video_b),
                    top_path=str(args.video_a),
                    score_a=role_detection.score_a,
                    score_b=role_detection.score_b,
                    reason=role_detection.reason + " 手动覆盖: force_top_video=a。",
                )
            elif args.force_top_video == "b":
                role_detection = RoleDetection(
                    video_a_path=str(args.video_a),
                    video_b_path=str(args.video_b),
                    underwater_path=str(args.video_a),
                    top_path=str(args.video_b),
                    score_a=role_detection.score_a,
                    score_b=role_detection.score_b,
                    reason=role_detection.reason + " 手动覆盖: force_top_video=b。",
                )

            if args.rotate_video_a_180:
                final_preprocess_a.rotate_180 = True
            if args.rotate_video_b_180:
                final_preprocess_b.rotate_180 = True

            if str(role_detection.top_path) == str(args.video_a):
                top_preprocess = final_preprocess_a
                bottom_preprocess = final_preprocess_b
            else:
                top_preprocess = final_preprocess_b
                bottom_preprocess = final_preprocess_a
            top_video = Path(role_detection.top_path)
            bottom_video = Path(role_detection.underwater_path)
        elif args.top_video and args.bottom_video:
            top_video = args.top_video
            bottom_video = args.bottom_video
            top_preprocess = analyze_usable_segment(top_video)
            bottom_preprocess = analyze_usable_segment(bottom_video)
            top_preprocess.rotate_180 = orientation_score_for_role(top_video, top_preprocess, role="top", rotate_180=True) > orientation_score_for_role(top_video, top_preprocess, role="top", rotate_180=False)
            bottom_preprocess.rotate_180 = orientation_score_for_role(bottom_video, bottom_preprocess, role="underwater", rotate_180=True) > orientation_score_for_role(bottom_video, bottom_preprocess, role="underwater", rotate_180=False)
            if args.rotate_top_video_180:
                top_preprocess.rotate_180 = True
            if args.rotate_bottom_video_180:
                bottom_preprocess.rotate_180 = True
        else:
            raise ValueError("请提供 --batch-video-root 批处理，或提供 --video-a/--video-b 自动识别，或提供 --top-video/--bottom-video 显式指定。")

        result = run_alignment_and_fusion(
            top_video=top_video,
            bottom_video=bottom_video,
            top_preprocess=top_preprocess,
            bottom_preprocess=bottom_preprocess,
            output_dir=args.output_dir,
            sample_fps=args.sample_fps,
            anchor_step_sec=args.anchor_step_sec,
            window_sec=args.window_sec,
            search_radius_sec=args.search_radius_sec,
            waterline_ratio=args.waterline_ratio,
            blend_px=args.blend_px,
            output_fps=args.output_fps,
            max_output_width=args.max_output_width,
            use_person_segmentation=not args.disable_person_segmentation,
            skip_fuse=args.skip_fuse,
            segmentation_device=("cuda:0" if torch_cuda_available() else "cpu") if args.segmentation_device == "auto" else args.segmentation_device,
            video_codec=detect_best_video_codec(prefer_gpu_encoder=not args.disable_gpu_encode),
            use_roi_segmentation=not args.disable_roi_segmentation,
            yolo_interval=args.yolo_interval,
            role_detection=role_detection,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
