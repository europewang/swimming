"""
视频分析脚本：判断水上/水下 + 是否需要旋转
"""
import cv2
import numpy as np
import imageio.v2 as iio
from pathlib import Path

# 输出目录
OUTPUT_DIR = Path(r"d:\myflie\ALL_CODE\swimming\outputs_single")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 视频路径
VIDEO1 = Path(r"d:\myflie\ALL_CODE\swimming\video_merge\video1_min\DJI_20260515181348_0009_D_min.mp4")
VIDEO2 = Path(r"d:\myflie\ALL_CODE\swimming\video_merge\video2_min\DJI_20260515181343_0009_D_min.mp4")

def analyze_video(video_path: Path, sample_times: list[float] = [5.0, 10.0, 20.0]):
    """分析单个视频"""
    print(f"\n{'='*60}")
    print(f"分析视频: {video_path.name}")
    print(f"{'='*60}")

    # 使用 imageio-ffmpeg 读取视频
    reader = iio.get_reader(str(video_path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    size = meta.get("size", (0, 0))
    duration = float(meta.get("duration", 0.0))
    nframes = meta.get("nframes")

    print(f"\n[视频元数据]")
    print(f"  FPS: {fps}")
    print(f"  尺寸: {size[0]}x{size[1]} (WxH)")
    print(f"  时长: {duration:.2f}s")
    print(f"  总帧数: {nframes}")

    # 采样帧
    frames_at_times = {}
    for t in sample_times:
        frame_idx = int(round(t * fps))
        try:
            frame = reader.get_data(frame_idx)
            # 确保是 BGR 格式 (OpenCV 格式)
            if frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            frames_at_times[t] = frame_bgr
            print(f"\n[t={t}s] 帧 {frame_idx}:")
            print(f"  尺寸(HxW): {frame_bgr.shape[0]}x{frame_bgr.shape[1]}")
            print(f"  通道数: {frame_bgr.shape[2] if len(frame_bgr.shape) == 3 else 1}")
            print(f"  数据类型: {frame_bgr.dtype}")
        except Exception as e:
            print(f"\n[t={t}s] 读取失败: {e}")
            frames_at_times[t] = None

    reader.close()

    # 保存 t=5s 的帧截图
    if frames_at_times.get(5.0) is not None:
        frame_5s = frames_at_times[5.0]
        output_path = OUTPUT_DIR / f"{video_path.stem}_frame_t5.png"
        cv2.imwrite(str(output_path), frame_5s)
        print(f"\n[截图保存] {output_path}")

    # 分析 t=5s 帧
    print(f"\n{'='*60}")
    print(f"[详细分析 t=5s 帧]")
    print(f"{'='*60}")

    frame = frames_at_times.get(5.0)
    if frame is None:
        return None

    h, w = frame.shape[:2]

    # 分割上下半部分
    upper = frame[:h//2]
    lower = frame[h//2:]

    # Sobel 梯度计算
    gray_upper = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
    gray_lower = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)

    sobelx_upper = cv2.Sobel(gray_upper, cv2.CV_64F, 1, 0, ksize=3)
    sobely_upper = cv2.Sobel(gray_upper, cv2.CV_64F, 0, 1, ksize=3)
    gradient_upper = np.sqrt(sobelx_upper**2 + sobely_upper**2)

    sobelx_lower = cv2.Sobel(gray_lower, cv2.CV_64F, 1, 0, ksize=3)
    sobely_lower = cv2.Sobel(gray_lower, cv2.CV_64F, 0, 1, ksize=3)
    gradient_lower = np.sqrt(sobelx_lower**2 + sobely_lower**2)

    print(f"\n[Sobel 梯度分析]")
    print(f"  上半部平均梯度强度: {np.mean(gradient_upper):.4f}")
    print(f"  下半部平均梯度强度: {np.mean(gradient_lower):.4f}")
    print(f"  上半部梯度总和: {np.sum(gradient_upper):.4f}")
    print(f"  下半部梯度总和: {np.sum(gradient_lower):.4f}")

    # BGR 颜色分析
    print(f"\n[BGR 颜色分析]")
    upper_mean_b = np.mean(upper[:,:,0])
    upper_mean_g = np.mean(upper[:,:,1])
    upper_mean_r = np.mean(upper[:,:,2])
    lower_mean_b = np.mean(lower[:,:,0])
    lower_mean_g = np.mean(lower[:,:,1])
    lower_mean_r = np.mean(lower[:,:,2])

    print(f"  上半部 BGR: B={upper_mean_b:.2f}, G={upper_mean_g:.2f}, R={upper_mean_r:.2f}")
    print(f"  下半部 BGR: B={lower_mean_b:.2f}, G={lower_mean_g:.2f}, R={lower_mean_r:.2f}")
    print(f"  上半部蓝色 vs 红绿均值: {upper_mean_b:.2f} vs {(upper_mean_r + upper_mean_g)/2:.2f}")
    print(f"  下半部蓝色 vs 红绿均值: {lower_mean_b:.2f} vs {(lower_mean_r + lower_mean_g)/2:.2f}")

    # 水上视角旋转判断
    # 公式: np.mean(upper[:,:,2]) - np.mean(upper[:,:,0:2])/2.0 > np.mean(lower[:,:,2]) - np.mean(lower[:,:,0:2])/2.0
    upper_score = np.mean(upper[:,:,2]) - np.mean(upper[:,:,0:2])/2.0
    lower_score = np.mean(lower[:,:,2]) - np.mean(lower[:,:,0:2])/2.0
    is_top_rotated = upper_score > lower_score

    print(f"\n[水上视角旋转判断 (is_top_camera_rotated)]")
    print(f"  上半部红色权重: {upper_score:.4f}")
    print(f"  下半部红色权重: {lower_score:.4f}")
    print(f"  需要旋转180°: {is_top_rotated}")
    print(f"  逻辑: {'上半部红色权重 > 下半部红色权重' if is_top_rotated else '下半部红色权重 >= 上半部红色权重'}")

    # 水下视角旋转判断
    # 公式: np.mean(gradient_upper) > np.mean(gradient_lower)
    is_underwater_rotated = np.mean(gradient_upper) > np.mean(gradient_lower)

    print(f"\n[水下视角旋转判断 (is_underwater_camera_rotated)]")
    print(f"  上半部梯度均值: {np.mean(gradient_upper):.4f}")
    print(f"  下半部梯度均值: {np.mean(gradient_lower):.4f}")
    print(f"  需要旋转180°: {is_underwater_rotated}")
    print(f"  逻辑: {'上半部梯度 > 下半部梯度' if is_underwater_rotated else '下半部梯度 >= 上半部梯度'}")

    return {
        "video": video_path.name,
        "fps": fps,
        "size": size,
        "duration": duration,
        "frame_5s_path": str(OUTPUT_DIR / f"{video_path.stem}_frame_t5.png"),
        "gradient_upper_mean": float(np.mean(gradient_upper)),
        "gradient_lower_mean": float(np.mean(gradient_lower)),
        "upper_bgr": (float(upper_mean_b), float(upper_mean_g), float(upper_mean_r)),
        "lower_bgr": (float(lower_mean_b), float(lower_mean_g), float(lower_mean_r)),
        "is_top_rotated": is_top_rotated,
        "is_underwater_rotated": is_underwater_rotated,
    }


def main():
    print("=" * 70)
    print("视频分析: 判断水上/水下 + 是否需要旋转")
    print("=" * 70)

    # 分析两个视频
    result1 = analyze_video(VIDEO1, [5.0, 10.0, 20.0])
    result2 = analyze_video(VIDEO2, [5.0, 10.0, 20.0])

    # 综合判断
    print(f"\n{'='*70}")
    print("综合判断")
    print(f"{'='*70}")

    if result1 and result2:
        # 根据旋转判断哪个是水上哪个是水下
        print(f"\n[视频1: {VIDEO1.name}]")
        print(f"  - 水上视角需要旋转: {result1['is_top_rotated']}")
        print(f"  - 水下视角需要旋转: {result1['is_underwater_rotated']}")

        print(f"\n[视频2: {VIDEO2.name}]")
        print(f"  - 水上视角需要旋转: {result2['is_top_rotated']}")
        print(f"  - 水下视角需要旋转: {result2['is_underwater_rotated']}")

        # 判断逻辑：
        # 如果视频是水上拍摄，上半部通常是水面/天空(蓝色)，下半部是水/运动员(红色权重更高)
        # 如果视频是水下拍摄，上半部通常是水面波光(梯度高)，下半部是池底(梯度低)

        print(f"\n[分析结论]")
        # 水上视频特征：下半部红色权重更高（运动员皮肤色）
        # 水下视频特征：上半部梯度更高（水面波光折射）

        video1_above_score = result1['upper_bgr'][0] - (result1['upper_bgr'][1] + result1['upper_bgr'][2])/2  # 蓝色偏多
        video1_below_score = result1['lower_bgr'][0] - (result1['lower_bgr'][1] + result1['lower_bgr'][2])/2

        video2_above_score = result2['upper_bgr'][0] - (result2['upper_bgr'][1] + result2['upper_bgr'][2])/2
        video2_below_score = result2['lower_bgr'][0] - (result2['lower_bgr'][1] + result2['lower_bgr'][2])/2

        print(f"  视频1 上半部蓝色偏度: {video1_above_score:.2f}")
        print(f"  视频1 下半部蓝色偏度: {video1_below_score:.2f}")
        print(f"  视频2 上半部蓝色偏度: {video2_above_score:.2f}")
        print(f"  视频2 下半部蓝色偏度: {video2_below_score:.2f}")

        # 更蓝 = 水上（水面/天空）
        if video1_above_score > video2_above_score:
            print(f"\n  -> 视频1 更可能是【水上视角】(上半部更蓝)")
        else:
            print(f"\n  -> 视频2 更可能是【水上视角】(上半部更蓝)")

        if result1['gradient_upper_mean'] > result2['gradient_upper_mean']:
            print(f"  -> 视频1 更可能是【水下视角】(上半部梯度更大，水面波光)")
        else:
            print(f"  -> 视频2 更可能是【水下视角】(上半部梯度更大，水面波光)")

        print(f"\n[截图路径]")
        print(f"  视频1 t=5s: {result1['frame_5s_path']}")
        print(f"  视频2 t=5s: {result2['frame_5s_path']}")

    print(f"\n{'='*70}")
    print("分析完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
