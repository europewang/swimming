"""
水下视频帧颜色分布和亮度分布分析
"""
import cv2
import numpy as np
import imageio_ffmpeg

def analyze_underwater_frame(video_path, timestamp_sec=5):
    """
    分析水下视频帧的颜色分布和亮度分布
    """
    print(f"读取视频: {video_path}")
    print(f"分析时间点: t={timestamp_sec}s")

    # 使用 imageio_ffmpeg 读取视频
    import imageio
    reader = imageio.get_reader(video_path, 'ffmpeg')
    meta = reader.get_meta_data()
    fps = meta.get('fps', 30)
    frame_idx = int(timestamp_sec * fps)

    print(f"视频 FPS: {fps}, 目标帧索引: {frame_idx}")

    # 跳到目标帧
    frame = None
    for i, f in enumerate(reader):
        if i == frame_idx:
            frame = f
            break

    if frame is None:
        # 如果视频不够长，读取最后一帧
        frames = list(reader.iter_data())
        if frames:
            frame = frames[-1]
            print(f"视频长度不足，使用最后一帧 (索引 {len(frames)-1})")
        else:
            raise ValueError("无法读取视频帧")

    reader.close()

    # 转换为 BGR 格式 (OpenCV 格式)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    h, w = frame.shape[:2]
    print(f"帧尺寸: {w}x{h}")

    # 计算灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 逐行计算亮度均值
    row_brightness = np.mean(gray, axis=1)
    rows = np.arange(h)

    # 找出亮度最大的行（水面位置）
    max_brightness_row = np.argmax(row_brightness)
    max_brightness_value = row_brightness[max_brightness_row]
    normalized_water_surface = max_brightness_row / h

    print(f"\n{'='*60}")
    print(f"水面位置分析")
    print(f"{'='*60}")
    print(f"亮度最大的行: {max_brightness_row} (归一化: {normalized_water_surface:.4f})")
    print(f"该行亮度值: {max_brightness_value:.2f}")

    # 判断摄像头方向
    # 如果水面在帧的上半部 (< 0.5h)，说明摄像头朝上（正常）
    # 如果水面在帧的下半部 (> 0.5h)，说明摄像头装反了
    is_reversed = normalized_water_surface > 0.5

    print(f"\n{'='*60}")
    print(f"摄像头方向判断")
    print(f"{'='*60}")
    if is_reversed:
        print(f"⚠️ 摄像头可能装反了（水面在帧下半部）")
        print(f"   水面归一化位置: {normalized_water_surface:.4f} > 0.5")
    else:
        print(f"✓ 摄像头朝向正常（水面在帧上半部）")
        print(f"   水面归一化位置: {normalized_water_surface:.4f} <= 0.5")

    # 计算旋转值（基于水面位置）
    # 如果水面不在中间，需要旋转使得水面在中间位置
    center_offset = normalized_water_surface - 0.5
    rotation_value = center_offset * h  # 正值表示需要向上旋转，负值向下
    print(f"建议旋转值: {rotation_value:.1f} 像素 (基于水面位置偏移)")

    # 分割水面区域和水下区域
    water_surface_row = max_brightness_row

    # 水面以上区域 (0 到 最大亮度行)
    above_water_region = frame[:water_surface_row+1, :] if water_surface_row > 0 else frame[:1, :]
    # 水下区域 (最大亮度行 到 h)
    underwater_region = frame[water_surface_row:, :] if water_surface_row < h else frame

    print(f"\n{'='*60}")
    print(f"区域划分")
    print(f"{'='*60}")
    print(f"水面行: {water_surface_row}")
    print(f"水面以上区域: 0 到 {water_surface_row} ({above_water_region.shape[0]} 行)")
    print(f"水下区域: {water_surface_row} 到 {h} ({underwater_region.shape[0]} 行)")

    # 计算水面以上区域的统计
    if above_water_region.shape[0] > 0:
        above_gray = cv2.cvtColor(above_water_region, cv2.COLOR_BGR2GRAY)
        above_brightness = np.mean(above_gray)
        above_blue = np.mean(above_water_region[:, :, 0])  # B channel
        above_green = np.mean(above_water_region[:, :, 1])  # G channel
        above_red = np.mean(above_water_region[:, :, 2])    # R channel
        above_blue_deviation = above_blue - (above_red + above_green) / 2
    else:
        above_brightness = 0
        above_blue = 0
        above_green = 0
        above_red = 0
        above_blue_deviation = 0

    # 计算水下区域的统计
    if underwater_region.shape[0] > 0:
        underwater_gray = cv2.cvtColor(underwater_region, cv2.COLOR_BGR2GRAY)
        underwater_brightness = np.mean(underwater_gray)
        underwater_blue = np.mean(underwater_region[:, :, 0])   # B channel
        underwater_green = np.mean(underwater_region[:, :, 1])  # G channel
        underwater_red = np.mean(underwater_region[:, :, 2])    # R channel
        underwater_blue_deviation = underwater_blue - (underwater_red + underwater_green) / 2
    else:
        underwater_brightness = 0
        underwater_blue = 0
        underwater_green = 0
        underwater_red = 0
        underwater_blue_deviation = 0

    print(f"\n{'='*60}")
    print(f"水面以上区域统计")
    print(f"{'='*60}")
    print(f"平均亮度: {above_brightness:.2f}")
    print(f"平均蓝色通道 (B): {above_blue:.2f}")
    print(f"平均绿色通道 (G): {above_green:.2f}")
    print(f"平均红色通道 (R): {above_red:.2f}")
    print(f"蓝色偏差 B - (R+G)/2: {above_blue_deviation:.2f}")

    print(f"\n{'='*60}")
    print(f"水下区域统计")
    print(f"{'='*60}")
    print(f"平均亮度: {underwater_brightness:.2f}")
    print(f"平均蓝色通道 (B): {underwater_blue:.2f}")
    print(f"平均绿色通道 (G): {underwater_green:.2f}")
    print(f"平均红色通道 (R): {underwater_red:.2f}")
    print(f"蓝色偏差 B - (R+G)/2: {underwater_blue_deviation:.2f}")

    # ========== 保存亮度分布图 ==========
    print(f"\n{'='*60}")
    print(f"保存图像")
    print(f"{'='*60}")

    import matplotlib.pyplot as plt

    # 亮度分布图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 子图1: 逐行亮度分布
    ax1 = axes[0]
    ax1.plot(rows, row_brightness, 'b-', linewidth=0.8, label='Row Brightness')
    ax1.axhline(y=max_brightness_value, color='r', linestyle='--', alpha=0.7, label=f'Max Brightness: {max_brightness_value:.2f}')
    ax1.axvline(x=max_brightness_row, color='g', linestyle='--', alpha=0.7, label=f'Water Surface Row: {max_brightness_row}')
    ax1.axhline(y=255/2, color='orange', linestyle=':', alpha=0.5, label='Middle Brightness')
    ax1.fill_between(rows, 0, row_brightness, alpha=0.3)
    ax1.set_xlabel('Row Index (Y)')
    ax1.set_ylabel('Brightness (0-255)')
    ax1.set_title(f'Underwater Video Frame Brightness Profile (t={timestamp_sec}s)\nFrame: {w}x{h}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 添加区域标注
    ax1.annotate('Above Water', xy=(max_brightness_row/2, max_brightness_value),
                fontsize=10, color='brown')
    ax1.annotate('Underwater', xy=(max_brightness_row + (h-max_brightness_row)/2, max_brightness_value),
                fontsize=10, color='blue')

    # 子图2: 亮度梯度（帮助检测边缘）
    brightness_gradient = np.gradient(row_brightness)
    ax2 = axes[1]
    ax2.plot(rows, brightness_gradient, 'purple', linewidth=0.8, label='Brightness Gradient')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axvline(x=max_brightness_row, color='g', linestyle='--', alpha=0.7, label=f'Water Surface Row: {max_brightness_row}')
    ax2.set_xlabel('Row Index (Y)')
    ax2.set_ylabel('Brightness Gradient')
    ax2.set_title('Brightness Gradient (Helps Detect Edges)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(r'd:\myflie\ALL_CODE\swimming\underwater_brightness_profile.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存: underwater_brightness_profile.png")

    # ========== 保存标注水面位置的帧 ==========
    frame_marked = frame.copy()
    cv2.line(frame_marked, (0, water_surface_row), (w, water_surface_row), (0, 255, 0), 2)
    cv2.putText(frame_marked, f'Water Surface: Row {water_surface_row} ({normalized_water_surface:.2%})',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if is_reversed:
        cv2.putText(frame_marked, 'WARNING: Camera may be reversed!',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 添加颜色条显示各区域颜色特征
    bar_height = 30
    bar_y = h - bar_height

    # 水面上方颜色条
    if above_water_region.shape[0] > 0:
        avg_color_above = np.mean(above_water_region, axis=(0, 1)).astype(np.uint8)
        frame_marked[bar_y:bar_y+bar_height//2, :] = avg_color_above

    # 水下颜色条
    if underwater_region.shape[0] > 0:
        avg_color_underwater = np.mean(underwater_region, axis=(0, 1)).astype(np.uint8)
        frame_marked[bar_y+bar_height//2:bar_y+bar_height, :] = avg_color_underwater

    cv2.imwrite(r'd:\myflie\ALL_CODE\swimming\underwater_frame_marked.png', frame_marked)
    print(f"✓ 保存: underwater_frame_marked.png")

    # ========== 返回结果汇总 ==========
    results = {
        'frame_shape': {'width': w, 'height': h},
        'timestamp_sec': timestamp_sec,
        'water_surface': {
            'row_index': int(max_brightness_row),
            'normalized_position': float(normalized_water_surface),
            'brightness_value': float(max_brightness_value),
            'is_reversed': is_reversed
        },
        'rotation_suggestion': {
            'pixel_offset': float(rotation_value),
            'description': '正值表示向上旋转，负值向下旋转'
        },
        'above_water_region': {
            'row_range': f'0 to {water_surface_row}',
            'row_count': int(above_water_region.shape[0]),
            'avg_brightness': float(above_brightness),
            'avg_blue': float(above_blue),
            'avg_green': float(above_green),
            'avg_red': float(above_red),
            'blue_deviation': float(above_blue_deviation)
        },
        'underwater_region': {
            'row_range': f'{water_surface_row} to {h}',
            'row_count': int(underwater_region.shape[0]),
            'avg_brightness': float(underwater_brightness),
            'avg_blue': float(underwater_blue),
            'avg_green': float(underwater_green),
            'avg_red': float(underwater_red),
            'blue_deviation': float(underwater_blue_deviation)
        }
    }

    print(f"\n{'='*60}")
    print(f"分析完成")
    print(f"{'='*60}")
    print(f"输出文件:")
    print(f"  1. underwater_brightness_profile.png - 亮度分布图")
    print(f"  2. underwater_frame_marked.png - 标注水面位置的帧")
    print(f"\n旋转值: {rotation_value:.1f} 像素")
    print(f"摄像头是否反转: {is_reversed}")

    return results

if __name__ == '__main__':
    video_path = r'd:\myflie\ALL_CODE\swimming\video_merge\video2_min\DJI_20260515181343_0009_D_min.mp4'
    results = analyze_underwater_frame(video_path, timestamp_sec=5)

    import json
    print("\n" + "="*60)
    print("完整结果 (JSON)")
    print("="*60)
    # 转换 numpy 类型为 Python 原生类型
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    results_native = convert_to_native(results)
    print(json.dumps(results_native, indent=2, ensure_ascii=False))
