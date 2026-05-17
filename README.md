# 游泳双机位自动对齐原型

这个原型脚本解决两件事：

1. 先估计水上视频和水下视频的整体时间差。
2. 每隔 5 秒做一次局部微调，再把偏移曲线平滑化，最后输出一个上下拼接的人身融合视频。

## 适用场景

- 水上视角只能断续看到头部、肩部、手部。
- 水下视角有明显反射，可能出现上下对称的“第二个人”。
- 两个视频不是严格同时开始，且中间可能有轻微漂移。

## 方案思路

### 1. 粗对齐

脚本先低帧率采样两个视频，提取“运动签名”：

- 水平方向运动分布
- 垂直方向运动分布
- 主运动区域中心点
- 运动能量

然后用互相关估计整体时间差。

### 2. 每 5 秒做局部微调

以 `5s` 为一个锚点，在锚点前后取一个短窗口，在一个小范围内搜索最相似的偏移量。

这样可以处理：

- 两个视频录制时的微小时钟漂移
- 某些时刻水上只看到手、头，导致全局对齐不够稳的问题

### 3. 水下反射抑制

水下画面常见的问题是上半部分是水面的镜像人影。脚本在水下视角里会：

- 降低上半区域权重
- 连通域里优先保留更靠下的主目标

这不是完美的人体分割，但对“保住真实身体、压掉镜像”已经够实用。

### 4. 自动摆正与有效片段裁剪

很多原始素材在真正下水前后会有一段无关片段，比如：

- 手持自拍
- 出水后的岸上画面
- 颜色明显不是泳池蓝色调的过渡镜头

脚本现在会先做两步预处理：

- 自动检测视频是否需要旋转 `180°`
- 自动找出“稳定泳池蓝色段”的起止时间，只在这段上做后续对齐和融合

这样可以避免把前后无关片段拿去做时间对齐，也能减少视频方向颠倒导致的误判。

### 5. 上下拼接

输出视频默认：

- 水线上方主要来自水上视角
- 水线下方主要来自水下视角
- 水线附近做一个平滑过渡
- 人体区域优先覆盖，尽量把身体拼成一个连续的人

## 安装

当前脚本依赖：

- `numpy`
- `scipy`
- `imageio`
- `imageio-ffmpeg`
- `opencv-python-headless`

如果缺少 `opencv-python-headless`，执行：

```powershell
pip install opencv-python-headless
```

如果你要批量做“无损压缩到新目录”，还需要：

```powershell
pip install imageio-ffmpeg
```

## 使用

### 无损批量压缩单个文件夹

如果你的目标是把一个文件夹里的所有视频，在保证视频帧无损的前提下尽量压到更小，并输出到一个新的文件夹，可以直接用：

```powershell
python -X utf8 .\lossless_video_minimizer.py .\video
```

默认行为：

- 会在 `video` 同级生成新目录 `video_min`
- 每个输出文件按 `原文件名_min` 命名，并自动选择最合适的扩展名
- 会把“原样复制源文件”也作为候选之一，避免无损重编码反而更大
- 只使用无损视频方案
- 默认逐帧校验输出视频是否和输入完全一致
- 默认原样复制音频和字幕流，不重新编码

如果你希望递归处理子目录：

```powershell
python -X utf8 .\lossless_video_minimizer.py .\video --recursive
```

如果你想覆盖已经生成的结果：

```powershell
python -X utf8 .\lossless_video_minimizer.py .\video --overwrite
```

说明：

- 脚本会尝试多种无损方案，例如 `libx265 lossless`、`libx264 qp=0`、`ffv1`
- 还会把“直接复制原文件”纳入比较，然后自动保留当前环境下体积最小的那一个结果
- “绝对全局最小”无法通过有限次编码穷举保证，但输出本身会保持无损
- 如果某个无损重编码候选更优，通常会输出成 `.mkv`

自动识别机位时：

```powershell
python -X utf8 .\swim_video_sync.py `
  --video-a "path\to\video_a.mp4" `
  --video-b "path\to\video_b.mp4" `
  --output-dir .\outputs
```

现在这条命令默认还会自动完成：

- 判断谁是水上、谁是水下
- 检测视频是否需要旋转 `180°`
- 裁掉前后明显不属于泳池蓝色稳定段的无效片段

批量处理 `video` 根目录下的两个相机文件夹时：

```powershell
python -X utf8 .\swim_video_sync.py `
  --batch-video-root .\video
```

脚本会：

- 自动扫描 `video` 下的两个相机子目录
- 根据文件名时间戳按“最近时间”自动配对
- 自动识别每组里的水上 / 水下视频
- 自动摆正每段视频方向
- 自动裁掉前后不适合拼接的无效片段
- 直接把每一组的融合视频和报告放到 `video\融合`

如果你的文件名是 GoPro / DJI 这种带前缀的格式，例如：

- `20210627_1231.mp4`
- `DJI_20260513180601_0002_D_min.mp4`

脚本现在都能自动解析时间并做配对。

如果你已经明确知道谁是水上、谁是水下，也可以显式指定：

```powershell
python -X utf8 .\swim_video_sync.py `
  --top-video "path\to\top.mp4" `
  --bottom-video "path\to\underwater.mp4" `
  --output-dir .\outputs
```

如果你暂时只想看时间对齐结果，不输出融合视频：

```powershell
python -X utf8 .\swim_video_sync.py `
  --video-a "path\to\video_a.mp4" `
  --video-b "path\to\video_b.mp4" `
  --output-dir .\outputs `
  --skip-fuse
```

如果你在批处理模式下加了 `--skip-fuse`，脚本只会生成配对结果和 `alignment_report`，不会生成 `.mp4` 视频。

要真正生成融合视频，请不要加 `--skip-fuse`。

### 普通批处理

适合先快速看结果：

```powershell
python -X utf8 .\swim_video_sync.py `
  --batch-video-root .\video `
  --batch-workers 2 `
  --output-fps 15 `
  --max-output-width 1280
```

特点：

- 速度更快
- 分辨率会被压到 `1280` 宽
- 更适合先做粗看和筛查

默认情况下，`YOLO` 会按每个输出帧都尝试调用一次，也就是：

- `--yolo-interval 1`

这是质量优先的设置。

### 高分辨率并行批处理

适合你现在这种“希望高分辨率，并且并行处理”的场景：

```powershell
python -X utf8 .\swim_video_sync.py `
  --batch-video-root .\video `
  --batch-workers 2 `
  --segmentation-device cpu `
  --output-fps 30 `
  --max-output-width 3812 `
  --yolo-interval 2
```

说明：

- `--max-output-width 3812`
  基本保持你当前素材的原始宽度，不主动压缩分辨率。

- `--batch-workers 2`
  两组视频并行处理。

- `--segmentation-device cpu`
  高分辨率并行时建议先明确用 `cpu`，这样两个进程可以同时跑；如果改成 `cuda:0`，脚本会自动退回单进程，反而不并行。

- `--output-fps 30`
  保持更自然的视频流畅度。

- `--yolo-interval 2`
  表示 YOLO 每隔 1 帧才真正调用一次，中间帧复用上一轮结果。这样通常会更快，但人体边缘稳定性可能略下降。

如果你想强制重新生成已经存在的视频，再加：

```powershell
--overwrite-existing
```

如果你想提高速度，优先看这几个参数：

- `--batch-workers 2`
  批处理并行数。当前若人体分割仍跑在 CPU，可以设成 `2` 到 `4` 试；如果后面切到 CUDA，脚本会自动退回 `1`，避免多个进程抢同一张显卡。

- `--segmentation-device auto`
  人体分割设备，默认自动选择。当前环境如果是 CPU 版 `torch`，这里会落到 `cpu`；如果你后面安装了 CUDA 版 `torch`，它会自动切到 `cuda:0`。

- `--segmentation-device cpu`
  如果你要“高分辨率 + 并行”，这个参数通常更实用，因为它允许多个 worker 同时工作。

- `--disable-gpu-encode`
  默认不加时，脚本会优先尝试 `h264_nvenc` 做硬件编码；只有在你想强制退回软件编码时才需要加这个参数。

- `--yolo-interval 1`
  默认值，表示每一帧都尝试跑 YOLO，质量优先。

- `--yolo-interval 2`
  常用提速设置，表示每隔 1 帧跑一次 YOLO，中间帧复用。

- `--yolo-interval 3` 或 `4`
  更激进的提速设置，适合先做批量粗处理，但人物边缘可能更容易抖动或滞后。

## 关键参数

- `--anchor-step-sec 5`
  默认每 5 秒做一次局部对齐。

- `--window-sec 2`
  每个局部对齐窗口长度，默认 2 秒。

- `--search-radius-sec 1.2`
  每个锚点在全局偏移周围搜索的范围。

- `--waterline-ratio 0.44`
  水线位置，表示画面高度的比例。不同拍摄机位通常需要你手动微调。

- `--blend-px 90`
  水线上下融合带宽。

- `alignment_report.json` 里的 `top_preprocess` / `bottom_preprocess`
  会记录每个视频最终实际采用的裁剪起止时间，以及是否自动做了 `180°` 旋转。

- `--max-output-width 1280`
  先缩小到这个宽度再融合，适合 4K 素材做快速测试。

- `--max-pair-delta-minutes 10`
  批处理配对时允许的最大时间差，单位分钟。

- `--output-fps 15`
  输出视频帧率，越高越流畅，但处理更慢。

- `--disable-person-segmentation`
  禁用 YOLO 人体分割，仅使用传统图像法。一般不建议默认加。

## 性能说明

- 当前脚本已经支持批处理并行。
- 当前脚本已经支持优先使用 `NVENC` 做硬件视频编码。
- 当前脚本已经默认启用“人体附近 ROI 分割”优化：先用便宜的粗掩码定位人体，再只在人体附近区域跑更贵的分割。
- 当前脚本已经支持降低 YOLO 调用频率，通过 `--yolo-interval` 控制。
- 如果想让 `YOLO` 人体分割也走 GPU，需要安装 CUDA 版 `torch`；如果安装的是 CPU 版 `torch`，即使机器里有 NVIDIA 显卡，分割仍然会跑在 CPU 上。

## 输出内容

- `outputs/alignment_report.json`
  包含自动机位识别结果、全局时间差、每个 5 秒锚点的局部偏移、平滑后的偏移曲线。

- `outputs/fused_swim.mp4`
  上下融合后的结果视频。

批处理时还会额外输出：

- `video\融合\batch_summary.json`
  记录整批配对和输出结果。

- `video\融合\<配对名>.mp4`
  每组视频的融合结果，直接平铺在 `融合` 文件夹里。

- `video\融合\<配对名>_alignment_report.json`
  每组视频的时间对齐报告，也直接平铺在 `融合` 文件夹里。

## 你下一步最值得做的两件事

1. 准备真正的两路视频，优先直接用 `--video-a` 和 `--video-b` 让脚本自动识别机位。
2. 先运行 `--skip-fuse` 看 `alignment_report.json` 是否合理，再微调 `--waterline-ratio` 和 `--search-radius-sec`。

## 局限

- 这是一版工程原型，不是训练过的人体分割模型。
- 如果水上视角几乎长期看不到人体，只能看到零星水花，那么最好再加一个手动初始偏移。
- 如果你后面想把“人体完整拼出来”做得更稳，下一步建议加 `YOLOv8-seg` 或 `SAM2` 做人像掩码，再把当前这套时间对齐逻辑保留下来。
