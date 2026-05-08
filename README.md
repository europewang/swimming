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

### 4. 上下拼接

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

## 使用

自动识别机位时：

```powershell
python -X utf8 .\swim_video_sync.py `
  --video-a "path\to\video_a.mp4" `
  --video-b "path\to\video_b.mp4" `
  --output-dir .\outputs
```

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

- `--max-output-width 1280`
  先缩小到这个宽度再融合，适合 4K 素材做快速测试。

## 输出内容

- `outputs/alignment_report.json`
  包含自动机位识别结果、全局时间差、每个 5 秒锚点的局部偏移、平滑后的偏移曲线。

- `outputs/fused_swim.mp4`
  上下融合后的结果视频。

## 你下一步最值得做的两件事

1. 准备真正的两路视频，优先直接用 `--video-a` 和 `--video-b` 让脚本自动识别机位。
2. 先运行 `--skip-fuse` 看 `alignment_report.json` 是否合理，再微调 `--waterline-ratio` 和 `--search-radius-sec`。

## 局限

- 这是一版工程原型，不是训练过的人体分割模型。
- 如果水上视角几乎长期看不到人体，只能看到零星水花，那么最好再加一个手动初始偏移。
- 如果你后面想把“人体完整拼出来”做得更稳，下一步建议加 `YOLOv8-seg` 或 `SAM2` 做人像掩码，再把当前这套时间对齐逻辑保留下来。
