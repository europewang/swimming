# swimming 项目总览

这个仓库目前主要分成 3 部分：

## 1. Android App

- 目录：`android-yolo-pose-demo/`
- 作用：手机端应用
- 当前用途：
  - 登录后连接后端
  - 接收 `ESP32-CAM` 视频流
  - 调用本地识别逻辑进行人物/颜色相关识别
  - 负责页面交互、录制、设备连接状态展示

## 2. 硬件端固件代码

- 目录：`ESP32_CAM_Phone_App/`
- 作用：烧录到 `ESP32-CAM` 开发板上的固件
- 当前用途：
  - BLE 配网
  - 连接手机热点
  - 启动摄像头服务
  - 把本次设备 IP 上报到后端
  - 向 Android App 提供抓拍和视频流接口

## 3. 之前的视频拼接 / 处理代码

- 主要文件：
  - `swim_video_sync.py`
  - `swimmer_following_system.py`
  - `swim_video_v2.py`
  - `swim_video_v3.py`
  - `lossless_video_minimizer.py`
- 作用：历史上的视频分析、双机位拼接、视频处理和跟随控制原型脚本
- 当前用途：
  - 处理水上/水下双机位视频
  - 做时间同步、拼接、融合
  - 做历史版本算法验证和脚本实验

## 其他目录说明

- `CameraWebServer/`
  - 早期相机服务相关代码或参考代码

- `androidapp/`
  - 旧的 Android 相关目录，当前主开发目录不是这里

- `video/`、`video_merge/`
  - 测试视频、输出视频或中间处理结果目录

## 当前建议关注的目录

如果你现在主要在继续做项目，优先看这两个目录：

- `android-yolo-pose-demo/`
- `ESP32_CAM_Phone_App/`
