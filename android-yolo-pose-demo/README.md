# Android YOLO Pose Demo

一个独立的原生 Android Kotlin 项目骨架，用于接入 Ultralytics 导出的 YOLO Pose TFLite 模型，实现：

- CameraX 实时相机预览
- 人体框识别
- 17 个关键点解析
- 骨骼连线 Overlay 绘制
- 实时 FPS 显示

## 已做内容

- 单独目录，不影响现有工程
- Gradle 仓库优先使用国内镜像
- Kotlin + ViewBinding + CameraX 基础链路
- TensorFlow Lite 推理入口
- Ultralytics Pose 两种常见输出布局解析

## 国内镜像

项目里已经优先配置了：

- 阿里云 Maven 镜像
- 腾讯云 Gradle 分发镜像

## 模型接入

项目里现在已经放入可直接运行的模型：

`app/src/main/assets/models/yolo_pose.tflite`

来源模型：

`yolo11n-pose.pt -> yolo11n-pose_float32.tflite`

如果后续你想换成自己训练的 pose 模型，把导出的 `.tflite` 模型放到：

`app/src/main/assets/models/yolo_pose.tflite`

参考导出命令：

```bash
yolo export model=yolo11n-pose.pt format=tflite imgsz=640
```

或者直接复用我补好的脚本：

```powershell
.\scripts\export_pose_tflite.ps1 -Model "D:\path\to\your-pose-model.pt"
```

## 当前限制

- 当前只解析单人最高置信度结果，适合先跑通链路
- 当前骨骼连接规则按 COCO 17 点写死
- 没有加入 NMS、多人姿态管理、前后摄像头切换和录制功能
- 为了稳定导出 TFLite，项目目录下额外放了一套独立 Python 3.11 导出环境在 `tools/py311-export`

## 下一步建议

1. 放入真实的 `yolo_pose.tflite`
2. 在 Android Studio 打开项目，等待同步
3. 真机运行，先验证权限、预览、关键点输出
4. 再补多人姿态、稳定追踪、游泳场景定制
