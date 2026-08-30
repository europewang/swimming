把 Ultralytics 导出的 pose 模型重命名为 `yolo_pose.tflite` 后放到这个目录。

推荐先用轻量模型验证链路，例如：

- `yolo11n-pose.pt` 导出为 TFLite
- 或其他 Ultralytics pose 模型导出的 `.tflite`

示例导出命令：

```bash
yolo export model=yolo11n-pose.pt format=tflite imgsz=640
```

注意：

- 这个项目当前按 17 个人体关键点解析，适配 COCO pose 常见输出。
- 代码兼容 `[1,56,N]` 和 `[1,N,56]` 两种常见 TFLite 输出布局。
- 第一次建议先走 CPU/GPU 自动择优，不要一开始就强依赖 GPU。
