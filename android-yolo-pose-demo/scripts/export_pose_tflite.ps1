param(
    [string]$Model = "D:\myflie\ALL_CODE\swimming\yolo11n-pose.pt",
    [string]$ProjectRoot = "D:\myflie\ALL_CODE\swimming\android-yolo-pose-demo",
    [int]$ImgSize = 640
)

$python = Join-Path $ProjectRoot "tools\py311-export\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python export environment not found: $python"
}

if (-not (Test-Path $Model)) {
    throw "Model not found: $Model"
}

$script = @"
from ultralytics import YOLO
from pathlib import Path
import shutil

model_path = Path(r"$Model")
project_root = Path(r"$ProjectRoot")
imgsz = $ImgSize

model = YOLO(str(model_path))
result = model.export(format="tflite", imgsz=imgsz)

result_path = Path(result)
target = project_root / "app" / "src" / "main" / "assets" / "models" / "yolo_pose.tflite"
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(result_path, target)
print(target)
"@

& $python -c $script
