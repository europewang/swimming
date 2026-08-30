# 当前项目代码思路详解

本文档基于当前仓库代码状态编写，目标是把这个 Android 项目“现在到底是怎么工作的”讲清楚，尽量做到：

- 从用户点击开始，到相机采样、模型推理、目标跟踪、界面绘制，按真实执行顺序说明
- 区分两条核心模式：
  - `自动识别`
  - `选择身体颜色`
- 解释当前代码里保留下来的“人物骨架逻辑”和“颜色跟踪逻辑”分别承担什么角色
- 解释当前项目为什么会出现某些现象，比如：
  - 水下颜色漂移
  - 快速移动时容易丢失
  - 自动识别和颜色识别行为不完全一样
- 给后续继续改造的人一个比较完整的认知地图

---

## 1. 项目目标

这个项目当前可以理解成一个“以泳者为目标的双模式跟踪 Demo”：

1. `自动识别` 模式
   - 先依赖 YOLO Pose 模型识别人和关键点
   - 再结合颜色信息，提取人体框、肤色区域、泳衣区域
   - 当骨架暂时不稳定时，尝试用颜色或历史记忆短时续跟踪

2. `选择身体颜色` 模式
   - 不先做人检测
   - 用户直接点击画面中的某种颜色
   - 系统记录该颜色
   - 后续只按该颜色寻找连通区域并持续跟踪
   - 这是更纯粹的“颜色主跟踪”路线

换句话说，当前项目不是一个单一算法，而是：

- 一条“人物识别主导”的路径
- 一条“颜色锁定主导”的路径
- 二者共用同一个 `PersonTracker`

---

## 2. 整体结构

核心代码主要集中在下面几个文件：

- [MainActivity.kt](/D:/myflie/ALL_CODE/swimming/android-yolo-pose-demo/app/src/main/java/com/example/yolopose/MainActivity.kt)
- [CameraFrameAnalyzer.kt](/D:/myflie/ALL_CODE/swimming/android-yolo-pose-demo/app/src/main/java/com/example/yolopose/camera/CameraFrameAnalyzer.kt)
- [YoloPoseDetector.kt](/D:/myflie/ALL_CODE/swimming/android-yolo-pose-demo/app/src/main/java/com/example/yolopose/ml/YoloPoseDetector.kt)
- [PersonTracker.kt](/D:/myflie/ALL_CODE/swimming/android-yolo-pose-demo/app/src/main/java/com/example/yolopose/ml/PersonTracker.kt)
- [TrackingModels.kt](/D:/myflie/ALL_CODE/swimming/android-yolo-pose-demo/app/src/main/java/com/example/yolopose/ml/TrackingModels.kt)
- [PoseOverlayView.kt](/D:/myflie/ALL_CODE/swimming/android-yolo-pose-demo/app/src/main/java/com/example/yolopose/ui/PoseOverlayView.kt)
- [PoseResult.kt](/D:/myflie/ALL_CODE/swimming/android-yolo-pose-demo/app/src/main/java/com/example/yolopose/ml/PoseResult.kt)

可以把它们理解成四层：

1. 页面和交互层
   - `MainActivity`
   - 负责按钮、返回、点击画面、状态文案、启动相机

2. 图像采集层
   - `CameraFrameAnalyzer`
   - 负责把 CameraX 的 `ImageProxy` 转成 `Bitmap`

3. 感知与跟踪层
   - `YoloPoseDetector`
   - `PersonTracker`
   - 前者负责模型推理，后者负责业务逻辑和跟踪状态机

4. 可视化绘制层
   - `PoseOverlayView`
   - 负责把框、色块、关键点、脚腿辅助线等画到屏幕上

---

## 3. 页面流程

### 3.1 主页模式

`MainActivity` 中定义了一个内部枚举：

- `HOME`
- `AUTO_SELECT`
- `COLOR_SELECT`

用户进入页面后，默认是 `HOME`。

主页只有两个主要按钮：

1. `自动识别`
2. `选择身体颜色`

这个设计非常重要，因为项目当前是“先选模式，再点画面”，而不是一开始就启动复杂交互。

### 3.2 进入识别页

点击其中一个按钮后，调用：

- `enterRecognitionMode(ScreenMode.AUTO_SELECT)`
- 或 `enterRecognitionMode(ScreenMode.COLOR_SELECT)`

这个函数会做几件事：

1. 记录当前模式
2. 把跟踪器 `personTracker.reset()`
3. 清空选中点和叠加层
4. 设置提示文案
5. 显示相机预览页
6. 检查相机权限
7. 启动相机

### 3.3 识别页交互

识别页本身没有额外的“开始识别”按钮。

当前设计是：

- 用户进入识别页后
- 直接点击预览画面中的某个位置
- 系统把这个点击位置映射回原始视频帧坐标
- 再根据当前模式走不同逻辑

这部分逻辑在 `handleTargetTap()`。

---

## 4. 相机到位图的链路

### 4.1 CameraX 配置

`startCamera()` 里使用了：

- `Preview`
- `ImageAnalysis`

其中 `ImageAnalysis` 的设置比较关键：

- 目标分辨率：`480 x 360`
- 背压策略：`STRATEGY_KEEP_ONLY_LATEST`

这意味着：

1. 分析分辨率被限制在较低级别，目的是加快处理
2. 如果处理跟不上，相机不会把所有帧都压进队列，而是只保留最新一帧

这是一种典型的实时视觉策略，牺牲部分帧完整性，换取更低延迟。

### 4.2 YUV 转 Bitmap

`CameraFrameAnalyzer` 做的事情是：

1. 接收 `ImageProxy`
2. 判断格式是否为 `YUV_420_888`
3. 用 `yuv420ToArgb()` 手工转成 ARGB 数组
4. 写入复用的 `Bitmap`
5. 按相机旋转角度做旋转
6. 回调给 `MainActivity.analyzeFrame(bitmap)`

这里有两个值得注意的点：

#### 4.2.1 复用 Bitmap

`lastBitmap` 会被尽量复用，避免每帧重新分配大内存。

这能减少：

- GC 压力
- 帧间卡顿

#### 4.2.2 旋转在这里完成

相机原始帧和屏幕方向可能不同，所以这里先做旋转，让后面的识别逻辑拿到的是“方向正确”的图像。

---

## 5. YOLO Pose 模型层

模型封装在 [YoloPoseDetector.kt](/D:/myflie/ALL_CODE/swimming/android-yolo-pose-demo/app/src/main/java/com/example/yolopose/ml/YoloPoseDetector.kt)。

### 5.1 模型职责

它只负责一件事：

- 输入 `Bitmap`
- 输出一个 `PoseResult?`

它不关心业务状态，不关心是不是水下，不关心是不是跟踪同一个人。

它只负责“当前这一帧模型看到了什么”。

### 5.2 推理流程

`estimatePose(bitmap)` 做的事：

1. 把输入图缩放到模型输入尺寸 `inputSize x inputSize`
2. 转成 float 类型 RGB buffer
3. 调用 TFLite `interpreter.run()`
4. 解码输出张量
5. 选出置信度最高的候选人

### 5.3 输出结构

当前模型输出被解码成：

- `boundingBox`
- `confidence`
- `17` 个关键点

关键点结构定义在 [PoseResult.kt](/D:/myflie/ALL_CODE/swimming/android-yolo-pose-demo/app/src/main/java/com/example/yolopose/ml/PoseResult.kt)：

- `KeyPoint(x, y, confidence)`
- `PoseResult(boundingBox, confidence, keyPoints)`

### 5.4 当前模型层的特点

当前模型层有几个明显特点：

1. 只取置信度最高的一个人
   - 当前不是多目标跟踪架构
   - 更像“单主目标 Demo”

2. 没有额外 NMS 管理层
   - 解码逻辑偏轻量
   - 更强调单人主目标

3. 线程数固定为 4
   - `Interpreter.Options().setNumThreads(4)`
   - 这是性能和设备兼容的折中

---

## 6. `PersonTracker` 是整个系统的大脑

如果说 `YoloPoseDetector` 只是在“看一帧”，那么 [PersonTracker.kt](/D:/myflie/ALL_CODE/swimming/android-yolo-pose-demo/app/src/main/java/com/example/yolopose/ml/PersonTracker.kt) 才是在“理解连续视频”。

它负责：

- 当前模式状态
- 用户点击点
- 初始化阶段
- 模板确认
- 历史跟踪结果
- 颜色模板
- 骨架主跟踪
- 颜色主跟踪
- 丢失后的短时记忆

可以把它理解成一个“单目标状态机 + 颜色提取器 + 轻量时序跟踪器”。

---

## 7. `PersonTracker` 的核心状态

下面这些成员变量决定了它的行为：

### 7.1 历史结果

- `previousTracked`
- `lastTracked`
- `lastUpdatedAtMs`

作用：

- 用于平滑
- 用于预测下一帧可能位置
- 用于丢失后短时延续

### 7.2 模板

- `template`
- `templateConfirmed`

这里的模板不完全等于“颜色模板”，它其实是一个混合模板。

在自动识别模式里，它可能包含：

- 平均颜色
- 锚点颜色
- 肤色
- 泳衣色
- 泳衣调色板
- 泳衣主色相范围
- 目标平均宽高

在颜色识别模式里，当前代码会复用同一个数据结构，但只真正依赖其中的“直接颜色锁定”相关部分。

### 7.3 选择点

- `selectedPoint`

作用：

- 记录用户点击的目标位置
- 帮助从多个可能区域里选出“更接近用户点击”的那个

### 7.4 初始化相关

- `colorPickPending`
- `initializing`
- `initStartAtMs`
- 一组 `init...Colors` / `initWidths` / `initHeights`

作用：

- 管理初始化阶段
- 在 10 秒内收集样本
- 生成模板

---

## 8. 两条主流程

当前最重要的是理解，项目内部其实有两条主流程。

---

## 9. 流程一：自动识别模式

### 9.1 入口

在 `MainActivity.handleTargetTap()` 中：

- `AUTO_SELECT` 模式下
- 调用 `personTracker.setSelectionPoint(...)`
- 再调用 `personTracker.startInitialization(...)`

这表示：

- 用户点击的不是“颜色”
- 而是“我希望锁定这个位置附近的人”

### 9.2 初始化阶段

初始化阶段持续 10 秒，`INIT_SECONDS = 10`。

在这 10 秒里，程序会尽量持续拿到稳定的人体检测和颜色样本。

`PersonTracker.update()` 中，如果此时模型成功检测到人：

1. `resolveMode(detectedPose)`
   - 判断当前骨架属于全身、腿部、脚部还是丢失

2. `smoothPose()`
   - 用上一帧结果做平滑

3. `deriveTrackingBox()`
   - 根据关键点推导一个更适合业务的跟踪框

4. `matchesSelectedPoint()`
   - 检查这个框是不是包含用户点击区域
   - 防止选错人

5. `buildColorCorrectionProfile()`
   - 生成一个局部颜色校正参数

6. `extractAverageColor()` / `extractAnchorColor()`
   - 提取人体框的整体颜色特征

7. `buildPoseAwareSkinHintBox()`
   - 根据肩、髋、膝等关键点推导肤色候选区域

8. `buildPoseAwareSwimwearHintBox()`
   - 根据髋部、膝部等位置推导泳衣候选区域

9. `detectColorRegion()`
   - 在上面两个候选框里找更接近目标颜色的实际连通区域

10. `collectTemplateSample()`
   - 持续收集颜色和尺寸样本

### 9.3 模板生成

10 秒后，`finishInitialization()` 会尝试生成模板。

自动识别模式下，这个模板包含：

- 人体平均颜色
- 锚点颜色
- 肤色
- 泳衣色
- 泳衣调色板
- 泳衣主色相及范围
- 目标平均宽高

然后界面会弹确认框：

- 是否固定后续识别模板

确认后：

- `templateConfirmed = true`

### 9.4 锁定后跟踪

模板确认后，`update()` 会优先进入：

- `updateWithLockedTemplate()`

这条逻辑的核心思想是：

1. 不一定每帧都完全依赖 YOLO 骨架
2. 会在一个预测搜索窗口内
3. 优先找符合“泳衣颜色模板”的区域
4. 再配合肤色区域做组合
5. 输出一个更稳定的组合框

因此，自动识别模式的本质不是“纯骨架”，而是：

**骨架初始化 + 颜色辅助锁定 + 局部区域跟踪**

---

## 10. 流程二：选择身体颜色模式

### 10.1 入口

在 `MainActivity.handleTargetTap()` 中：

- `COLOR_SELECT` 模式下
- 调用 `personTracker.startDirectColorLock(...)`

这意味着此模式不是选“这个人”，而是选“这个颜色”。

### 10.2 为什么这个模式会跳过 YOLO

`analyzeFrame()` 中有一个判断：

- 如果当前是 `COLOR_SELECT`
- 且 `personTracker.shouldSkipPoseInference() == true`
- 那么 `result = null`

也就是说，在颜色锁定已经开始后，这条路径会直接跳过模型推理。

这就是当前颜色模式和自动模式最核心的区别：

- 自动模式：以 YOLO Pose 为主，颜色为辅
- 颜色模式：以颜色为主，YOLO 可以完全不参与

### 10.3 颜色初始化

`startDirectColorLock()` 会：

1. `reset()`
2. 保存点击点
3. 设置 `colorPickPending = true`

随后 `update()` 会进入：

- `initializeDirectColorTemplate()`

### 10.4 当前颜色初始化逻辑

这是当前颜色模式最重要的部分。

它现在的思路是：

1. 整帧先做一个颜色校正
2. 估计整帧水背景颜色 `waterColor`
3. 以点击点为中心，构造一个小取样框 `sampleBox`
4. 取出这个小框的平均颜色 `sampledColor`
5. 在整幅图上调用 `detectColorRegion()`
   - 参考色就是 `sampledColor`
   - 不用颜色族
   - 不用独立块分割优先筛选
   - 只找接近这个颜色的连通区域
6. 同时排除接近大面积水背景的颜色
7. 找到后立即生成一个“直接颜色锁定模板”

注意：

这里虽然模板结构仍然叫 `ColorTrackingTemplate`，还保留了很多字段，但对当前颜色模式来说，真正有用的是：

- `swimwearColor`
- `meanWidth`
- `meanHeight`
- `directColorLock = true`

而像：

- `swimwearPalette`
- `swimwearHue`
- `swimwearHueTolerance`

在当前“纯颜色识别”回退版本里，已经不再作为主匹配逻辑使用。

### 10.5 颜色持续跟踪

颜色模板建立后，后续帧会走：

- `updateWithDirectColorTemplate()`

其思路是：

1. 根据历史框做一个预测搜索窗口 `buildPredictedSearchWindow()`
2. 在这个搜索窗口中重新做颜色校正
3. 估算局部水背景颜色
4. 在这个窗口里调用 `detectColorRegion()`
5. 用“用户选中的颜色”找连通区域
6. 用 `isMotionPlausible()` 检查移动是否合理
7. 用 `stabilizeTrackingBox()` 平滑框位置
8. 输出新的颜色跟踪结果

如果局部窗口里找不到：

- 会每隔一小段时间做一次全局找回 `findGlobalDirectColorRegion()`

如果还找不到：

- 会短暂返回 `TRACKED_MEMORY`
- 再不行就 `LOST`

所以颜色模式现在不是“简单取颜色就完事”，而是：

**颜色取样 + 局部连通区域跟踪 + 预测窗口 + 全局找回 + 短时记忆**

---

## 11. `detectColorRegion()` 是颜色相关最核心的函数

无论自动模式还是颜色模式，真正找“颜色区域”的底层核心都在这个函数里。

它的思路可以概括成：

1. 在某个搜索框中划一个网格
2. 每个网格单元算一个平均颜色
3. 判断这个网格是否匹配参考颜色
4. 把匹配的格子标记成 `true`
5. 在布尔网格中找连通块
6. 给每个连通块打分
7. 选择最优的那个
8. 输出：
   - 外接框 `box`
   - 简化轮廓 `contour`
   - 组成这个区域的网格 `cells`

这意味着当前代码里的“颜色区域”不是按像素级精确分割得到的，而是：

**按低分辨率网格进行近似连通区域检测**

这样做的好处：

- 算得快
- 实时性更容易保证

坏处：

- 边缘不够精细
- 遇到倒影、波纹、强折射时容易碎裂

---

## 12. 颜色匹配不是单一公式

颜色匹配函数是 `colorMatchesReference()`。

它不是一刀切，而是分情况：

### 12.1 深色

如果参考色很暗，比如黑色泳裤：

- 用 `darkColorDistance()`

因为这类颜色在 HSV 下很容易不稳定，亮度变化更敏感。

### 12.2 低饱和色

如果颜色本身饱和度低：

- 用 `neutralColorDistance()`

这比单纯靠色相更稳。

### 12.3 普通彩色

如果颜色不是太暗、也不是低饱和：

- 用 `hsvDistance()`

这更适合彩色泳衣。

这说明当前代码其实已经在做一个实用化折中：

- 深色、灰色、彩色分别处理

---

## 13. 水下颜色校正是怎么做的

`buildColorCorrectionProfile()` 的作用不是高级水下复原，而是一个轻量的局部白平衡近似。

逻辑是：

1. 在目标区域取样
2. 计算平均 `R/G/B`
3. 用三通道平均值作为灰参考
4. 反推出三个增益：
   - `redGain`
   - `greenGain`
   - `blueGain`

然后 `correctPixel()` 会用这些增益修正像素。

它解决的问题主要是：

- 水下整体偏蓝
- 红色衰减严重
- 同一色块在不同位置亮度不同

但要明确：

这不是物理级水下去雾算法，也不是专业色彩恢复。

它只是一个：

**局部、轻量、实时优先的颜色归一化步骤**

---

## 14. 水背景过滤是怎么做的

水下最容易误识别的就是整池蓝色背景。

当前代码里主要有两层过滤：

### 14.1 背景排除颜色

`estimateWaterBackgroundColor()` 会估计水背景主色。

后续在 `detectColorRegion()` 中：

- 如果某个网格颜色太接近这个背景色
- 就直接排除

### 14.2 `looksLikeWater()`

这个函数会额外判断：

- 当前格子是不是偏蓝
- 并且是不是比参考目标色更接近水背景色

如果是，就排除。

这让颜色模式不至于刚点了紫色、黑色、红色，结果大面积追到泳池蓝色背景上去。

---

## 15. 连通块为什么还能“断裂”

你之前一直觉得“明明是一个整体颜色，为什么识别不好”，本质上跟这里有关。

当前连通块构建是这样做的：

1. 先判断网格格子是否匹配目标色
2. 再判断两个相邻格子能不能连起来 `canCellsConnect()`

连接条件里既看：

- 颜色差
- 也看亮度差

也就是说，即使两个格子都“像目标色”，如果相邻格子的亮度/颜色差过大，也可能不会被连成一个块。

这在水下很常见，因为：

- 波纹反光
- 阴影
- 折射
- 水面倒影

都会让同一块衣服看起来像多个小块。

所以当前代码的本质不是“精确分割衣物轮廓”，而是：

**从低分辨率网格里找一个尽量连贯、尽量接近历史位置的目标颜色区域**

---

## 16. 为什么快速移动时容易丢

当前代码虽然有“预测”和“找回”，但快速移动仍然容易丢，原因主要有四层。

### 16.1 分析分辨率低

分析输入只有 `480 x 360`，颜色网格又进一步降采样。

所以目标一旦快速移动：

- 新位置变化会被量化得更粗
- 小区域更容易直接跳格

### 16.2 搜索窗口是基于历史预测的

`buildPredictedSearchWindow()` 会根据上一帧和前一帧中心点做速度预测。

如果目标瞬间偏离预测很多：

- 局部搜索窗口就可能压根没覆盖到真实目标

### 16.3 `isMotionPlausible()` 限制了突变

为了防止跳到别的颜色块上，代码会限制：

- 位置跳变
- 框大小跳变

这能防误匹配，但也会让真实高速移动更容易被拒绝。

### 16.4 全局找回不是每帧都做

全局找回有节流：

- `GLOBAL_RECOVERY_INTERVAL_MS = 360L`

这意味着不是每帧都全图搜索，避免过慢。

优点是省性能。

缺点是高速移动时，重新找回会有时间差。

---

## 17. 跟踪结果有哪些模式

`DetectionMode` 定义在 [TrackingModels.kt](/D:/myflie/ALL_CODE/swimming/android-yolo-pose-demo/app/src/main/java/com/example/yolopose/ml/TrackingModels.kt)：

- `INITIALIZING`
- `COLOR_TRACK`
- `FULL_BODY`
- `LEG_ONLY`
- `FOOT_ONLY`
- `TRACKED_MEMORY`
- `LOST`

这些模式不仅是显示文案，也直接决定叠加层怎么画。

### 17.1 `FULL_BODY`

表示骨架识别到的身体信息比较完整。

### 17.2 `LEG_ONLY`

表示骨架只在腿部比较可靠。

### 17.3 `FOOT_ONLY`

表示更弱，只能用脚点辅助。

### 17.4 `COLOR_TRACK`

表示当前主要依赖颜色区域跟踪。

### 17.5 `TRACKED_MEMORY`

表示这一帧没找到新结果，但短时间内继续沿用上一帧结果，避免框突然消失。

### 17.6 `LOST`

表示目标彻底丢失。

---

## 18. 叠加层绘制思路

[PoseOverlayView.kt](/D:/myflie/ALL_CODE/swimming/android-yolo-pose-demo/app/src/main/java/com/example/yolopose/ui/PoseOverlayView.kt) 负责“把识别结果画出来”。

它现在不是只画一个框，而是按不同模式画不同元素。

### 18.1 公共元素

- 当前平均颜色圆点
- 锚点颜色小方块
- 用户选中十字准星

### 18.2 自动识别模式下

会尽量画：

- 主人体框
- 锚定区域
- 肤色区域
- 泳衣区域
- 腿部高亮
- 脚点

### 18.3 颜色模式下

优先画：

- 颜色主跟踪区域
- 连通网格区域填充
- 轮廓或外框

如果当前 `skinCells` / `swimwearCells` 存在，也会继续画出来。

这就是为什么有时你会感觉：

- 颜色模式虽然已经回退成纯颜色识别
- 但叠加层和文案里仍然带着“肤色区域 / 泳衣区域”的历史痕迹

因为现在的显示层仍然是一个“通用绘制器”，还没有把两个模式完全拆开。

---

## 19. 当前代码的真实特点

如果不用营销语言，而是非常实事求是地总结，这个项目当前是：

### 19.1 优点

1. 已经形成了比较完整的单目标实时链路
   - 相机
   - 模型
   - 跟踪器
   - 叠加层

2. 自动模式和颜色模式都能跑

3. 已经考虑了水下最关键的几个现实问题
   - 偏蓝
   - 低对比
   - 色块漂移
   - 目标短时丢失

4. 颜色模式不是纯暴力逐像素搜索
   - 性能上更可控

### 19.2 局限

1. 颜色模式仍然是网格级近似，不是像素级精细分割

2. 自动模式和颜色模式共用一个 `PersonTracker`
   - 功能强
   - 但复杂度较高
   - 后续维护容易相互影响

3. 模板结构混合了很多历史字段
   - 当前颜色模式已经不用颜色族主逻辑了
   - 但数据结构仍然保留着

4. 当前更像“单主目标 Demo”
   - 不是成熟的多目标水下跟踪系统

5. 没有真正使用专业时序跟踪器
   - 例如 Kalman + appearance embedding
   - 或光流辅助
   - 或分割模型

---

## 20. 如果把当前项目一句话说清楚

一句话版本：

**当前项目是一个以单泳者为主目标的 Android 实时跟踪 Demo，支持“骨架主导的人体识别”和“颜色主导的色块跟踪”两种模式；自动模式依赖 YOLO Pose 初始化并用颜色辅助锁定，颜色模式则绕过人体识别，直接基于用户点击颜色做连通区域跟踪。**

---

## 21. 当前执行顺序总览

为了方便以后查代码，这里再给一个“从点击到画框”的简化时序。

### 21.1 自动识别模式

1. 用户进入 `自动识别`
2. CameraX 开始输出帧
3. 用户点击目标位置
4. 记录 `selectedPoint`
5. 开始 10 秒初始化
6. 每帧调用 YOLO Pose
7. 根据骨架得到人体框
8. 提取平均色、锚点色、肤色、泳衣色
9. 收集样本
10. 10 秒后生成模板
11. 用户确认模板
12. 后续在局部窗口中优先找匹配泳衣色和肤色区域
13. 输出组合框并绘制

### 21.2 颜色识别模式

1. 用户进入 `选择身体颜色`
2. CameraX 开始输出帧
3. 用户点击某个颜色
4. 记录点击点
5. 取点击区域平均颜色
6. 整帧搜索接近该颜色的连通区域
7. 建立直接颜色模板
8. 后续跳过 YOLO 推理
9. 在预测搜索窗口中找同色连通区域
10. 找不到时做全局找回
11. 再找不到时短时延续或丢失
12. 把颜色区域和轮廓画出来

---

## 22. 后续最推荐的重构方向

如果以后还要继续做，这里是最推荐的工程方向。

### 22.1 把 `PersonTracker` 拆成两个跟踪器

建议拆成：

- `PosePersonTracker`
- `DirectColorTracker`

原因：

- 现在一个类里同时承担两种哲学完全不同的跟踪策略
- 逻辑耦合很高

### 22.2 把模板结构拆开

当前 `ColorTrackingTemplate` 同时服务：

- 自动模式
- 颜色模式

建议拆成：

- `PoseLockTemplate`
- `DirectColorTemplate`

### 22.3 把叠加层按模式拆显示

现在 `PoseOverlayView` 还是通用绘制。

建议至少把绘制逻辑拆成：

- 人体模式绘制
- 颜色模式绘制

这样颜色模式就不会继续出现“肤色区域”“泳衣区域”这类历史文案。

### 22.4 如果要继续提升颜色模式

当前最值得加强的不是继续加更多规则，而是：

1. 更稳定的水下颜色归一化
2. 更精细的边界提取
3. 更强的时序先验
4. 更清晰的全局找回策略

---

## 23. 结论

当前这个项目已经不是一个“简单 YOLO 示例”，而是一个带明显业务导向的单目标水下跟踪原型。

它的代码思路可以概括为：

- 页面层负责模式切换和点击交互
- 相机层负责把实时帧转成可分析 `Bitmap`
- 模型层负责给出单帧人体骨架
- 跟踪层负责把“单帧感知”变成“连续目标”
- 绘制层负责把人体框、色块区域、辅助信息实时可视化

其中最核心的设计是：

- `自动识别` 走“人体主导”
- `选择身体颜色` 走“颜色主导”

这也是你后续再改这个项目时，最不能混淆的一条主线。

