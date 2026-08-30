package com.example.swimming.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import com.example.swimming.R
import com.example.swimming.detector.DetectionDebugInfo
import com.example.swimming.detector.PersonDetector
import com.example.swimming.logic.FollowLogic
import com.example.swimming.model.CarCommand
import com.example.swimming.model.PersonDetection
import com.example.swimming.provider.CameraFrameProvider
import com.example.swimming.provider.LocalVideoProvider
import com.example.swimming.provider.VideoFrameProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

enum class InputSource {
    CAMERA,
    LOCAL_VIDEO,
    STILL_IMAGE
}

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private var videoProvider: VideoFrameProvider? = null
    private var cameraProvider: CameraFrameProvider? = null
    private var personDetector: PersonDetector? = null
    private val followLogic = FollowLogic()
    private val frameStream = MutableSharedFlow<Bitmap>(
        extraBufferCapacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    private val _currentFrame = MutableStateFlow<Bitmap?>(null)
    val currentFrame: StateFlow<Bitmap?> = _currentFrame

    private val _currentDetection = MutableStateFlow<PersonDetection?>(null)
    val currentDetection: StateFlow<PersonDetection?> = _currentDetection

    private val _currentCommand = MutableStateFlow(CarCommand.STOP)
    val currentCommand: StateFlow<CarCommand> = _currentCommand

    private val _detectorReady = MutableStateFlow(false)
    val detectorReady: StateFlow<Boolean> = _detectorReady.asStateFlow()

    private val _statusMessage = MutableStateFlow("正在初始化...")
    val statusMessage: StateFlow<String> = _statusMessage.asStateFlow()

    private val _inputSource = MutableStateFlow(InputSource.CAMERA)
    val inputSource: StateFlow<InputSource> = _inputSource.asStateFlow()

    private val _debugInfo = MutableStateFlow(DetectionDebugInfo(message = "waiting"))
    val debugInfo: StateFlow<DetectionDebugInfo> = _debugInfo.asStateFlow()

    init {
        initializeDetector()

        viewModelScope.launch(Dispatchers.Default) {
            frameStream.collectLatest { bitmap ->
                runDetection(bitmap, isSingleFrame = false)
            }
        }
    }

    private fun initializeDetector() {
        if (personDetector != null) return

        try {
            personDetector = PersonDetector(getApplication())
            _detectorReady.value = personDetector?.isReady() == true
            _statusMessage.value = if (_detectorReady.value) {
                "YOLO11 ONNX 已加载，准备接收画面"
            } else {
                "YOLO11 ONNX 初始化失败"
            }
        } catch (t: Throwable) {
            t.printStackTrace()
            personDetector = null
            _detectorReady.value = false
            _statusMessage.value = "YOLO11 初始化异常: ${t.javaClass.simpleName}: ${t.message ?: "unknown"}"
        }
    }

    fun startLocalVideo(uri: Uri) {
        stopActiveSource()
        _inputSource.value = InputSource.LOCAL_VIDEO
        _statusMessage.value = "正在读取视频..."
        videoProvider = LocalVideoProvider(getApplication(), uri).apply {
            setOnFrameAvailableListener(::publishFrame)
            start()
        }
    }

    fun startStillImage(uri: Uri) {
        stopActiveSource()
        _inputSource.value = InputSource.STILL_IMAGE
        _statusMessage.value = "正在读取图片..."
        _currentCommand.value = CarCommand.STOP
        _currentDetection.value = null

        try {
            val resolver = getApplication<Application>().contentResolver
            val bitmap = resolver.openInputStream(uri)?.use(BitmapFactory::decodeStream)
            if (bitmap == null) {
                _statusMessage.value = "图片读取失败"
                _currentFrame.value = null
                return
            }
            _currentFrame.value = bitmap
            runDetectionOnce(bitmap)
        } catch (t: Throwable) {
            t.printStackTrace()
            _currentFrame.value = null
            _currentDetection.value = null
            _currentCommand.value = CarCommand.STOP
            _statusMessage.value = "图片读取异常: ${t.javaClass.simpleName}: ${t.message ?: "unknown"}"
        }
    }

    fun startBundledVideo() {
        val app = getApplication<Application>()
        val uri = Uri.parse("android.resource://${app.packageName}/${R.raw.test_follow}")
        startLocalVideo(uri)
    }

    fun startCamera(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        stopActiveSource()
        _inputSource.value = InputSource.CAMERA
        _statusMessage.value = "正在启动后摄像头..."
        cameraProvider = CameraFrameProvider(getApplication()).apply {
            setOnFrameAvailableListener(::publishFrame)
            start(lifecycleOwner, previewView)
        }
    }

    fun onCameraPermissionDenied() {
        _statusMessage.value = "相机权限被拒绝，无法打开后摄像头"
        _currentCommand.value = CarCommand.STOP
    }

    fun stopActiveSource() {
        videoProvider?.stop()
        videoProvider = null
        cameraProvider?.stop()
        cameraProvider = null
    }

    private fun publishFrame(bitmap: Bitmap) {
        _currentFrame.value = bitmap
        frameStream.tryEmit(bitmap)
    }

    private fun runDetectionOnce(bitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.Default) {
            runDetection(bitmap, isSingleFrame = true)
        }
    }

    private fun runDetection(bitmap: Bitmap, isSingleFrame: Boolean) {
        try {
            initializeDetector()
            val detection = personDetector?.detect(bitmap)
            _debugInfo.value = personDetector?.getLastDebugInfo()
                ?: DetectionDebugInfo(message = "detector unavailable")
            _currentDetection.value = detection
            _currentCommand.value = followLogic.determineCommand(detection)
            _statusMessage.value = when {
                detection != null && isSingleFrame ->
                    "单帧检测成功: conf=${"%.2f".format(detection.confidence)} centerX=${"%.2f".format(detection.centerX)}"
                detection != null ->
                    "检测成功: conf=${"%.2f".format(detection.confidence)}"
                isSingleFrame ->
                    "单帧检测结果为空"
                else ->
                    "未检测到人物"
            }
        } catch (t: Throwable) {
            t.printStackTrace()
            _currentDetection.value = null
            _currentCommand.value = CarCommand.STOP
            _debugInfo.value = DetectionDebugInfo(message = "exception: ${t.javaClass.simpleName}")
            _statusMessage.value = if (isSingleFrame) {
                "单帧检测异常: ${t.javaClass.simpleName}: ${t.message ?: "unknown"}"
            } else {
                "检测异常: ${t.javaClass.simpleName}: ${t.message ?: "unknown"}"
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        stopActiveSource()
        personDetector?.close()
    }
}
