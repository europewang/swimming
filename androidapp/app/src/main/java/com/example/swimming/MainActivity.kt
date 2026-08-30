package com.example.swimming

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.camera.view.PreviewView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.swimming.model.CarCommand
import com.example.swimming.model.PersonDetection
import com.example.swimming.viewmodel.InputSource
import com.example.swimming.viewmodel.MainViewModel

class MainActivity : ComponentActivity() {
    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen(viewModel)
                }
            }
        }
    }
}

private const val DEBUG_BUILD_LABEL = "debug-2026-06-23-v4-yolo11"

@Composable
fun MainScreen(viewModel: MainViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val currentFrame by viewModel.currentFrame.collectAsState()
    val currentDetection by viewModel.currentDetection.collectAsState()
    val currentCommand by viewModel.currentCommand.collectAsState()
    val detectorReady by viewModel.detectorReady.collectAsState()
    val statusMessage by viewModel.statusMessage.collectAsState()
    val inputSource by viewModel.inputSource.collectAsState()
    val debugInfo by viewModel.debugInfo.collectAsState()

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }

    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
        }
    }

    val videoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let(viewModel::startLocalVideo)
    }

    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let(viewModel::startStillImage)
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        if (granted) {
            viewModel.startCamera(lifecycleOwner, previewView)
        } else {
            viewModel.onCameraPermissionDenied()
        }
    }

    LaunchedEffect(hasCameraPermission) {
        if (hasCameraPermission && inputSource == InputSource.CAMERA) {
            viewModel.startCamera(lifecycleOwner, previewView)
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            viewModel.stopActiveSource()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .statusBarsPadding(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = DEBUG_BUILD_LABEL,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = when (inputSource) {
                    InputSource.CAMERA -> "实时模式"
                    InputSource.LOCAL_VIDEO -> "视频模式"
                    InputSource.STILL_IMAGE -> "单帧图片模式"
                },
                color = MaterialTheme.colorScheme.onBackground
            )
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            SourceButton(
                text = "使用摄像头",
                selected = inputSource == InputSource.CAMERA,
                onClick = {
                    if (hasCameraPermission) {
                        viewModel.startCamera(lifecycleOwner, previewView)
                    } else {
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                },
                modifier = Modifier.weight(1f)
            )
            SourceButton(
                text = "选择本地视频",
                selected = inputSource == InputSource.LOCAL_VIDEO,
                onClick = { videoPicker.launch("video/*") },
                modifier = Modifier.weight(1f)
            )
            SourceButton(
                text = "选择图片",
                selected = inputSource == InputSource.STILL_IMAGE,
                onClick = { imagePicker.launch("image/*") },
                modifier = Modifier.weight(1f)
            )
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = when (inputSource) {
                    InputSource.CAMERA -> "当前输入: 后摄像头"
                    InputSource.LOCAL_VIDEO -> "当前输入: 本地视频"
                    InputSource.STILL_IMAGE -> "当前输入: 单张图片"
                },
                color = MaterialTheme.colorScheme.onBackground
            )
            CommandBadge(currentCommand)
        }

        Text(
            text = if (detectorReady) "当前算法: YOLO11n ONNX person" else "YOLO11n 不可用，暂时只显示画面",
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            color = if (detectorReady) MaterialTheme.colorScheme.onBackground else MaterialTheme.colorScheme.error
        )

        Text(
            text = statusMessage,
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            color = MaterialTheme.colorScheme.onBackground
        )

        if (inputSource == InputSource.STILL_IMAGE) {
            Text(
                text = if (currentDetection == null) "单帧结果为空" else "单帧结果已检出",
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 6.dp),
                color = if (currentDetection == null) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Medium
            )
        }

        Text(
            text = "debug candidates=${debugInfo.candidateCount} selected=${debugInfo.selectedCount} best=${"%.2f".format(debugInfo.bestConfidence)} ${debugInfo.message}",
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 6.dp),
            color = MaterialTheme.colorScheme.onBackground,
            fontSize = 12.sp
        )

        Spacer(modifier = Modifier.height(12.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .background(Color.Black),
            contentAlignment = Alignment.Center
        ) {
            when {
                inputSource == InputSource.CAMERA && hasCameraPermission -> {
                    AndroidView(
                        factory = { previewView },
                        modifier = Modifier.fillMaxSize()
                    )
                    DetectionOverlay(
                        frameWidth = currentFrame?.width,
                        frameHeight = currentFrame?.height,
                        detection = currentDetection
                    )
                }

                currentFrame != null -> {
                    Image(
                        bitmap = currentFrame!!.asImageBitmap(),
                        contentDescription = "Input Frame",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Fit
                    )
                    DetectionOverlay(
                        frameWidth = currentFrame?.width,
                        frameHeight = currentFrame?.height,
                        detection = currentDetection
                    )
                }

                else -> {
                    Text(
                        text = when (inputSource) {
                            InputSource.CAMERA -> "等待相机权限或相机启动..."
                            InputSource.LOCAL_VIDEO -> "请选择一个本地视频"
                            InputSource.STILL_IMAGE -> "请选择一张图片"
                        },
                        color = Color.White
                    )
                }
            }
        }

        currentDetection?.let { detection ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("centerX: ${"%.2f".format(detection.centerX)}")
                Text("area: ${"%.2f".format(detection.width * detection.height)}")
                Text("conf: ${"%.2f".format(detection.confidence)}")
            }
        } ?: Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            contentAlignment = Alignment.Center
        ) {
            Text("未检测到人物")
        }
    }
}

@Composable
private fun DetectionOverlay(
    frameWidth: Int?,
    frameHeight: Int?,
    detection: PersonDetection?
) {
    if (frameWidth == null || frameHeight == null || detection == null) return

    Canvas(modifier = Modifier.fillMaxSize()) {
        val canvasWidth = size.width
        val canvasHeight = size.height
        val imageRatio = frameWidth.toFloat() / frameHeight.toFloat()
        val canvasRatio = canvasWidth / canvasHeight

        val actualImageWidth: Float
        val actualImageHeight: Float
        val startX: Float
        val startY: Float

        if (imageRatio > canvasRatio) {
            actualImageWidth = canvasWidth
            actualImageHeight = canvasWidth / imageRatio
            startX = 0f
            startY = (canvasHeight - actualImageHeight) / 2f
        } else {
            actualImageHeight = canvasHeight
            actualImageWidth = canvasHeight * imageRatio
            startX = (canvasWidth - actualImageWidth) / 2f
            startY = 0f
        }

        drawRect(
            color = Color.Green,
            topLeft = Offset(
                startX + detection.left * actualImageWidth,
                startY + detection.top * actualImageHeight
            ),
            size = Size(
                detection.width * actualImageWidth,
                detection.height * actualImageHeight
            ),
            style = Stroke(width = 6f)
        )
    }
}

@Composable
private fun SourceButton(
    text: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    if (selected) {
        Button(onClick = onClick, modifier = modifier) {
            Text(text)
        }
    } else {
        OutlinedButton(
            onClick = onClick,
            modifier = modifier,
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline),
            shape = RoundedCornerShape(12.dp)
        ) {
            Text(text)
        }
    }
}

@Composable
fun CommandBadge(command: CarCommand) {
    val color = when (command) {
        CarCommand.FORWARD -> Color(0xFF4CAF50)
        CarCommand.LEFT -> Color(0xFF2196F3)
        CarCommand.RIGHT -> Color(0xFFFF9800)
        CarCommand.STOP -> Color(0xFFF44336)
    }

    Surface(
        color = color,
        shape = MaterialTheme.shapes.medium,
        modifier = Modifier.padding(8.dp)
    ) {
        Text(
            text = command.name,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
            color = Color.White,
            fontWeight = FontWeight.Bold,
            fontSize = 18.sp
        )
    }
}
