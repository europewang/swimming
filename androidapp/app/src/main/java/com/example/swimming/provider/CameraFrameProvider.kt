package com.example.swimming.provider

import android.content.Context
import android.graphics.Bitmap
import android.util.Size
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.swimming.util.ImageProxyBitmapConverter
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraFrameProvider(
    private val context: Context
) {
    private var listener: ((Bitmap) -> Unit)? = null
    private var cameraExecutor: ExecutorService? = null
    private var boundCameraProvider: ProcessCameraProvider? = null

    fun setOnFrameAvailableListener(listener: (Bitmap) -> Unit) {
        this.listener = listener
    }

    fun start(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        stop()
        cameraExecutor = Executors.newSingleThreadExecutor()

        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener(
            {
                val cameraProvider = cameraProviderFuture.get()
                boundCameraProvider = cameraProvider

                val preview = Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .build()
                    .also { it.setSurfaceProvider(previewView.getSurfaceProvider()) }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { analysis ->
                        val executor = cameraExecutor ?: return@also
                        analysis.setAnalyzer(executor) { imageProxy ->
                            try {
                                val bitmap = ImageProxyBitmapConverter.toBitmap(imageProxy)
                                listener?.invoke(bitmap)
                            } finally {
                                imageProxy.close()
                            }
                        }
                    }

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalysis
                )
            },
            ContextCompat.getMainExecutor(context)
        )
    }

    fun stop() {
        boundCameraProvider?.unbindAll()
        boundCameraProvider = null
        cameraExecutor?.shutdownNow()
        cameraExecutor = null
    }
}
