package com.example.yolopose.camera

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

class CameraFrameAnalyzer(
    private val onFrame: (Bitmap) -> Unit
) : ImageAnalysis.Analyzer {

    private var lastBitmap: Bitmap? = null

    override fun analyze(image: ImageProxy) {
        val bitmap = imageProxyToBitmap(image)
        if (bitmap != null) {
            onFrame(bitmap)
        }
        image.close()
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        return when (image.format) {
            ImageFormat.YUV_420_888 -> yuvImageToBitmap(image)
            else -> null
        }
    }

    private fun yuvImageToBitmap(image: ImageProxy): Bitmap {
        val argb = IntArray(image.width * image.height)
        yuv420ToArgb(image, argb)

        val bitmap = lastBitmap?.takeIf { it.width == image.width && it.height == image.height }
            ?: Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888).also {
                lastBitmap = it
            }
        bitmap.setPixels(argb, 0, image.width, 0, 0, image.width, image.height)
        return rotateBitmap(bitmap, image.imageInfo.rotationDegrees)
    }

    private fun yuv420ToArgb(image: ImageProxy, out: IntArray) {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        var outputIndex = 0
        for (row in 0 until image.height) {
            val yRowOffset = row * yRowStride
            val uvRowOffset = (row / 2) * uvRowStride
            for (col in 0 until image.width) {
                val y = yBuffer.get(yRowOffset + col * yPixelStride).toInt() and 0xFF
                val uvOffset = uvRowOffset + (col / 2) * uvPixelStride
                val u = uBuffer.get(uvOffset).toInt() and 0xFF
                val v = vBuffer.get(uvOffset).toInt() and 0xFF
                out[outputIndex++] = yuvToRgb(y, u, v)
            }
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) {
            return bitmap
        }
        val matrix = Matrix().apply {
            postRotate(rotationDegrees.toFloat())
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun yuvToRgb(yValue: Int, uValue: Int, vValue: Int): Int {
        val y = (yValue - 16).coerceAtLeast(0)
        val u = uValue - 128
        val v = vValue - 128
        val y1192 = 1192 * y
        val r = (y1192 + 1634 * v).coerceIn(0, 262143)
        val g = (y1192 - 833 * v - 400 * u).coerceIn(0, 262143)
        val b = (y1192 + 2066 * u).coerceIn(0, 262143)
        return -0x1000000 or
            (r shl 6 and 0xff0000) or
            (g shr 2 and 0xff00) or
            (b shr 10 and 0xff)
    }
}
