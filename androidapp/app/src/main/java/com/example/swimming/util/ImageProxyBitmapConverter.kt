package com.example.swimming.util

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

object ImageProxyBitmapConverter {

    fun toBitmap(imageProxy: ImageProxy): Bitmap {
        val nv21 = yuv420888ToNv21(imageProxy)
        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            imageProxy.width,
            imageProxy.height,
            null
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 90, out)
        val imageBytes = out.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        val matrix = Matrix().apply {
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
        }

        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer.toByteArray()
        val uBuffer = image.planes[1].buffer.toByteArray()
        val vBuffer = image.planes[2].buffer.toByteArray()

        val nv21 = ByteArray(yBuffer.size + uBuffer.size + vBuffer.size)
        System.arraycopy(yBuffer, 0, nv21, 0, yBuffer.size)

        var position = yBuffer.size
        val rowStride = image.planes[1].rowStride
        val pixelStride = image.planes[1].pixelStride
        val chromaWidth = image.width / 2
        val chromaHeight = image.height / 2

        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val index = row * rowStride + col * pixelStride
                nv21[position++] = vBuffer[index]
                nv21[position++] = uBuffer[index]
            }
        }

        return nv21
    }

    private fun ByteBuffer.toByteArray(): ByteArray {
        rewind()
        val bytes = ByteArray(remaining())
        get(bytes)
        return bytes
    }
}
