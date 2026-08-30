package com.example.swimming.provider

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class LocalVideoProvider(
    private val context: Context,
    private val videoUri: Uri
) : VideoFrameProvider {

    private var listener: ((Bitmap) -> Unit)? = null
    private var job: Job? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    override fun setOnFrameAvailableListener(listener: (Bitmap) -> Unit) {
        this.listener = listener
    }

    override fun start() {
        if (job?.isActive == true) return
        job = scope.launch {
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(context, videoUri)
                val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                val durationMs = durationStr?.toLongOrNull() ?: 0L
                val fps = 15 // Limit fps for performance
                val frameIntervalMs = 1000L / fps
                var currentTimeMs = 0L

                while (isActive) {
                    val timeStart = System.currentTimeMillis()

                    val bitmap = retriever.getFrameAtTime(
                        currentTimeMs * 1000,
                        MediaMetadataRetriever.OPTION_CLOSEST
                    )
                    
                    if (bitmap != null) {
                        val scaledBitmap = scaleBitmap(bitmap, 640)
                        listener?.invoke(scaledBitmap)
                    }

                    currentTimeMs += frameIntervalMs
                    if (currentTimeMs >= durationMs && durationMs > 0L) {
                        currentTimeMs = 0L
                    }

                    val elapsed = System.currentTimeMillis() - timeStart
                    val sleepTime = frameIntervalMs - elapsed
                    if (sleepTime > 0) {
                        delay(sleepTime)
                    } else {
                        delay(10)
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                try {
                    retriever.release()
                } catch (_: Exception) {
                }
            }
        }
    }

    override fun stop() {
        job?.cancel()
        job = null
    }

    private fun scaleBitmap(bitmap: Bitmap, maxDimension: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        if (width <= maxDimension && height <= maxDimension) return bitmap

        val ratio = width.toFloat() / height.toFloat()
        val newWidth = if (ratio > 1) maxDimension else (maxDimension * ratio).toInt()
        val newHeight = if (ratio > 1) (maxDimension / ratio).toInt() else maxDimension

        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }
}
