package com.example.yolopose.network

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Network
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

class Esp32MjpegFrameSource(
    private val scope: CoroutineScope,
    private val streamUrl: String,
    private val networkProvider: () -> Network?,
    private val onFrame: (Bitmap) -> Unit,
    private val onError: (String) -> Unit
) {
    private var streamJob: Job? = null

    fun start() {
        if (streamJob?.isActive == true) return
        streamJob = scope.launch(Dispatchers.IO) {
            while (isActive) {
                runCatching {
                    openAndConsumeStream()
                }.onFailure { error ->
                    withContext(Dispatchers.Main) {
                        onError(error.message ?: error.javaClass.simpleName)
                    }
                    kotlinx.coroutines.delay(500L)
                }
            }
        }
    }

    fun stop() {
        streamJob?.cancel()
        streamJob = null
    }

    private suspend fun openAndConsumeStream() {
        val url = URL(streamUrl)
        val network = networkProvider()
        val connection = ((network?.openConnection(url) ?: url.openConnection()) as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 2500
            readTimeout = 15000
            useCaches = false
            setRequestProperty("Connection", "Keep-Alive")
        }
        try {
            connection.connect()
            connection.inputStream.use { input ->
                consumeMjpegFrames(input)
            }
        } finally {
            connection.disconnect()
        }
    }

    private suspend fun consumeMjpegFrames(input: InputStream) {
        val buffer = ByteArray(8192)
        val frameBuffer = ByteArrayOutputStream(48 * 1024)
        var collecting = false
        var previousByte = -1

        while (scope.isActive) {
            scope.ensureActive()
            val count = input.read(buffer)
            if (count <= 0) break

            for (index in 0 until count) {
                val current = buffer[index].toInt() and 0xFF

                if (!collecting) {
                    if (previousByte == 0xFF && current == 0xD8) {
                        frameBuffer.reset()
                        frameBuffer.write(0xFF)
                        frameBuffer.write(0xD8)
                        collecting = true
                    }
                } else {
                    frameBuffer.write(current)
                    if (previousByte == 0xFF && current == 0xD9) {
                        val jpegBytes = frameBuffer.toByteArray()
                        BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)?.let { bitmap ->
                            withContext(Dispatchers.Main) {
                                onFrame(bitmap)
                            }
                        }
                        collecting = false
                        frameBuffer.reset()
                    }
                }

                previousByte = current
            }
        }
    }
}
