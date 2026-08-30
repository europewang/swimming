package com.example.yolopose.network

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Network
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

class Esp32SnapshotFrameSource(
    private val scope: CoroutineScope,
    private val snapshotUrl: String,
    private val networkProvider: () -> Network?,
    private val pollIntervalMs: Long = 150L,
    private val onFrame: (Bitmap) -> Unit,
    private val onError: (String) -> Unit
) {
    private var pollingJob: Job? = null

    fun start() {
        if (pollingJob?.isActive == true) return
        pollingJob = scope.launch(Dispatchers.IO) {
            while (isActive) {
                try {
                    val bitmap = fetchSnapshot()
                    if (bitmap != null) {
                        withContext(Dispatchers.Main) {
                            onFrame(bitmap)
                        }
                    } else {
                        withContext(Dispatchers.Main) {
                            onError("ESP32 返回了空图像")
                        }
                    }
                } catch (error: Exception) {
                    withContext(Dispatchers.Main) {
                        onError(error.message ?: error.javaClass.simpleName)
                    }
                    delay(400L)
                }
                delay(pollIntervalMs)
            }
        }
    }

    fun stop() {
        pollingJob?.cancel()
        pollingJob = null
    }

    @Throws(IOException::class)
    private fun fetchSnapshot(): Bitmap? {
        val url = URL(snapshotUrl)
        val network = networkProvider()
        val connection = ((network?.openConnection(url) ?: url.openConnection()) as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 1800
            readTimeout = 2200
            useCaches = false
            setRequestProperty("Connection", "close")
        }
        return try {
            connection.connect()
            connection.inputStream.use { input ->
                BitmapFactory.decodeStream(input)
            }
        } finally {
            connection.disconnect()
        }
    }
}
