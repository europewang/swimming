package com.example.swimming.provider

import android.graphics.Bitmap

class Esp32MjpegProvider(
    private val streamUrl: String
) : VideoFrameProvider {

    private var listener: ((Bitmap) -> Unit)? = null

    override fun start() {
        error("ESP32-CAM MJPEG support is reserved for the next stage. URL: $streamUrl")
    }

    override fun stop() = Unit

    override fun setOnFrameAvailableListener(listener: (Bitmap) -> Unit) {
        this.listener = listener
    }
}
