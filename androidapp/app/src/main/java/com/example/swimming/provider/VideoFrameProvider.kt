package com.example.swimming.provider

import android.graphics.Bitmap

interface VideoFrameProvider {
    fun start()
    fun stop()
    fun setOnFrameAvailableListener(listener: (Bitmap) -> Unit)
}
