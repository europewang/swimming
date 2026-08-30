package com.example.swimming.model

data class PersonDetection(
    val centerX: Float,
    val centerY: Float,
    val width: Float,
    val height: Float,
    val confidence: Float,
    // Box coordinates for drawing: values are 0.0 to 1.0 (relative to image width/height)
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)
