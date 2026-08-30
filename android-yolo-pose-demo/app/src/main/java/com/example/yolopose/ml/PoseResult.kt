package com.example.yolopose.ml

data class KeyPoint(
    val x: Float,
    val y: Float,
    val confidence: Float
)

data class PoseResult(
    val boundingBox: FloatArray,
    val confidence: Float,
    val keyPoints: List<KeyPoint>
)
