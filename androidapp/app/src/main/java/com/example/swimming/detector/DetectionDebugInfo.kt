package com.example.swimming.detector

data class DetectionDebugInfo(
    val candidateCount: Int = 0,
    val selectedCount: Int = 0,
    val bestConfidence: Float = 0f,
    val message: String = ""
)
