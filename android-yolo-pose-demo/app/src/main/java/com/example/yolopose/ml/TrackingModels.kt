package com.example.yolopose.ml

enum class DetectionMode {
    INITIALIZING,
    COLOR_TRACK,
    FULL_BODY,
    LEG_ONLY,
    FOOT_ONLY,
    TRACKED_MEMORY,
    LOST
}

data class ColorTrackingTemplate(
    val averageColor: Int,
    val anchorColor: Int,
    val skinColor: Int,
    val swimwearColor: Int,
    val swimwearPalette: List<Int>,
    val swimwearHue: Float,
    val swimwearHueTolerance: Float,
    val swimwearMinSaturation: Float,
    val swimwearMaxSaturation: Float,
    val swimwearMinValue: Float,
    val swimwearMaxValue: Float,
    val meanWidth: Float,
    val meanHeight: Float,
    val sampleCount: Int,
    val directColorLock: Boolean = false
)

data class ColorRegionDetection(
    val box: FloatArray,
    val contour: List<FloatArray>,
    val cells: List<FloatArray>
)

data class TrackedPerson(
    val poseResult: PoseResult?,
    val mode: DetectionMode,
    val trackId: Int,
    val averageColor: Int?,
    val anchorColor: Int?,
    val skinColor: Int?,
    val swimwearColor: Int?,
    val trackingBox: FloatArray?,
    val skinBox: FloatArray?,
    val swimwearBox: FloatArray?,
    val skinContour: List<FloatArray>?,
    val swimwearContour: List<FloatArray>?,
    val skinCells: List<FloatArray>?,
    val swimwearCells: List<FloatArray>?,
    val colorComboBox: FloatArray?,
    val confidence: Float
)
