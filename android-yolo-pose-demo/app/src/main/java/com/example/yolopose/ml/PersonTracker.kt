package com.example.yolopose.ml

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt

class PersonTracker {
    private data class ColorCorrectionProfile(
        val redGain: Float,
        val greenGain: Float,
        val blueGain: Float
    )

    private data class SamplingFrame(
        val sourceWidth: Int,
        val sourceHeight: Int,
        val sampleWidth: Int,
        val sampleHeight: Int,
        val pixels: IntArray
    )

    private data class MotionSnapshot(
        val centerX: Float,
        val centerY: Float,
        val width: Float,
        val height: Float
    )

    private data class ConnectedComponent(
        val cells: List<Pair<Int, Int>>,
        val minRow: Int,
        val minCol: Int,
        val maxRow: Int,
        val maxCol: Int
    )

    private data class SegmentedRegion(
        val detection: ColorRegionDetection,
        val representativeColor: Int,
        val area: Float
    )

    private var previousTracked: TrackedPerson? = null
    private var lastTracked: TrackedPerson? = null
    private var lastUpdatedAtMs: Long = 0L
    private var lastGlobalRecoveryAtMs: Long = 0L
    private var nextTrackId: Int = 1
    private var template: ColorTrackingTemplate? = null
    private var templateConfirmed: Boolean = false
    private var selectedPoint: FloatArray? = null
    private var colorPickPending: Boolean = false
    private var initializing: Boolean = false
    private var initStartAtMs: Long = 0L
    private val initAverageColors = mutableListOf<Int>()
    private val initAnchorColors = mutableListOf<Int>()
    private val initSkinColors = mutableListOf<Int>()
    private val initSwimwearColors = mutableListOf<Int>()
    private val initWidths = mutableListOf<Float>()
    private val initHeights = mutableListOf<Float>()

    fun reset() {
        previousTracked = null
        lastTracked = null
        lastUpdatedAtMs = 0L
        lastGlobalRecoveryAtMs = 0L
        template = null
        templateConfirmed = false
        colorPickPending = false
        initializing = false
        initStartAtMs = 0L
        initAverageColors.clear()
        initAnchorColors.clear()
        initSkinColors.clear()
        initSwimwearColors.clear()
        initWidths.clear()
        initHeights.clear()
    }

    fun startInitialization(nowMs: Long) {
        val preservedSelection = selectedPoint?.copyOf()
        reset()
        selectedPoint = preservedSelection
        initializing = true
        initStartAtMs = nowMs
    }

    fun isInitializing(): Boolean = initializing

    fun initializationElapsedSeconds(nowMs: Long): Int {
        if (!initializing) return 0
        return ((nowMs - initStartAtMs) / 1000L).toInt().coerceAtLeast(0)
    }

    fun templateReady(): Boolean = template != null

    fun isTemplateConfirmed(): Boolean = templateConfirmed

    fun isColorPickPending(): Boolean = colorPickPending

    fun shouldSkipPoseInference(): Boolean = colorPickPending || (templateConfirmed && template?.directColorLock == true)

    fun confirmTemplateLock() {
        if (template != null) {
            templateConfirmed = true
        }
    }

    private fun commitTracked(tracked: TrackedPerson, nowMs: Long) {
        previousTracked = lastTracked
        lastTracked = tracked
        lastUpdatedAtMs = nowMs
    }

    fun setSelectionPoint(x: Float, y: Float) {
        selectedPoint = floatArrayOf(x, y)
    }

    fun clearSelectionPoint() {
        selectedPoint = null
    }

    fun startDirectColorLock(x: Float, y: Float, nowMs: Long) {
        reset()
        selectedPoint = floatArrayOf(x, y)
        colorPickPending = true
        initStartAtMs = nowMs
    }

    fun finishInitialization(): Boolean {
        if (initAverageColors.size < MIN_TEMPLATE_SAMPLES || initAnchorColors.size < MIN_TEMPLATE_SAMPLES) {
            return false
        }
        template = ColorTrackingTemplate(
            averageColor = averageColor(initAverageColors),
            anchorColor = averageColor(initAnchorColors),
            skinColor = averageColor(initSkinColors),
            swimwearColor = averageColor(initSwimwearColors),
            swimwearPalette = buildPaletteFromSamples(initSwimwearColors),
            swimwearHue = dominantHue(initSwimwearColors),
            swimwearHueTolerance = DEFAULT_HUE_TOLERANCE,
            swimwearMinSaturation = paletteMinSaturation(initSwimwearColors),
            swimwearMaxSaturation = paletteMaxSaturation(initSwimwearColors),
            swimwearMinValue = paletteMinValue(initSwimwearColors),
            swimwearMaxValue = paletteMaxValue(initSwimwearColors),
            meanWidth = initWidths.average().toFloat(),
            meanHeight = initHeights.average().toFloat(),
            sampleCount = initAverageColors.size
        )
        initializing = false
        return true
    }

    fun update(detectedPose: PoseResult?, frame: Bitmap, nowMs: Long): TrackedPerson {
        val samplingFrame = buildSamplingFrame(frame)
        if (colorPickPending) {
            return initializeDirectColorTemplate(frame, samplingFrame, nowMs)
        }

        if (templateConfirmed && template != null) {
            return updateWithLockedTemplate(frame, samplingFrame, nowMs)
        }

        if (detectedPose != null) {
            val detectedMode = resolveMode(detectedPose)
            if (detectedMode == DetectionMode.LOST) {
                return fallbackToColorOrMemory(frame, samplingFrame, nowMs)
            }
            val smoothedPose = lastTracked?.poseResult?.let { smoothPose(it, detectedPose) } ?: detectedPose
            val trackingBox = deriveTrackingBox(smoothedPose, frame.width.toFloat(), frame.height.toFloat())
            if (!matchesSelectedPoint(trackingBox)) {
                return fallbackToColorOrMemory(frame, samplingFrame, nowMs)
            }
            val correction = buildColorCorrectionProfile(samplingFrame, trackingBox)
            val averageColor = extractAverageColor(samplingFrame, trackingBox, correction)
            val anchorColor = extractAnchorColor(samplingFrame, trackingBox, correction)
            if (template != null && !isColorConsistentWithTemplate(averageColor, anchorColor, correction)) {
                return fallbackToColorOrMemory(frame, samplingFrame, nowMs)
            }
            val provisionalSkinBox = buildPoseAwareSkinHintBox(smoothedPose, trackingBox)
            val provisionalSwimwearBox = buildPoseAwareSwimwearHintBox(smoothedPose, trackingBox)
            val sampledSkinColor = extractAverageColor(samplingFrame, provisionalSkinBox, correction)
            val sampledSwimwearColor = extractAverageColor(samplingFrame, provisionalSwimwearBox, correction)
            collectTemplateSample(
                trackingBox = trackingBox,
                averageColor = averageColor,
                anchorColor = anchorColor,
                skinColor = sampledSkinColor,
                swimwearColor = sampledSwimwearColor
            )
            val skinReference = template?.skinColor ?: sampledSkinColor
            val swimwearReference = template?.swimwearColor ?: sampledSwimwearColor
            val skinRegion = detectColorRegion(
                samplingFrame = samplingFrame,
                trackingBox = provisionalSkinBox,
                referenceColor = skinReference,
                referencePalette = listOf(skinReference),
                template = null,
                correction = correction,
                bandTopRatio = 0f,
                bandBottomRatio = 1f
            ) ?: ColorRegionDetection(provisionalSkinBox, boxToContour(provisionalSkinBox), listOf(provisionalSkinBox))
            val swimwearRegion = detectColorRegion(
                samplingFrame = samplingFrame,
                trackingBox = provisionalSwimwearBox,
                referenceColor = swimwearReference,
                referencePalette = template?.swimwearPalette ?: listOf(swimwearReference),
                template = template,
                correction = correction,
                bandTopRatio = 0f,
                bandBottomRatio = 1f
            ) ?: ColorRegionDetection(provisionalSwimwearBox, boxToContour(provisionalSwimwearBox), listOf(provisionalSwimwearBox))
            val colorComboBox = buildColorComboBox(trackingBox, skinRegion.box, swimwearRegion.box)
            val tracked = TrackedPerson(
                poseResult = smoothedPose.copy(boundingBox = trackingBox.copyOf()),
                mode = if (initializing) DetectionMode.INITIALIZING else refineModeWithPriority(detectedMode),
                trackId = lastTracked?.trackId ?: nextTrackId++,
                averageColor = averageColor,
                anchorColor = anchorColor,
                skinColor = skinReference,
                swimwearColor = swimwearReference,
                trackingBox = trackingBox,
                skinBox = skinRegion.box,
                swimwearBox = swimwearRegion.box,
                skinContour = null,
                swimwearContour = null,
                skinCells = null,
                swimwearCells = null,
                colorComboBox = colorComboBox,
                confidence = smoothedPose.confidence
            )
            commitTracked(tracked, nowMs)
            return tracked
        }
        return fallbackToColorOrMemory(frame, samplingFrame, nowMs)
    }

    private fun initializeDirectColorTemplate(frame: Bitmap, samplingFrame: SamplingFrame, nowMs: Long): TrackedPerson {
        val point = selectedPoint ?: return TrackedPerson(
            poseResult = null,
            mode = DetectionMode.LOST,
            trackId = nextTrackId,
            averageColor = null,
            anchorColor = null,
            skinColor = null,
            swimwearColor = null,
            trackingBox = null,
            skinBox = null,
            swimwearBox = null,
            skinContour = null,
            swimwearContour = null,
            skinCells = null,
            swimwearCells = null,
            colorComboBox = null,
            confidence = 0f
        )
        val fullFrame = floatArrayOf(0f, 0f, frame.width.toFloat(), frame.height.toFloat())
        val correction = buildColorCorrectionProfile(samplingFrame, fullFrame)
        val waterColor = estimateWaterBackgroundColor(samplingFrame, fullFrame, correction)
        val sampleBox = buildTapSampleBox(frame, point)
        val sampledColor = extractAverageColor(samplingFrame, sampleBox, correction)
        val selectedRegion = detectColorRegion(
            samplingFrame = samplingFrame,
            trackingBox = fullFrame,
            referenceColor = sampledColor,
            referencePalette = null,
            template = null,
            correction = correction,
            bandTopRatio = 0f,
            bandBottomRatio = 1f,
            excludedColor = waterColor,
            distanceThreshold = DIRECT_COLOR_THRESHOLD,
            requireCompact = true,
            preferredPoint = point,
            expectedBox = sampleBox,
            cols = 18,
            rows = 20
        ) ?: ColorRegionDetection(sampleBox, boxToContour(sampleBox), listOf(sampleBox))

        template = ColorTrackingTemplate(
            averageColor = sampledColor,
            anchorColor = sampledColor,
            skinColor = sampledColor,
            swimwearColor = sampledColor,
            swimwearPalette = emptyList(),
            swimwearHue = 0f,
            swimwearHueTolerance = 0f,
            swimwearMinSaturation = 0f,
            swimwearMaxSaturation = 1f,
            swimwearMinValue = 0f,
            swimwearMaxValue = 1f,
            meanWidth = (selectedRegion.box[2] - selectedRegion.box[0]).coerceAtLeast(24f),
            meanHeight = (selectedRegion.box[3] - selectedRegion.box[1]).coerceAtLeast(24f),
            sampleCount = 1,
            directColorLock = true
        )
        templateConfirmed = true
        colorPickPending = false
        val tracked = TrackedPerson(
            poseResult = null,
            mode = DetectionMode.COLOR_TRACK,
            trackId = nextTrackId++,
            averageColor = sampledColor,
            anchorColor = sampledColor,
            skinColor = null,
            swimwearColor = sampledColor,
            trackingBox = selectedRegion.box,
            skinBox = null,
            swimwearBox = selectedRegion.box,
            skinContour = null,
            swimwearContour = selectedRegion.contour,
            skinCells = null,
            swimwearCells = selectedRegion.cells,
            colorComboBox = selectedRegion.box,
            confidence = 0.82f
        )
        commitTracked(tracked, nowMs)
        return tracked
    }

    private fun updateWithLockedTemplate(
        frame: Bitmap,
        samplingFrame: SamplingFrame,
        nowMs: Long
    ): TrackedPerson {
        val activeTemplate = template ?: return fallbackToColorOrMemory(frame, samplingFrame, nowMs)
        if (activeTemplate.directColorLock) {
            return updateWithDirectColorTemplate(frame, samplingFrame, nowMs, activeTemplate)
        }
        val searchWindow = buildLockedSearchWindow(frame, activeTemplate)
        val correction = buildColorCorrectionProfile(samplingFrame, searchWindow)
        val waterColor = estimateWaterBackgroundColor(samplingFrame, searchWindow, correction)
        val swimwearRegion = detectColorRegion(
            samplingFrame = samplingFrame,
            trackingBox = searchWindow,
            referenceColor = activeTemplate.swimwearColor,
            referencePalette = activeTemplate.swimwearPalette,
            template = activeTemplate,
            correction = correction,
            bandTopRatio = 0f,
            bandBottomRatio = 1f,
            excludedColor = waterColor,
            distanceThreshold = LOCKED_SWIMWEAR_THRESHOLD,
            requireCompact = true
        )
        if (swimwearRegion != null) {
            val supportBox = expandBox(swimwearRegion.box, frame.width.toFloat(), frame.height.toFloat(), 0.38f, 0.34f)
            val skinRegion = detectColorRegion(
                samplingFrame = samplingFrame,
                trackingBox = supportBox,
                referenceColor = activeTemplate.skinColor,
                referencePalette = listOf(activeTemplate.skinColor),
                template = null,
                correction = correction,
                bandTopRatio = 0f,
                bandBottomRatio = 1f,
                excludedColor = waterColor,
                distanceThreshold = LOCKED_SKIN_THRESHOLD,
                requireCompact = false
            )
            val colorComboBox = buildLockedColorComboBox(swimwearRegion.box, skinRegion?.box)
            val averageColor = extractAverageColor(samplingFrame, swimwearRegion.box, correction)
            val anchorColor = extractAnchorColor(samplingFrame, swimwearRegion.box, correction)
            val tracked = TrackedPerson(
                poseResult = null,
                mode = DetectionMode.COLOR_TRACK,
                trackId = lastTracked?.trackId ?: nextTrackId++,
                averageColor = averageColor,
                anchorColor = anchorColor,
                skinColor = activeTemplate.skinColor,
                swimwearColor = activeTemplate.swimwearColor,
                trackingBox = colorComboBox.copyOf(),
                skinBox = skinRegion?.box,
                swimwearBox = swimwearRegion.box,
                skinContour = skinRegion?.contour,
                swimwearContour = swimwearRegion.contour,
                skinCells = skinRegion?.cells,
                swimwearCells = swimwearRegion.cells,
                colorComboBox = colorComboBox,
                confidence = 0.72f
            )
            commitTracked(tracked, nowMs)
            return tracked
        }

        val memory = lastTracked
        if (memory != null && nowMs - lastUpdatedAtMs <= MEMORY_HOLD_MS * 3) {
            return memory.copy(mode = DetectionMode.TRACKED_MEMORY)
        }

        return TrackedPerson(
            poseResult = null,
            mode = DetectionMode.LOST,
            trackId = memory?.trackId ?: nextTrackId,
            averageColor = memory?.averageColor ?: activeTemplate.averageColor,
            anchorColor = memory?.anchorColor ?: activeTemplate.anchorColor,
            skinColor = memory?.skinColor ?: activeTemplate.skinColor,
            swimwearColor = memory?.swimwearColor ?: activeTemplate.swimwearColor,
            trackingBox = memory?.trackingBox,
            skinBox = memory?.skinBox,
            swimwearBox = memory?.swimwearBox,
            skinContour = memory?.skinContour,
            swimwearContour = memory?.swimwearContour,
            skinCells = memory?.skinCells,
            swimwearCells = memory?.swimwearCells,
            colorComboBox = memory?.colorComboBox,
            confidence = 0f
        )
    }

    private fun updateWithDirectColorTemplate(
        frame: Bitmap,
        samplingFrame: SamplingFrame,
        nowMs: Long,
        activeTemplate: ColorTrackingTemplate
    ): TrackedPerson {
        val searchWindow = buildPredictedSearchWindow(frame, activeTemplate)
        val correction = buildColorCorrectionProfile(samplingFrame, searchWindow)
        val waterColor = estimateWaterBackgroundColor(samplingFrame, searchWindow, correction)
        val selectedRegion = detectColorRegion(
            samplingFrame = samplingFrame,
            trackingBox = searchWindow,
            referenceColor = activeTemplate.swimwearColor,
            referencePalette = null,
            template = null,
            correction = correction,
            bandTopRatio = 0f,
            bandBottomRatio = 1f,
            excludedColor = waterColor,
            distanceThreshold = DIRECT_COLOR_THRESHOLD,
            requireCompact = true,
            preferredPoint = lastTracked?.trackingBox?.centerPoint() ?: selectedPoint,
            expectedBox = lastTracked?.trackingBox,
            cols = 18,
            rows = 20
        )
        if (selectedRegion != null && isMotionPlausible(selectedRegion.box)) {
            val stabilizedBox = stabilizeTrackingBox(selectedRegion.box, frame.width.toFloat(), frame.height.toFloat())
            val regionColor = extractAverageColor(samplingFrame, stabilizedBox, correction)
            val tracked = TrackedPerson(
                poseResult = null,
                mode = DetectionMode.COLOR_TRACK,
                trackId = lastTracked?.trackId ?: nextTrackId++,
                averageColor = regionColor,
                anchorColor = regionColor,
                skinColor = null,
                swimwearColor = activeTemplate.swimwearColor,
                trackingBox = stabilizedBox,
                skinBox = null,
                swimwearBox = stabilizedBox,
                skinContour = null,
                swimwearContour = selectedRegion.contour,
                skinCells = null,
                swimwearCells = selectedRegion.cells,
                colorComboBox = stabilizedBox,
                confidence = 0.84f
            )
            commitTracked(tracked, nowMs)
            return tracked
        }

        val globalRecovery = if (nowMs - lastGlobalRecoveryAtMs >= GLOBAL_RECOVERY_INTERVAL_MS) {
            findGlobalDirectColorRegion(frame, samplingFrame, activeTemplate)?.also {
                lastGlobalRecoveryAtMs = nowMs
            }
        } else {
            null
        }
        if (globalRecovery != null) {
            val tracked = TrackedPerson(
                poseResult = null,
                mode = DetectionMode.COLOR_TRACK,
                trackId = lastTracked?.trackId ?: nextTrackId++,
                averageColor = globalRecovery.first,
                anchorColor = globalRecovery.first,
                skinColor = null,
                swimwearColor = activeTemplate.swimwearColor,
                trackingBox = globalRecovery.second.box,
                skinBox = null,
                swimwearBox = globalRecovery.second.box,
                skinContour = null,
                swimwearContour = globalRecovery.second.contour,
                skinCells = null,
                swimwearCells = globalRecovery.second.cells,
                colorComboBox = globalRecovery.second.box,
                confidence = 0.56f
            )
            commitTracked(tracked, nowMs)
            return tracked
        }

        val memory = lastTracked
        if (memory != null && nowMs - lastUpdatedAtMs <= MEMORY_HOLD_MS * 3) {
            return memory.copy(mode = DetectionMode.TRACKED_MEMORY)
        }
        return TrackedPerson(
            poseResult = null,
            mode = DetectionMode.LOST,
            trackId = memory?.trackId ?: nextTrackId,
            averageColor = memory?.averageColor ?: activeTemplate.averageColor,
            anchorColor = memory?.anchorColor ?: activeTemplate.anchorColor,
            skinColor = null,
            swimwearColor = activeTemplate.swimwearColor,
            trackingBox = memory?.trackingBox,
            skinBox = null,
            swimwearBox = memory?.swimwearBox,
            skinContour = null,
            swimwearContour = memory?.swimwearContour,
            skinCells = null,
            swimwearCells = memory?.swimwearCells,
            colorComboBox = memory?.colorComboBox,
            confidence = 0f
        )
    }

    private fun findGlobalDirectColorRegion(
        frame: Bitmap,
        samplingFrame: SamplingFrame,
        activeTemplate: ColorTrackingTemplate
    ): Pair<Int, ColorRegionDetection>? {
        val fullFrame = floatArrayOf(0f, 0f, frame.width.toFloat(), frame.height.toFloat())
        val correction = buildColorCorrectionProfile(samplingFrame, fullFrame)
        val waterColor = estimateWaterBackgroundColor(samplingFrame, fullFrame, correction)
        val region = detectColorRegion(
            samplingFrame = samplingFrame,
            trackingBox = fullFrame,
            referenceColor = activeTemplate.swimwearColor,
            referencePalette = null,
            template = null,
            correction = correction,
            bandTopRatio = 0f,
            bandBottomRatio = 1f,
            excludedColor = waterColor,
            distanceThreshold = DIRECT_COLOR_GLOBAL_THRESHOLD,
            requireCompact = true,
            preferredPoint = lastTracked?.trackingBox?.centerPoint() ?: selectedPoint,
            expectedBox = null,
            cols = 20,
            rows = 22
        ) ?: return null
        return Pair(extractAverageColor(samplingFrame, region.box, correction), region)
    }

    private fun buildPredictedSearchWindow(bitmap: Bitmap, activeTemplate: ColorTrackingTemplate): FloatArray {
        val predicted = predictedMotionSnapshot()
        if (predicted == null) {
            return buildLockedSearchWindow(bitmap, activeTemplate)
        }
        val left = (predicted.centerX - predicted.width / 2f).coerceIn(0f, (bitmap.width - predicted.width).coerceAtLeast(0f))
        val top = (predicted.centerY - predicted.height / 2f).coerceIn(0f, (bitmap.height - predicted.height).coerceAtLeast(0f))
        return expandBox(
            floatArrayOf(left, top, left + predicted.width, top + predicted.height),
            bitmap.width.toFloat(),
            bitmap.height.toFloat(),
            0.78f,
            0.78f
        )
    }

    private fun predictedMotionSnapshot(): MotionSnapshot? {
        val current = lastTracked?.trackingBox?.toSnapshot() ?: return null
        val previous = previousTracked?.trackingBox?.toSnapshot() ?: return current
        val vx = current.centerX - previous.centerX
        val vy = current.centerY - previous.centerY
        return MotionSnapshot(
            centerX = current.centerX + vx * MOTION_PREDICTION_WEIGHT,
            centerY = current.centerY + vy * MOTION_PREDICTION_WEIGHT,
            width = current.width,
            height = current.height
        )
    }

    private fun FloatArray.toSnapshot(): MotionSnapshot {
        return MotionSnapshot(
            centerX = (this[0] + this[2]) / 2f,
            centerY = (this[1] + this[3]) / 2f,
            width = (this[2] - this[0]).coerceAtLeast(1f),
            height = (this[3] - this[1]).coerceAtLeast(1f)
        )
    }

    private fun isMotionPlausible(candidateBox: FloatArray): Boolean {
        val previous = lastTracked?.trackingBox ?: return true
        val previousSnapshot = previous.toSnapshot()
        val candidateSnapshot = candidateBox.toSnapshot()
        val maxDx = previousSnapshot.width * MAX_POSITION_JUMP_RATIO
        val maxDy = previousSnapshot.height * MAX_POSITION_JUMP_RATIO
        val dx = abs(candidateSnapshot.centerX - previousSnapshot.centerX)
        val dy = abs(candidateSnapshot.centerY - previousSnapshot.centerY)
        val widthRatio = candidateSnapshot.width / previousSnapshot.width
        val heightRatio = candidateSnapshot.height / previousSnapshot.height
        return dx <= maxDx &&
            dy <= maxDy &&
            widthRatio >= MIN_BOX_SCALE_RATIO &&
            widthRatio <= MAX_BOX_SCALE_RATIO &&
            heightRatio >= MIN_BOX_SCALE_RATIO &&
            heightRatio <= MAX_BOX_SCALE_RATIO
    }

    private fun stabilizeTrackingBox(candidateBox: FloatArray, frameWidth: Float, frameHeight: Float): FloatArray {
        val previous = lastTracked?.trackingBox ?: return candidateBox
        val left = previous[0] * TRACK_BOX_SMOOTHING + candidateBox[0] * (1f - TRACK_BOX_SMOOTHING)
        val top = previous[1] * TRACK_BOX_SMOOTHING + candidateBox[1] * (1f - TRACK_BOX_SMOOTHING)
        val right = previous[2] * TRACK_BOX_SMOOTHING + candidateBox[2] * (1f - TRACK_BOX_SMOOTHING)
        val bottom = previous[3] * TRACK_BOX_SMOOTHING + candidateBox[3] * (1f - TRACK_BOX_SMOOTHING)
        return floatArrayOf(
            left.coerceIn(0f, frameWidth - 1f),
            top.coerceIn(0f, frameHeight - 1f),
            right.coerceIn(left.coerceIn(0f, frameWidth - 1f) + 1f, frameWidth),
            bottom.coerceIn(top.coerceIn(0f, frameHeight - 1f) + 1f, frameHeight)
        )
    }

    private fun fallbackToColorOrMemory(frame: Bitmap, samplingFrame: SamplingFrame, nowMs: Long): TrackedPerson {
        if (initializing && lastTracked == null) {
            val initialBox = buildCenterInitializationBox(frame)
            val correction = buildColorCorrectionProfile(samplingFrame, initialBox)
            val averageColor = extractAverageColor(samplingFrame, initialBox, correction)
            val anchorColor = extractAnchorColor(samplingFrame, initialBox, correction)
            val provisionalSkinBox = buildSkinHintBox(initialBox)
            val provisionalSwimwearBox = buildSwimwearHintBox(initialBox)
            val sampledSkinColor = extractAverageColor(samplingFrame, provisionalSkinBox, correction)
            val sampledSwimwearColor = extractAverageColor(samplingFrame, provisionalSwimwearBox, correction)
            collectTemplateSample(
                trackingBox = initialBox,
                averageColor = averageColor,
                anchorColor = anchorColor,
                skinColor = sampledSkinColor,
                swimwearColor = sampledSwimwearColor
            )
            val skinRegion = detectColorRegion(samplingFrame, provisionalSkinBox, sampledSkinColor, listOf(sampledSkinColor), null, correction, 0f, 1f)
                ?: ColorRegionDetection(provisionalSkinBox, boxToContour(provisionalSkinBox), listOf(provisionalSkinBox))
            val swimwearRegion = detectColorRegion(
                samplingFrame,
                provisionalSwimwearBox,
                sampledSwimwearColor,
                buildColorPalette(samplingFrame, provisionalSwimwearBox, correction, sampledSwimwearColor),
                null,
                correction,
                0f,
                1f
            )
                ?: ColorRegionDetection(provisionalSwimwearBox, boxToContour(provisionalSwimwearBox), listOf(provisionalSwimwearBox))
            val tracked = TrackedPerson(
                poseResult = null,
                mode = DetectionMode.INITIALIZING,
                trackId = nextTrackId,
                averageColor = averageColor,
                anchorColor = anchorColor,
                skinColor = sampledSkinColor,
                swimwearColor = sampledSwimwearColor,
                trackingBox = initialBox,
                skinBox = skinRegion.box,
                swimwearBox = swimwearRegion.box,
                skinContour = null,
                swimwearContour = null,
                skinCells = skinRegion.cells,
                swimwearCells = swimwearRegion.cells,
                colorComboBox = buildColorComboBox(initialBox, skinRegion.box, swimwearRegion.box),
                confidence = 0.2f
            )
            commitTracked(tracked, nowMs)
            return tracked
        }

        val correction = buildColorCorrectionProfile(samplingFrame, lastTracked?.trackingBox)
        val colorTracked = template?.let { findByColorTemplate(samplingFrame, it, correction) }
        if (colorTracked != null) {
            val poseForHint = lastTracked?.poseResult
            val provisionalSkinBox = poseForHint?.let { buildPoseAwareSkinHintBox(it, colorTracked.third) }
                ?: buildSkinHintBox(colorTracked.third)
            val provisionalSwimwearBox = poseForHint?.let { buildPoseAwareSwimwearHintBox(it, colorTracked.third) }
                ?: buildSwimwearHintBox(colorTracked.third)
            val skinRegion = detectColorRegion(
                samplingFrame = samplingFrame,
                trackingBox = provisionalSkinBox,
                referenceColor = template?.skinColor ?: colorTracked.first,
                referencePalette = listOf(template?.skinColor ?: colorTracked.first),
                template = null,
                correction = correction,
                bandTopRatio = 0f,
                bandBottomRatio = 1f
            ) ?: ColorRegionDetection(provisionalSkinBox, boxToContour(provisionalSkinBox), listOf(provisionalSkinBox))
            val swimwearRegion = detectColorRegion(
                samplingFrame = samplingFrame,
                trackingBox = provisionalSwimwearBox,
                referenceColor = template?.swimwearColor ?: colorTracked.second,
                referencePalette = template?.swimwearPalette ?: listOf(template?.swimwearColor ?: colorTracked.second),
                template = template,
                correction = correction,
                bandTopRatio = 0f,
                bandBottomRatio = 1f
            ) ?: ColorRegionDetection(provisionalSwimwearBox, boxToContour(provisionalSwimwearBox), listOf(provisionalSwimwearBox))
            val colorComboBox = buildColorComboBox(colorTracked.third, skinRegion.box, swimwearRegion.box)
            val tracked = TrackedPerson(
                poseResult = null,
                mode = DetectionMode.COLOR_TRACK,
                trackId = lastTracked?.trackId ?: nextTrackId++,
                averageColor = colorTracked.first,
                anchorColor = colorTracked.second,
                skinColor = template?.skinColor,
                swimwearColor = template?.swimwearColor,
                trackingBox = colorComboBox.copyOf(),
                skinBox = skinRegion.box,
                swimwearBox = swimwearRegion.box,
                skinContour = null,
                swimwearContour = null,
                skinCells = skinRegion.cells,
                swimwearCells = swimwearRegion.cells,
                colorComboBox = colorComboBox,
                confidence = 0.45f
            )
            commitTracked(tracked, nowMs)
            return tracked
        }

        val memory = lastTracked
        if (memory != null && nowMs - lastUpdatedAtMs <= MEMORY_HOLD_MS) {
            return memory.copy(mode = DetectionMode.TRACKED_MEMORY)
        }

        val lostTrackId = memory?.trackId ?: nextTrackId
        return TrackedPerson(
            poseResult = null,
            mode = DetectionMode.LOST,
            trackId = lostTrackId,
            averageColor = memory?.averageColor,
            anchorColor = memory?.anchorColor,
            skinColor = memory?.skinColor,
            swimwearColor = memory?.swimwearColor,
            trackingBox = memory?.trackingBox,
            skinBox = memory?.skinBox,
            swimwearBox = memory?.swimwearBox,
            skinContour = memory?.skinContour,
            swimwearContour = memory?.swimwearContour,
            skinCells = memory?.skinCells,
            swimwearCells = memory?.swimwearCells,
            colorComboBox = memory?.colorComboBox,
            confidence = 0f
        )
    }

    private fun buildCenterInitializationBox(frame: Bitmap): FloatArray {
        val width = frame.width.toFloat()
        val height = frame.height.toFloat()
        val boxWidth = width * 0.42f
        val boxHeight = height * 0.52f
        val anchorX = selectedPoint?.getOrNull(0) ?: width / 2f
        val anchorY = selectedPoint?.getOrNull(1) ?: height / 2f
        val left = (anchorX - boxWidth / 2f).coerceIn(0f, (width - boxWidth).coerceAtLeast(0f))
        val top = (anchorY - boxHeight / 2f).coerceIn(0f, (height - boxHeight).coerceAtLeast(0f))
        return floatArrayOf(left, top, left + boxWidth, top + boxHeight)
    }

    private fun matchesSelectedPoint(box: FloatArray): Boolean {
        val point = selectedPoint ?: return true
        val marginX = (box[2] - box[0]) * 0.18f
        val marginY = (box[3] - box[1]) * 0.18f
        return point[0] in (box[0] - marginX)..(box[2] + marginX) &&
            point[1] in (box[1] - marginY)..(box[3] + marginY)
    }

    private fun refineModeWithPriority(mode: DetectionMode): DetectionMode {
        return when (mode) {
            DetectionMode.INITIALIZING -> DetectionMode.INITIALIZING
            DetectionMode.COLOR_TRACK -> DetectionMode.COLOR_TRACK
            DetectionMode.FULL_BODY -> DetectionMode.FULL_BODY
            DetectionMode.LEG_ONLY -> DetectionMode.LEG_ONLY
            DetectionMode.FOOT_ONLY -> DetectionMode.FOOT_ONLY
            DetectionMode.TRACKED_MEMORY -> DetectionMode.TRACKED_MEMORY
            DetectionMode.LOST -> DetectionMode.LOST
        }
    }

    private fun resolveMode(result: PoseResult): DetectionMode {
        val visibleCount = result.keyPoints.count { it.confidence > VISIBILITY_THRESHOLD }
        val torsoVisible = listOf(5, 6, 11, 12).count { result.keyPoints.getOrNull(it)?.confidence ?: 0f > VISIBILITY_THRESHOLD }
        val leftLegVisible = hasLeg(result, 11, 13, 15)
        val rightLegVisible = hasLeg(result, 12, 14, 16)
        val footVisible = hasFoot(result, 15) || hasFoot(result, 16)

        return when {
            visibleCount >= 8 || torsoVisible >= 3 -> DetectionMode.FULL_BODY
            leftLegVisible || rightLegVisible -> DetectionMode.LEG_ONLY
            footVisible -> DetectionMode.FOOT_ONLY
            else -> DetectionMode.LOST
        }
    }

    private fun hasLeg(result: PoseResult, hip: Int, knee: Int, ankle: Int): Boolean {
        return listOf(hip, knee, ankle).all { result.keyPoints.getOrNull(it)?.confidence ?: 0f > VISIBILITY_THRESHOLD }
    }

    private fun hasFoot(result: PoseResult, ankle: Int): Boolean {
        return result.keyPoints.getOrNull(ankle)?.confidence ?: 0f > VISIBILITY_THRESHOLD
    }

    private fun smoothPose(previous: PoseResult, current: PoseResult): PoseResult {
        if (previous.keyPoints.size != current.keyPoints.size) {
            return current
        }
        val smoothed = current.keyPoints.mapIndexed { index, point ->
            val prev = previous.keyPoints[index]
            if (point.confidence > VISIBILITY_THRESHOLD && prev.confidence > VISIBILITY_THRESHOLD) {
                KeyPoint(
                    x = prev.x * SMOOTHING_WEIGHT + point.x * (1f - SMOOTHING_WEIGHT),
                    y = prev.y * SMOOTHING_WEIGHT + point.y * (1f - SMOOTHING_WEIGHT),
                    confidence = maxOf(prev.confidence, point.confidence)
                )
            } else {
                point
            }
        }
        return current.copy(keyPoints = smoothed)
    }

    private fun deriveTrackingBox(
        result: PoseResult,
        frameWidth: Float,
        frameHeight: Float
    ): FloatArray {
        val visiblePoints = result.keyPoints.filter { it.confidence > VISIBILITY_THRESHOLD }
        if (visiblePoints.isEmpty()) {
            return result.boundingBox.copyOf()
        }

        val minX = visiblePoints.minOf { it.x }
        val minY = visiblePoints.minOf { it.y }
        val maxX = visiblePoints.maxOf { it.x }
        val maxY = visiblePoints.maxOf { it.y }
        val pointWidth = (maxX - minX).coerceAtLeast(24f)
        val pointHeight = (maxY - minY).coerceAtLeast(24f)

        val fallback = result.boundingBox
        val left = minOf(minX - pointWidth * 0.18f, fallback[0])
        val top = minOf(minY - pointHeight * 0.22f, fallback[1])
        val right = maxOf(maxX + pointWidth * 0.18f, fallback[2])
        val bottom = maxOf(maxY + pointHeight * 0.18f, fallback[3])

        return floatArrayOf(
            left.coerceIn(0f, frameWidth),
            top.coerceIn(0f, frameHeight),
            right.coerceIn(0f, frameWidth),
            bottom.coerceIn(0f, frameHeight)
        )
    }

    private fun buildSamplingFrame(bitmap: Bitmap): SamplingFrame {
        val maxSide = ANALYSIS_MAX_SIDE
        val longest = max(bitmap.width, bitmap.height)
        val scale = if (longest <= maxSide) 1f else maxSide.toFloat() / longest.toFloat()
        val sampleWidth = (bitmap.width * scale).roundToInt().coerceAtLeast(1)
        val sampleHeight = (bitmap.height * scale).roundToInt().coerceAtLeast(1)
        val sampledBitmap = if (sampleWidth == bitmap.width && sampleHeight == bitmap.height) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, sampleWidth, sampleHeight, true)
        }
        val pixels = IntArray(sampleWidth * sampleHeight)
        sampledBitmap.getPixels(pixels, 0, sampleWidth, 0, 0, sampleWidth, sampleHeight)
        return SamplingFrame(
            sourceWidth = bitmap.width,
            sourceHeight = bitmap.height,
            sampleWidth = sampleWidth,
            sampleHeight = sampleHeight,
            pixels = pixels
        )
    }


    private fun mapSourceXToSample(samplingFrame: SamplingFrame, x: Float): Int {
        return (x * samplingFrame.sampleWidth / samplingFrame.sourceWidth.toFloat()).roundToInt()
    }

    private fun mapSourceYToSample(samplingFrame: SamplingFrame, y: Float): Int {
        return (y * samplingFrame.sampleHeight / samplingFrame.sourceHeight.toFloat()).roundToInt()
    }

    private fun extractAverageColor(samplingFrame: SamplingFrame, box: FloatArray, correction: ColorCorrectionProfile): Int {
        return sampleColor(samplingFrame, box, correction, 0.0f, 1.0f)
    }

    private fun extractAnchorColor(samplingFrame: SamplingFrame, box: FloatArray, correction: ColorCorrectionProfile): Int {
        return sampleColor(samplingFrame, box, correction, 0.25f, 0.75f)
    }

    private fun sampleColor(
        samplingFrame: SamplingFrame,
        box: FloatArray,
        correction: ColorCorrectionProfile,
        startRatio: Float,
        endRatio: Float
    ): Int {
        val left = mapSourceXToSample(samplingFrame, box[0]).coerceIn(0, samplingFrame.sampleWidth - 1)
        val top = mapSourceYToSample(samplingFrame, box[1]).coerceIn(0, samplingFrame.sampleHeight - 1)
        val right = mapSourceXToSample(samplingFrame, box[2]).coerceIn(left + 1, samplingFrame.sampleWidth)
        val bottom = mapSourceYToSample(samplingFrame, box[3]).coerceIn(top + 1, samplingFrame.sampleHeight)
        val sampleTop = (top + (bottom - top) * startRatio).roundToInt().coerceIn(top, bottom - 1)
        val sampleBottom = (top + (bottom - top) * endRatio).roundToInt().coerceIn(sampleTop + 1, bottom)

        var red = 0L
        var green = 0L
        var blue = 0L
        var count = 0L
        val stepX = ((right - left) / 24).coerceAtLeast(1)
        val stepY = ((sampleBottom - sampleTop) / 18).coerceAtLeast(1)

        for (y in sampleTop until sampleBottom step stepY) {
            for (x in left until right step stepX) {
                val corrected = correctPixel(samplingFrame.pixels[y * samplingFrame.sampleWidth + x], correction)
                red += Color.red(corrected)
                green += Color.green(corrected)
                blue += Color.blue(corrected)
                count++
            }
        }

        if (count == 0L) {
            return Color.WHITE
        }
        return Color.rgb((red / count).toInt(), (green / count).toInt(), (blue / count).toInt())
    }

    fun isColorConsistent(candidateAverageColor: Int?, candidateAnchorColor: Int?): Boolean {
        val previous = lastTracked ?: return true
        val avgOk = previous.averageColor == null || candidateAverageColor == null || colorDistance(previous.averageColor, candidateAverageColor) <= COLOR_DISTANCE_THRESHOLD
        val anchorOk = previous.anchorColor == null || candidateAnchorColor == null || colorDistance(previous.anchorColor, candidateAnchorColor) <= COLOR_DISTANCE_THRESHOLD
        return avgOk && anchorOk
    }

    private fun isColorConsistentWithTemplate(
        candidateAverageColor: Int?,
        candidateAnchorColor: Int?,
        correction: ColorCorrectionProfile
    ): Boolean {
        val activeTemplate = template ?: return true
        val avgOk = candidateAverageColor == null || paletteMatchesReference(
            candidateAverageColor,
            activeTemplate.averageColor,
            HSV_DISTANCE_THRESHOLD,
            activeTemplate.swimwearPalette,
            activeTemplate
        )
        val anchorOk = candidateAnchorColor == null || paletteMatchesReference(
            candidateAnchorColor,
            activeTemplate.anchorColor,
            HSV_DISTANCE_THRESHOLD,
            activeTemplate.swimwearPalette,
            activeTemplate
        )
        return avgOk && anchorOk
    }

    private fun collectTemplateSample(
        trackingBox: FloatArray,
        averageColor: Int,
        anchorColor: Int,
        skinColor: Int,
        swimwearColor: Int
    ) {
        if (!initializing) return
        initAverageColors += averageColor
        initAnchorColors += anchorColor
        initSkinColors += skinColor
        initSwimwearColors += swimwearColor
        initWidths += (trackingBox[2] - trackingBox[0]).coerceAtLeast(1f)
        initHeights += (trackingBox[3] - trackingBox[1]).coerceAtLeast(1f)
    }

    private fun averageColor(colors: List<Int>): Int {
        var red = 0L
        var green = 0L
        var blue = 0L
        colors.forEach {
            red += Color.red(it)
            green += Color.green(it)
            blue += Color.blue(it)
        }
        val count = colors.size.coerceAtLeast(1)
        return Color.rgb((red / count).toInt(), (green / count).toInt(), (blue / count).toInt())
    }

    private fun buildPaletteFromSamples(colors: List<Int>): List<Int> {
        val baseHue = dominantHue(colors)
        val deduped = mutableListOf<Int>()
        colors.forEach { color ->
            if (!matchesHueBand(color, baseHue, DEFAULT_HUE_TOLERANCE)) return@forEach
            if (deduped.none { regionTemplateDistance(it, color) <= PALETTE_DEDUP_THRESHOLD }) {
                deduped += color
            }
        }
        return deduped.take(MAX_TEMPLATE_COLORS).ifEmpty { colors.distinct().filter { matchesHueBand(it, baseHue, DEFAULT_HUE_TOLERANCE) }.take(MAX_TEMPLATE_COLORS) }
    }

    private fun buildColorPalette(
        samplingFrame: SamplingFrame,
        box: FloatArray,
        correction: ColorCorrectionProfile,
        fallbackColor: Int
    ): List<Int> {
        val width = (box[2] - box[0]).coerceAtLeast(12f)
        val height = (box[3] - box[1]).coerceAtLeast(12f)
        val sampleBoxes = listOf(
            floatArrayOf(box[0], box[1], box[2], box[3]),
            floatArrayOf(box[0] + width * 0.18f, box[1] + height * 0.18f, box[0] + width * 0.52f, box[1] + height * 0.52f),
            floatArrayOf(box[0] + width * 0.48f, box[1] + height * 0.16f, box[0] + width * 0.82f, box[1] + height * 0.48f),
            floatArrayOf(box[0] + width * 0.18f, box[1] + height * 0.50f, box[0] + width * 0.52f, box[1] + height * 0.84f),
            floatArrayOf(box[0] + width * 0.48f, box[1] + height * 0.50f, box[0] + width * 0.82f, box[1] + height * 0.84f)
        ).map { normalizeBox(it, box) }

        val sampled = sampleBoxes.map { sampleBox ->
            extractAverageColor(samplingFrame, sampleBox, correction)
        }.toMutableList()
        sampled += fallbackColor
        return buildPaletteFromSamples(sampled)
    }

    private fun normalizeBox(candidate: FloatArray, bounds: FloatArray): FloatArray {
        val left = candidate[0].coerceIn(bounds[0], bounds[2] - 1f)
        val top = candidate[1].coerceIn(bounds[1], bounds[3] - 1f)
        val right = candidate[2].coerceIn(left + 1f, bounds[2])
        val bottom = candidate[3].coerceIn(top + 1f, bounds[3])
        return floatArrayOf(left, top, right, bottom)
    }

    private fun dominantHue(colors: List<Int>): Float {
        if (colors.isEmpty()) return 0f
        val filtered = colors.filterNot { isLowSaturation(it) }
        val source = if (filtered.isNotEmpty()) filtered else colors
        val x = source.sumOf { kotlin.math.cos(Math.toRadians(colorHue(it).toDouble())) }.toFloat()
        val y = source.sumOf { kotlin.math.sin(Math.toRadians(colorHue(it).toDouble())) }.toFloat()
        val angle = Math.toDegrees(kotlin.math.atan2(y.toDouble(), x.toDouble())).toFloat()
        return if (angle < 0f) angle + 360f else angle
    }

    private fun paletteMinSaturation(colors: List<Int>): Float = colors.minOfOrNull { colorSaturation(it) }?.coerceAtLeast(0.10f) ?: 0.10f

    private fun paletteMaxSaturation(colors: List<Int>): Float = colors.maxOfOrNull { colorSaturation(it) }?.coerceAtMost(1f) ?: 1f

    private fun paletteMinValue(colors: List<Int>): Float = colors.minOfOrNull { colorValue(it) }?.coerceAtLeast(0.08f) ?: 0.08f

    private fun paletteMaxValue(colors: List<Int>): Float = colors.maxOfOrNull { colorValue(it) }?.coerceAtMost(1f) ?: 1f

    private fun colorHue(color: Int): Float {
        val hsv = FloatArray(3)
        Color.colorToHSV(color, hsv)
        return hsv[0]
    }

    private fun colorSaturation(color: Int): Float {
        val hsv = FloatArray(3)
        Color.colorToHSV(color, hsv)
        return hsv[1]
    }

    private fun colorValue(color: Int): Float {
        val hsv = FloatArray(3)
        Color.colorToHSV(color, hsv)
        return hsv[2]
    }

    private fun hueDistance(a: Float, b: Float): Float {
        val diff = abs(a - b)
        return minOf(diff, 360f - diff)
    }

    private fun matchesHueBand(color: Int, baseHue: Float, hueTolerance: Float): Boolean {
        if (isLowSaturation(color)) return false
        return hueDistance(colorHue(color), baseHue) <= hueTolerance
    }

    private fun buildSegmentedRegions(
        frame: Bitmap,
        samplingFrame: SamplingFrame,
        correction: ColorCorrectionProfile,
        waterColor: Int
    ): List<SegmentedRegion> {
        val cols = SEGMENT_GRID_COLS
        val rows = SEGMENT_GRID_ROWS
        val cellWidth = (frame.width.toFloat() / cols.toFloat()).coerceAtLeast(1f)
        val cellHeight = (frame.height.toFloat() / rows.toFloat()).coerceAtLeast(1f)
        val cellColors = Array(rows) { IntArray(cols) }
        val visited = Array(rows) { BooleanArray(cols) }
        val regions = mutableListOf<SegmentedRegion>()

        for (row in 0 until rows) {
            for (col in 0 until cols) {
                val box = floatArrayOf(
                    col.toFloat() * cellWidth,
                    row.toFloat() * cellHeight,
                    ((col + 1).toFloat() * cellWidth).coerceAtMost(frame.width.toFloat()),
                    ((row + 1).toFloat() * cellHeight).coerceAtMost(frame.height.toFloat())
                )
                cellColors[row][col] = extractAverageColor(samplingFrame, box, correction)
            }
        }

        val directions = arrayOf(
            intArrayOf(1, 0), intArrayOf(-1, 0),
            intArrayOf(0, 1), intArrayOf(0, -1)
        )

        for (row in 0 until rows) {
            for (col in 0 until cols) {
                if (visited[row][col]) continue
                val queue = ArrayDeque<Pair<Int, Int>>()
                val component = mutableListOf<Pair<Int, Int>>()
                var minRow = row
                var minCol = col
                var maxRow = row
                var maxCol = col
                queue.add(Pair(row, col))
                visited[row][col] = true
                while (queue.isNotEmpty()) {
                    val (currentRow, currentCol) = queue.removeFirst()
                    component += Pair(currentRow, currentCol)
                    minRow = minOf(minRow, currentRow)
                    minCol = minOf(minCol, currentCol)
                    maxRow = maxOf(maxRow, currentRow)
                    maxCol = maxOf(maxCol, currentCol)
                    directions.forEach { direction ->
                        val nextRow = currentRow + direction[0]
                        val nextCol = currentCol + direction[1]
                        if (nextRow !in 0 until rows || nextCol !in 0 until cols) return@forEach
                        if (visited[nextRow][nextCol]) return@forEach
                        if (!shouldMergeSegmentCells(
                                cellColors[currentRow][currentCol],
                                cellColors[nextRow][nextCol],
                                waterColor
                            )
                        ) return@forEach
                        visited[nextRow][nextCol] = true
                        queue.add(Pair(nextRow, nextCol))
                    }
                }
                if (component.size < MIN_SEGMENT_COMPONENT_CELLS) continue

                val cells = component.map { (r, c) ->
                    floatArrayOf(
                        c.toFloat() * cellWidth,
                        r.toFloat() * cellHeight,
                        ((c + 1).toFloat() * cellWidth).coerceAtMost(frame.width.toFloat()),
                        ((r + 1).toFloat() * cellHeight).coerceAtMost(frame.height.toFloat())
                    )
                }
                val box = floatArrayOf(
                    minCol.toFloat() * cellWidth,
                    minRow.toFloat() * cellHeight,
                    ((maxCol + 1).toFloat() * cellWidth).coerceAtMost(frame.width.toFloat()),
                    ((maxRow + 1).toFloat() * cellHeight).coerceAtMost(frame.height.toFloat())
                )
                val representativeColor = averageColor(component.map { (r, c) -> cellColors[r][c] })
                val detection = ColorRegionDetection(
                    box = box,
                    contour = buildComponentContour(component, 0f, 0f, cellWidth, cellHeight),
                    cells = cells
                )
                val area = (box[2] - box[0]) * (box[3] - box[1])
                if (isBackgroundSegment(representativeColor, waterColor, area, frame.width.toFloat() * frame.height.toFloat())) {
                    continue
                }
                regions += SegmentedRegion(detection, representativeColor, area)
            }
        }

        return regions
    }

    private fun pickSegmentForPoint(regions: List<SegmentedRegion>, point: FloatArray): SegmentedRegion? {
        regions.firstOrNull { pointInsideBox(point, it.detection.box) }?.let { return it }
        return regions.minByOrNull { squaredDistanceToBoxCenter(it.detection.box, point) }
    }

    private fun pickBestSegment(
        regions: List<SegmentedRegion>,
        referencePalette: List<Int>,
        referenceColor: Int,
        expectedBox: FloatArray?,
        preferredPoint: FloatArray?,
        template: ColorTrackingTemplate? = null
    ): SegmentedRegion? {
        return regions.asSequence()
            .filter { region -> template == null || matchesTemplateHueBand(region.representativeColor, template) }
            .maxByOrNull { region ->
            var score = region.area * SEGMENT_AREA_SCORE_WEIGHT
            val colorScore = bestPaletteDistance(region.representativeColor, referencePalette.ifEmpty { listOf(referenceColor) })
            score -= colorScore * SEGMENT_COLOR_PENALTY_WEIGHT
            if (expectedBox != null) {
                score += intersectionOverUnion(region.detection.box, expectedBox) * SEGMENT_IOU_SCORE_WEIGHT
                val expectedArea = ((expectedBox[2] - expectedBox[0]) * (expectedBox[3] - expectedBox[1])).coerceAtLeast(1f)
                val areaRatio = max(region.area / expectedArea, expectedArea / region.area)
                score -= abs(areaRatio - 1f) * SEGMENT_AREA_RATIO_PENALTY
            }
            if (preferredPoint != null) {
                score -= squaredDistanceToBoxCenter(region.detection.box, preferredPoint) / SEGMENT_DISTANCE_NORMALIZER
            }
            score
        }
    }

    private fun isStrongSegmentMatch(
        region: SegmentedRegion,
        template: ColorTrackingTemplate,
        previousBox: FloatArray?
    ): Boolean {
        if (!matchesTemplateHueBand(region.representativeColor, template)) return false
        val paletteDistance = bestPaletteDistance(region.representativeColor, template.swimwearPalette.ifEmpty { listOf(template.swimwearColor) })
        if (paletteDistance > FAST_MOTION_MAX_COLOR_DISTANCE) return false
        if (previousBox == null) return true
        val previousArea = ((previousBox[2] - previousBox[0]) * (previousBox[3] - previousBox[1])).coerceAtLeast(1f)
        val areaRatio = max(region.area / previousArea, previousArea / region.area)
        return areaRatio <= FAST_MOTION_MAX_AREA_RATIO
    }

    private fun shouldMergeSegmentCells(a: Int, b: Int, waterColor: Int): Boolean {
        if (backgroundDistance(a, waterColor) <= SEGMENT_WATER_EXCLUSION_THRESHOLD &&
            backgroundDistance(b, waterColor) <= SEGMENT_WATER_EXCLUSION_THRESHOLD
        ) {
            return true
        }
        return regionTemplateDistance(a, b) <= SEGMENT_MERGE_COLOR_THRESHOLD &&
            abs(luma(a) - luma(b)) <= SEGMENT_MERGE_LUMA_THRESHOLD
    }

    private fun isBackgroundSegment(
        color: Int,
        waterColor: Int,
        area: Float,
        frameArea: Float
    ): Boolean {
        val closeToWater = backgroundDistance(color, waterColor) <= SEGMENT_WATER_EXCLUSION_THRESHOLD || isBlueishWater(color)
        val largeEnough = area >= frameArea * SEGMENT_BACKGROUND_AREA_RATIO
        return closeToWater && largeEnough
    }

    private fun isBlueishWater(color: Int): Boolean {
        val hsv = FloatArray(3)
        Color.colorToHSV(color, hsv)
        return hsv[0] in 150f..245f && hsv[1] >= 0.15f
    }

    private fun pointInsideBox(point: FloatArray, box: FloatArray): Boolean {
        return point[0] in box[0]..box[2] && point[1] in box[1]..box[3]
    }

    private fun squaredDistanceToBoxCenter(box: FloatArray, point: FloatArray): Float {
        val centerX = (box[0] + box[2]) * 0.5f
        val centerY = (box[1] + box[3]) * 0.5f
        val dx = centerX - point[0]
        val dy = centerY - point[1]
        return dx * dx + dy * dy
    }

    private fun bestPaletteDistance(candidate: Int, palette: List<Int>): Int {
        return palette.minOfOrNull { regionTemplateDistance(candidate, it) } ?: Int.MAX_VALUE
    }


    private fun findByColorTemplate(
        samplingFrame: SamplingFrame,
        activeTemplate: ColorTrackingTemplate,
        correction: ColorCorrectionProfile
    ): Triple<Int, Int, FloatArray>? {
        val previousBox = lastTracked?.trackingBox ?: return null
        val boxWidth = activeTemplate.meanWidth.coerceAtLeast(32f)
        val boxHeight = activeTemplate.meanHeight.coerceAtLeast(32f)
        val centerX = ((previousBox[0] + previousBox[2]) / 2f).roundToInt()
        val centerY = ((previousBox[1] + previousBox[3]) / 2f).roundToInt()
        val searchRadiusX = (boxWidth * 0.45f).roundToInt().coerceAtLeast(24)
        val searchRadiusY = (boxHeight * 0.45f).roundToInt().coerceAtLeast(24)
        val stride = (boxWidth / 6f).roundToInt().coerceAtLeast(8)

        var bestScore = Int.MAX_VALUE
        var bestBox: FloatArray? = null
        var bestAverage = activeTemplate.averageColor
        var bestAnchor = activeTemplate.anchorColor

        for (dy in -searchRadiusY..searchRadiusY step stride) {
            for (dx in -searchRadiusX..searchRadiusX step stride) {
                val left = (centerX + dx - boxWidth / 2f).coerceIn(0f, (samplingFrame.sourceWidth - boxWidth).coerceAtLeast(0f))
                val top = (centerY + dy - boxHeight / 2f).coerceIn(0f, (samplingFrame.sourceHeight - boxHeight).coerceAtLeast(0f))
                val candidate = floatArrayOf(left, top, left + boxWidth, top + boxHeight)
                val average = extractAverageColor(samplingFrame, candidate, correction)
                val anchor = extractAnchorColor(samplingFrame, candidate, correction)
                val score = hsvDistance(activeTemplate.averageColor, average) + hsvDistance(activeTemplate.anchorColor, anchor)
                if (score < bestScore) {
                    bestScore = score
                    bestBox = candidate
                    bestAverage = average
                    bestAnchor = anchor
                }
            }
        }

        return if (bestBox != null && bestScore <= HSV_SEARCH_THRESHOLD) {
            Triple(bestAverage, bestAnchor, bestBox)
        } else {
            null
        }
    }

    private fun buildLockedSearchWindow(bitmap: Bitmap, activeTemplate: ColorTrackingTemplate): FloatArray {
        val previous = lastTracked?.trackingBox
        if (previous != null) {
            return expandBox(previous, bitmap.width.toFloat(), bitmap.height.toFloat(), 0.60f, 0.60f)
        }
        selectedPoint?.let { point ->
            return buildPointCenteredSearchWindow(bitmap, point)
        }
        val boxWidth = max(activeTemplate.meanWidth, bitmap.width * 0.20f)
        val boxHeight = max(activeTemplate.meanHeight, bitmap.height * 0.20f)
        val centerX = bitmap.width / 2f
        val centerY = bitmap.height / 2f
        val left = (centerX - boxWidth / 2f).coerceIn(0f, (bitmap.width - boxWidth).coerceAtLeast(0f))
        val top = (centerY - boxHeight / 2f).coerceIn(0f, (bitmap.height - boxHeight).coerceAtLeast(0f))
        return floatArrayOf(left, top, left + boxWidth, top + boxHeight)
    }

    private fun buildPointCenteredSearchWindow(bitmap: Bitmap, point: FloatArray): FloatArray {
        val boxWidth = bitmap.width * 0.34f
        val boxHeight = bitmap.height * 0.34f
        val left = (point[0] - boxWidth / 2f).coerceIn(0f, (bitmap.width - boxWidth).coerceAtLeast(0f))
        val top = (point[1] - boxHeight / 2f).coerceIn(0f, (bitmap.height - boxHeight).coerceAtLeast(0f))
        return floatArrayOf(left, top, left + boxWidth, top + boxHeight)
    }

    private fun buildTapSampleBox(bitmap: Bitmap, point: FloatArray): FloatArray {
        val boxWidth = bitmap.width * 0.06f
        val boxHeight = bitmap.height * 0.06f
        val left = (point[0] - boxWidth / 2f).coerceIn(0f, (bitmap.width - boxWidth).coerceAtLeast(0f))
        val top = (point[1] - boxHeight / 2f).coerceIn(0f, (bitmap.height - boxHeight).coerceAtLeast(0f))
        return floatArrayOf(left, top, left + boxWidth, top + boxHeight)
    }

    private fun expandBox(
        box: FloatArray,
        frameWidth: Float,
        frameHeight: Float,
        ratioX: Float,
        ratioY: Float
    ): FloatArray {
        val width = (box[2] - box[0]).coerceAtLeast(1f)
        val height = (box[3] - box[1]).coerceAtLeast(1f)
        val extraX = width * ratioX
        val extraY = height * ratioY
        val left = (box[0] - extraX).coerceAtLeast(0f)
        val top = (box[1] - extraY).coerceAtLeast(0f)
        val right = (box[2] + extraX).coerceAtMost(frameWidth)
        val bottom = (box[3] + extraY).coerceAtMost(frameHeight)
        return floatArrayOf(left, top, right.coerceAtLeast(left + 1f), bottom.coerceAtLeast(top + 1f))
    }

    private fun estimateWaterBackgroundColor(
        samplingFrame: SamplingFrame,
        searchWindow: FloatArray,
        correction: ColorCorrectionProfile
    ): Int {
        val left = searchWindow[0].roundToInt().coerceIn(0, samplingFrame.sourceWidth - 1)
        val top = searchWindow[1].roundToInt().coerceIn(0, samplingFrame.sourceHeight - 1)
        val right = searchWindow[2].roundToInt().coerceIn(left + 1, samplingFrame.sourceWidth)
        val bottom = searchWindow[3].roundToInt().coerceIn(top + 1, samplingFrame.sourceHeight)
        val border = (((right - left).coerceAtLeast(bottom - top)) * 0.14f).roundToInt().coerceAtLeast(4)
        val strips = listOf(
            floatArrayOf(left.toFloat(), top.toFloat(), right.toFloat(), (top + border).coerceAtMost(bottom).toFloat()),
            floatArrayOf(left.toFloat(), (bottom - border).coerceAtLeast(top).toFloat(), right.toFloat(), bottom.toFloat()),
            floatArrayOf(left.toFloat(), top.toFloat(), (left + border).coerceAtMost(right).toFloat(), bottom.toFloat()),
            floatArrayOf((right - border).coerceAtLeast(left).toFloat(), top.toFloat(), right.toFloat(), bottom.toFloat())
        )
        val colors = strips.map { extractAverageColor(samplingFrame, it, correction) }
        return averageColor(colors)
    }

    private fun colorDistance(a: Int, b: Int): Int {
        return abs(Color.red(a) - Color.red(b)) +
            abs(Color.green(a) - Color.green(b)) +
            abs(Color.blue(a) - Color.blue(b))
    }

    private fun hsvDistance(a: Int, b: Int): Int {
        val hsvA = FloatArray(3)
        val hsvB = FloatArray(3)
        Color.colorToHSV(a, hsvA)
        Color.colorToHSV(b, hsvB)
        val hueDiff = abs(hsvA[0] - hsvB[0]).let { minOf(it, 360f - it) } / 2f
        val satDiff = abs(hsvA[1] - hsvB[1]) * 100f
        val valueDiff = abs(hsvA[2] - hsvB[2]) * 60f
        return (hueDiff + satDiff + valueDiff).roundToInt()
    }

    private fun buildSkinHintBox(box: FloatArray): FloatArray {
        val left = box[0]
        val top = box[1]
        val right = box[2]
        val bottom = box[3]
        val width = right - left
        val height = bottom - top
        return floatArrayOf(
            left + width * 0.10f,
            top + height * 0.08f,
            right - width * 0.10f,
            top + height * 0.40f
        )
    }

    private fun buildSwimwearHintBox(box: FloatArray): FloatArray {
        val left = box[0]
        val top = box[1]
        val right = box[2]
        val bottom = box[3]
        val width = right - left
        val height = bottom - top
        return floatArrayOf(
            left + width * 0.14f,
            top + height * 0.36f,
            right - width * 0.14f,
            top + height * 0.72f
        )
    }

    private fun buildPoseAwareSkinHintBox(result: PoseResult, fallbackBox: FloatArray): FloatArray {
        val shoulders = listOfNotNull(visiblePoint(result, 5), visiblePoint(result, 6))
        val hips = listOfNotNull(visiblePoint(result, 11), visiblePoint(result, 12))
        val knees = listOfNotNull(visiblePoint(result, 13), visiblePoint(result, 14))

        if (shoulders.isNotEmpty() && hips.isNotEmpty()) {
            val left = (shoulders.minOf { it.x }.coerceAtMost(hips.minOf { it.x }) - bodyWidth(result, fallbackBox) * 0.12f)
            val right = (shoulders.maxOf { it.x }.coerceAtLeast(hips.maxOf { it.x }) + bodyWidth(result, fallbackBox) * 0.12f)
            val top = shoulders.minOf { it.y } - bodyHeight(result, fallbackBox) * 0.10f
            val bottom = hips.minOf { it.y } + bodyHeight(result, fallbackBox) * 0.08f
            return clampBox(floatArrayOf(left, top, right, bottom), fallbackBox)
        }

        if (hips.isNotEmpty() && knees.isNotEmpty()) {
            val left = hips.minOf { it.x } - bodyWidth(result, fallbackBox) * 0.10f
            val right = hips.maxOf { it.x } + bodyWidth(result, fallbackBox) * 0.10f
            val top = hips.minOf { it.y } - bodyHeight(result, fallbackBox) * 0.28f
            val bottom = knees.minOf { it.y } - bodyHeight(result, fallbackBox) * 0.06f
            return clampBox(floatArrayOf(left, top, right, bottom), fallbackBox)
        }

        return buildSkinHintBox(fallbackBox)
    }

    private fun buildPoseAwareSwimwearHintBox(result: PoseResult, fallbackBox: FloatArray): FloatArray {
        val hips = listOfNotNull(visiblePoint(result, 11), visiblePoint(result, 12))
        val knees = listOfNotNull(visiblePoint(result, 13), visiblePoint(result, 14))
        val shoulders = listOfNotNull(visiblePoint(result, 5), visiblePoint(result, 6))

        if (hips.isNotEmpty() && knees.isNotEmpty()) {
            val left = hips.minOf { it.x } - bodyWidth(result, fallbackBox) * 0.12f
            val right = hips.maxOf { it.x } + bodyWidth(result, fallbackBox) * 0.12f
            val top = hips.minOf { it.y } - bodyHeight(result, fallbackBox) * 0.06f
            val bottom = knees.minOf { it.y } + bodyHeight(result, fallbackBox) * 0.10f
            return clampBox(floatArrayOf(left, top, right, bottom), fallbackBox)
        }

        if (shoulders.isNotEmpty() && hips.isNotEmpty()) {
            val left = hips.minOf { it.x } - bodyWidth(result, fallbackBox) * 0.10f
            val right = hips.maxOf { it.x } + bodyWidth(result, fallbackBox) * 0.10f
            val top = shoulders.maxOf { it.y } + (hips.minOf { it.y } - shoulders.maxOf { it.y }) * 0.55f
            val bottom = hips.maxOf { it.y } + bodyHeight(result, fallbackBox) * 0.10f
            return clampBox(floatArrayOf(left, top, right, bottom), fallbackBox)
        }

        return buildSwimwearHintBox(fallbackBox)
    }

    private fun visiblePoint(result: PoseResult, index: Int): KeyPoint? {
        val point = result.keyPoints.getOrNull(index) ?: return null
        return point.takeIf { it.confidence > VISIBILITY_THRESHOLD }
    }

    private fun bodyWidth(result: PoseResult, fallbackBox: FloatArray): Float {
        val xs = result.keyPoints.filter { it.confidence > VISIBILITY_THRESHOLD }.map { it.x }
        return if (xs.isNotEmpty()) (xs.max() - xs.min()).coerceAtLeast(24f) else (fallbackBox[2] - fallbackBox[0]).coerceAtLeast(24f)
    }

    private fun bodyHeight(result: PoseResult, fallbackBox: FloatArray): Float {
        val ys = result.keyPoints.filter { it.confidence > VISIBILITY_THRESHOLD }.map { it.y }
        return if (ys.isNotEmpty()) (ys.max() - ys.min()).coerceAtLeast(24f) else (fallbackBox[3] - fallbackBox[1]).coerceAtLeast(24f)
    }

    private fun clampBox(box: FloatArray, fallbackBox: FloatArray): FloatArray {
        val left = box[0].coerceIn(fallbackBox[0], fallbackBox[2] - 1f)
        val top = box[1].coerceIn(fallbackBox[1], fallbackBox[3] - 1f)
        val right = box[2].coerceIn(left + 1f, fallbackBox[2])
        val bottom = box[3].coerceIn(top + 1f, fallbackBox[3])
        return floatArrayOf(left, top, right, bottom)
    }

    private fun detectColorRegion(
        samplingFrame: SamplingFrame,
        trackingBox: FloatArray,
        referenceColor: Int,
        referencePalette: List<Int>? = null,
        template: ColorTrackingTemplate? = null,
        correction: ColorCorrectionProfile,
        bandTopRatio: Float,
        bandBottomRatio: Float,
        excludedColor: Int? = null,
        distanceThreshold: Int = HSV_REGION_THRESHOLD,
        requireCompact: Boolean = false,
        preferredPoint: FloatArray? = null,
        expectedBox: FloatArray? = null,
        cols: Int = 14,
        rows: Int = 16
    ): ColorRegionDetection? {
        val left = trackingBox[0].roundToInt().coerceIn(0, samplingFrame.sourceWidth - 1)
        val right = trackingBox[2].roundToInt().coerceIn(left + 1, samplingFrame.sourceWidth)
        val top = trackingBox[1].roundToInt().coerceIn(0, samplingFrame.sourceHeight - 1)
        val bottom = trackingBox[3].roundToInt().coerceIn(top + 1, samplingFrame.sourceHeight)
        val bandTop = (top + (bottom - top) * bandTopRatio).roundToInt().coerceIn(top, bottom - 1)
        val bandBottom = (top + (bottom - top) * bandBottomRatio).roundToInt().coerceIn(bandTop + 1, bottom)

        val cellWidth = ((right - left).toFloat() / cols).coerceAtLeast(1f)
        val cellHeight = ((bandBottom - bandTop).toFloat() / rows).coerceAtLeast(1f)
        val mask = Array(rows) { BooleanArray(cols) }
        val cellColors = Array(rows) { IntArray(cols) }

        for (row in 0 until rows) {
            for (col in 0 until cols) {
                val sampleLeft = (left + col * cellWidth).roundToInt().coerceIn(left, right - 1)
                val sampleTop = (bandTop + row * cellHeight).roundToInt().coerceIn(bandTop, bandBottom - 1)
                val sampleRight = (left + (col + 1) * cellWidth).roundToInt().coerceIn(sampleLeft + 1, right)
                val sampleBottom = (bandTop + (row + 1) * cellHeight).roundToInt().coerceIn(sampleTop + 1, bandBottom)
                val cellBox = floatArrayOf(
                    sampleLeft.toFloat(),
                    sampleTop.toFloat(),
                    sampleRight.toFloat(),
                    sampleBottom.toFloat()
                )
                val cellColor = extractAverageColor(samplingFrame, cellBox, correction)
                cellColors[row][col] = cellColor
                val matchesReference = colorMatchesReference(cellColor, referenceColor, distanceThreshold, referencePalette, template)
                val tooCloseToBackground = excludedColor != null &&
                    backgroundDistance(cellColor, excludedColor) <= BACKGROUND_EXCLUSION_THRESHOLD
                if (matchesReference && !tooCloseToBackground && !looksLikeWater(cellColor, referenceColor, excludedColor)) {
                    mask[row][col] = true
                }
            }
        }

        val component = findPreferredComponent(
            mask = mask,
            cellColors = cellColors,
            referenceColor = referenceColor,
            left = left,
            bandTop = bandTop,
            cellWidth = cellWidth,
            cellHeight = cellHeight,
            preferredPoint = preferredPoint,
            expectedBox = expectedBox
        ) ?: return null
        if (component.size < MIN_REGION_MATCHES) return null
        if (requireCompact && !isCompactEnough(component, rows, cols)) return null

        var minCol = cols
        var minRow = rows
        var maxCol = -1
        var maxRow = -1
        val cells = mutableListOf<FloatArray>()
        component.forEach { (row, col) ->
            minCol = minOf(minCol, col)
            minRow = minOf(minRow, row)
            maxCol = maxOf(maxCol, col)
            maxRow = maxOf(maxRow, row)
            val cellLeft = left + col * cellWidth
            val cellTop = bandTop + row * cellHeight
            val cellRight = left + (col + 1) * cellWidth
            val cellBottom = bandTop + (row + 1) * cellHeight
            cells += floatArrayOf(cellLeft, cellTop, cellRight, cellBottom)
        }

        val box = floatArrayOf(
            (left + minCol * cellWidth).coerceAtLeast(left.toFloat()),
            (bandTop + minRow * cellHeight).coerceAtLeast(top.toFloat()),
            (left + (maxCol + 1) * cellWidth).coerceAtMost(right.toFloat()),
            (bandTop + (maxRow + 1) * cellHeight).coerceAtMost(bottom.toFloat())
        )
        return ColorRegionDetection(
            box = box,
            contour = buildComponentContour(component, left.toFloat(), bandTop.toFloat(), cellWidth, cellHeight),
            cells = cells
        )
    }

    private fun looksLikeWater(cellColor: Int, referenceColor: Int, excludedColor: Int?): Boolean {
        val hsv = FloatArray(3)
        Color.colorToHSV(cellColor, hsv)
        val blueish = hsv[0] in 150f..245f && hsv[1] >= 0.18f
        if (isDarkColor(referenceColor)) {
            return false
        }
        val muchCloserToBlue = excludedColor != null &&
            hsvDistance(cellColor, excludedColor) + 8 < hsvDistance(cellColor, referenceColor)
        return blueish && muchCloserToBlue
    }

    private fun colorMatchesReference(
        candidate: Int,
        reference: Int,
        hsvThreshold: Int,
        referencePalette: List<Int>? = null,
        template: ColorTrackingTemplate? = null
    ): Boolean {
        if (template != null && !matchesTemplateHueBand(candidate, template)) {
            return false
        }
        if (referencePalette != null && paletteMatchesReference(candidate, reference, hsvThreshold, referencePalette, template)) {
            return true
        }
        if (isDarkColor(reference)) {
            return darkColorDistance(candidate, reference) <= DARK_COLOR_DISTANCE_THRESHOLD
        }
        if (isLowSaturation(reference)) {
            return neutralColorDistance(candidate, reference) <= NEUTRAL_COLOR_DISTANCE_THRESHOLD
        }
        return hsvDistance(candidate, reference) <= hsvThreshold
    }

    private fun paletteMatchesReference(
        candidate: Int,
        reference: Int,
        hsvThreshold: Int,
        referencePalette: List<Int>,
        template: ColorTrackingTemplate? = null
    ): Boolean {
        if (referencePalette.isEmpty()) return false
        if (template != null && !matchesTemplateHueBand(candidate, template)) {
            return false
        }
        return referencePalette.any { color ->
            if (template != null && !matchesHueBand(color, template.swimwearHue, template.swimwearHueTolerance + 2f)) {
                false
            } else if (isDarkColor(color)) {
                darkColorDistance(candidate, color) <= DARK_COLOR_DISTANCE_THRESHOLD + 8
            } else if (isLowSaturation(color)) {
                neutralColorDistance(candidate, color) <= NEUTRAL_COLOR_DISTANCE_THRESHOLD + 8
            } else {
                hsvDistance(candidate, color) <= hsvThreshold + PALETTE_THRESHOLD_BONUS
            }
        } || colorMatchesReference(candidate, reference, hsvThreshold, null, null)
    }

    private fun matchesTemplateHueBand(candidate: Int, template: ColorTrackingTemplate): Boolean {
        if (isLowSaturation(candidate)) return false
        if (!matchesHueBand(candidate, template.swimwearHue, template.swimwearHueTolerance)) return false
        val saturation = colorSaturation(candidate)
        val value = colorValue(candidate)
        return saturation in (template.swimwearMinSaturation - TEMPLATE_SATURATION_MARGIN).coerceAtLeast(0f)..
            (template.swimwearMaxSaturation + TEMPLATE_SATURATION_MARGIN).coerceAtMost(1f) &&
            value in (template.swimwearMinValue - TEMPLATE_VALUE_MARGIN).coerceAtLeast(0f)..
            (template.swimwearMaxValue + TEMPLATE_VALUE_MARGIN).coerceAtMost(1f)
    }

    private fun backgroundDistance(candidate: Int, background: Int): Int {
        return if (isDarkColor(candidate) || isDarkColor(background)) {
            darkColorDistance(candidate, background)
        } else {
            hsvDistance(candidate, background)
        }
    }

    private fun regionTemplateDistance(a: Int, b: Int): Int {
        return if (isDarkColor(a) || isDarkColor(b)) {
            darkColorDistance(a, b)
        } else if (isLowSaturation(a) || isLowSaturation(b)) {
            neutralColorDistance(a, b)
        } else {
            hsvDistance(a, b)
        }
    }

    private fun darkColorDistance(a: Int, b: Int): Int {
        val lumaDiff = abs(luma(a) - luma(b))
        val redDiff = abs(Color.red(a) - Color.red(b))
        val greenDiff = abs(Color.green(a) - Color.green(b))
        val blueDiff = abs(Color.blue(a) - Color.blue(b))
        return lumaDiff + ((redDiff + greenDiff + blueDiff) / 6)
    }

    private fun neutralColorDistance(a: Int, b: Int): Int {
        val lumaDiff = abs(luma(a) - luma(b))
        val chromaDiff = abs(Color.red(a) - Color.red(b)) +
            abs(Color.green(a) - Color.green(b)) +
            abs(Color.blue(a) - Color.blue(b))
        return lumaDiff + chromaDiff / 4
    }

    private fun isDarkColor(color: Int): Boolean = luma(color) <= 70

    private fun isLowSaturation(color: Int): Boolean {
        val hsv = FloatArray(3)
        Color.colorToHSV(color, hsv)
        return hsv[1] <= 0.22f
    }

    private fun luma(color: Int): Int {
        return ((Color.red(color) * 299) + (Color.green(color) * 587) + (Color.blue(color) * 114)) / 1000
    }

    private fun isCompactEnough(component: List<Pair<Int, Int>>, rows: Int, cols: Int): Boolean {
        var minRow = rows
        var minCol = cols
        var maxRow = -1
        var maxCol = -1
        component.forEach { (row, col) ->
            minRow = minOf(minRow, row)
            minCol = minOf(minCol, col)
            maxRow = maxOf(maxRow, row)
            maxCol = maxOf(maxCol, col)
        }
        val spanRows = (maxRow - minRow + 1).coerceAtLeast(1)
        val spanCols = (maxCol - minCol + 1).coerceAtLeast(1)
        val density = component.size.toFloat() / (spanRows * spanCols).toFloat()
        return density >= MIN_COMPONENT_DENSITY
    }

    private fun buildComponentContour(
        component: List<Pair<Int, Int>>,
        offsetX: Float,
        offsetY: Float,
        cellWidth: Float,
        cellHeight: Float
    ): List<FloatArray> {
        var minRow = Int.MAX_VALUE
        var minCol = Int.MAX_VALUE
        var maxRow = Int.MIN_VALUE
        var maxCol = Int.MIN_VALUE
        component.forEach { (row, col) ->
            minRow = minOf(minRow, row)
            minCol = minOf(minCol, col)
            maxRow = maxOf(maxRow, row)
            maxCol = maxOf(maxCol, col)
        }
        return listOf(
            floatArrayOf(offsetX + minCol * cellWidth, offsetY + minRow * cellHeight),
            floatArrayOf(offsetX + (maxCol + 1) * cellWidth, offsetY + minRow * cellHeight),
            floatArrayOf(offsetX + (maxCol + 1) * cellWidth, offsetY + (maxRow + 1) * cellHeight),
            floatArrayOf(offsetX + minCol * cellWidth, offsetY + (maxRow + 1) * cellHeight)
        )
    }

    private fun buildLockedColorComboBox(primaryBox: FloatArray, supportBox: FloatArray?): FloatArray {
        val secondBox = supportBox ?: primaryBox
        val left = minOf(primaryBox[0], secondBox[0])
        val top = minOf(primaryBox[1], secondBox[1])
        val right = maxOf(primaryBox[2], secondBox[2])
        val bottom = maxOf(primaryBox[3], secondBox[3])
        return floatArrayOf(left, top, right, bottom)
    }

    private fun buildColorCorrectionProfile(samplingFrame: SamplingFrame, focusBox: FloatArray?): ColorCorrectionProfile {
        val sampleBox = focusBox ?: floatArrayOf(0f, 0f, samplingFrame.sourceWidth.toFloat(), samplingFrame.sourceHeight.toFloat())
        val left = mapSourceXToSample(samplingFrame, sampleBox[0]).coerceIn(0, samplingFrame.sampleWidth - 1)
        val top = mapSourceYToSample(samplingFrame, sampleBox[1]).coerceIn(0, samplingFrame.sampleHeight - 1)
        val right = mapSourceXToSample(samplingFrame, sampleBox[2]).coerceIn(left + 1, samplingFrame.sampleWidth)
        val bottom = mapSourceYToSample(samplingFrame, sampleBox[3]).coerceIn(top + 1, samplingFrame.sampleHeight)

        var red = 0L
        var green = 0L
        var blue = 0L
        var count = 0L
        val stepX = ((right - left) / 20).coerceAtLeast(1)
        val stepY = ((bottom - top) / 16).coerceAtLeast(1)

        for (y in top until bottom step stepY) {
            for (x in left until right step stepX) {
                val pixel = samplingFrame.pixels[y * samplingFrame.sampleWidth + x]
                red += Color.red(pixel)
                green += Color.green(pixel)
                blue += Color.blue(pixel)
                count++
            }
        }

        if (count == 0L) {
            return ColorCorrectionProfile(1f, 1f, 1f)
        }

        val redAvg = (red / count).toFloat().coerceAtLeast(1f)
        val greenAvg = (green / count).toFloat().coerceAtLeast(1f)
        val blueAvg = (blue / count).toFloat().coerceAtLeast(1f)
        val gray = (redAvg + greenAvg + blueAvg) / 3f
        return ColorCorrectionProfile(
            redGain = (gray / redAvg).coerceIn(0.75f, 1.85f),
            greenGain = (gray / greenAvg).coerceIn(0.75f, 1.6f),
            blueGain = (gray / blueAvg).coerceIn(0.70f, 1.45f)
        )
    }

    private fun correctPixel(pixel: Int, correction: ColorCorrectionProfile): Int {
        val red = (Color.red(pixel) * correction.redGain).roundToInt().coerceIn(0, 255)
        val green = (Color.green(pixel) * correction.greenGain).roundToInt().coerceIn(0, 255)
        val blue = (Color.blue(pixel) * correction.blueGain).roundToInt().coerceIn(0, 255)
        return Color.rgb(red, green, blue)
    }

    private fun findPreferredComponent(
        mask: Array<BooleanArray>,
        cellColors: Array<IntArray>,
        referenceColor: Int,
        left: Int,
        bandTop: Int,
        cellWidth: Float,
        cellHeight: Float,
        preferredPoint: FloatArray?,
        expectedBox: FloatArray?
    ): List<Pair<Int, Int>>? {
        val rows = mask.size
        val cols = mask.firstOrNull()?.size ?: return null
        val visited = Array(rows) { BooleanArray(cols) }
        val components = mutableListOf<ConnectedComponent>()
        val directions = arrayOf(
            intArrayOf(1, 0), intArrayOf(-1, 0),
            intArrayOf(0, 1), intArrayOf(0, -1)
        )

        for (row in 0 until rows) {
            for (col in 0 until cols) {
                if (!mask[row][col] || visited[row][col]) continue
                val queue = ArrayDeque<Pair<Int, Int>>()
                val component = mutableListOf<Pair<Int, Int>>()
                var minRow = row
                var minCol = col
                var maxRow = row
                var maxCol = col
                queue.add(Pair(row, col))
                visited[row][col] = true
                while (queue.isNotEmpty()) {
                    val (currentRow, currentCol) = queue.removeFirst()
                    component += Pair(currentRow, currentCol)
                    minRow = minOf(minRow, currentRow)
                    minCol = minOf(minCol, currentCol)
                    maxRow = maxOf(maxRow, currentRow)
                    maxCol = maxOf(maxCol, currentCol)
                    directions.forEach { direction ->
                        val nextRow = currentRow + direction[0]
                        val nextCol = currentCol + direction[1]
                        if (nextRow !in 0 until rows || nextCol !in 0 until cols) return@forEach
                        if (!mask[nextRow][nextCol] || visited[nextRow][nextCol]) return@forEach
                        if (!canCellsConnect(cellColors[currentRow][currentCol], cellColors[nextRow][nextCol], referenceColor)) {
                            return@forEach
                        }
                        visited[nextRow][nextCol] = true
                        queue.add(Pair(nextRow, nextCol))
                    }
                }
                components += ConnectedComponent(component, minRow, minCol, maxRow, maxCol)
            }
        }
        if (components.isEmpty()) return null

        return components.maxByOrNull { component ->
            componentScore(
                component = component,
                preferredPoint = preferredPoint,
                expectedBox = expectedBox,
                offsetX = left.toFloat(),
                offsetY = bandTop.toFloat(),
                cellWidth = cellWidth,
                cellHeight = cellHeight
            )
        }?.cells
    }

    private fun squaredDistanceToComponentCenter(
        component: ConnectedComponent,
        preferredPoint: FloatArray,
        offsetX: Float,
        offsetY: Float,
        cellWidth: Float,
        cellHeight: Float
    ): Float {
        val centerX = offsetX + ((component.minCol + component.maxCol + 1) * 0.5f) * cellWidth
        val centerY = offsetY + ((component.minRow + component.maxRow + 1) * 0.5f) * cellHeight
        val dx = centerX - preferredPoint[0]
        val dy = centerY - preferredPoint[1]
        return dx * dx + dy * dy
    }

    private fun componentScore(
        component: ConnectedComponent,
        preferredPoint: FloatArray?,
        expectedBox: FloatArray?,
        offsetX: Float,
        offsetY: Float,
        cellWidth: Float,
        cellHeight: Float
    ): Float {
        var score = component.cells.size * COMPONENT_SIZE_SCORE_WEIGHT

        if (preferredPoint != null) {
            val centerPenalty = squaredDistanceToComponentCenter(
                component,
                preferredPoint,
                offsetX,
                offsetY,
                cellWidth,
                cellHeight
            ) / COMPONENT_DISTANCE_NORMALIZER
            score -= centerPenalty
        }

        if (expectedBox != null) {
            val componentBox = floatArrayOf(
                offsetX + component.minCol * cellWidth,
                offsetY + component.minRow * cellHeight,
                offsetX + (component.maxCol + 1) * cellWidth,
                offsetY + (component.maxRow + 1) * cellHeight
            )
            val expectedArea = ((expectedBox[2] - expectedBox[0]) * (expectedBox[3] - expectedBox[1])).coerceAtLeast(1f)
            val componentArea = ((componentBox[2] - componentBox[0]) * (componentBox[3] - componentBox[1])).coerceAtLeast(1f)
            val areaRatio = max(componentArea / expectedArea, expectedArea / componentArea)
            score -= abs(areaRatio - 1f) * COMPONENT_AREA_PENALTY_WEIGHT
            score += intersectionOverUnion(componentBox, expectedBox) * COMPONENT_IOU_SCORE_WEIGHT
        }

        return score
    }

    private fun canCellsConnect(colorA: Int, colorB: Int, referenceColor: Int): Boolean {
        val colorGap = when {
            isDarkColor(referenceColor) -> darkColorDistance(colorA, colorB)
            isLowSaturation(referenceColor) -> neutralColorDistance(colorA, colorB)
            else -> hsvDistance(colorA, colorB)
        }
        val lumaGap = abs(luma(colorA) - luma(colorB))
        return colorGap <= CELL_EDGE_COLOR_THRESHOLD && lumaGap <= CELL_EDGE_LUMA_THRESHOLD
    }

    private fun intersectionOverUnion(a: FloatArray, b: FloatArray): Float {
        val left = maxOf(a[0], b[0])
        val top = maxOf(a[1], b[1])
        val right = minOf(a[2], b[2])
        val bottom = minOf(a[3], b[3])
        if (right <= left || bottom <= top) return 0f
        val intersection = (right - left) * (bottom - top)
        val union = ((a[2] - a[0]) * (a[3] - a[1])) + ((b[2] - b[0]) * (b[3] - b[1])) - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private fun FloatArray.centerPoint(): FloatArray {
        return floatArrayOf((this[0] + this[2]) * 0.5f, (this[1] + this[3]) * 0.5f)
    }

    private fun boxToContour(box: FloatArray): List<FloatArray> {
        return listOf(
            floatArrayOf(box[0], box[1]),
            floatArrayOf(box[2], box[1]),
            floatArrayOf(box[2], box[3]),
            floatArrayOf(box[0], box[3])
        )
    }

    private fun buildColorComboBox(
        trackingBox: FloatArray,
        skinBox: FloatArray,
        swimwearBox: FloatArray
    ): FloatArray {
        val left = minOf(skinBox[0], swimwearBox[0]) - (trackingBox[2] - trackingBox[0]) * 0.04f
        val top = minOf(skinBox[1], swimwearBox[1]) - (trackingBox[3] - trackingBox[1]) * 0.04f
        val right = maxOf(skinBox[2], swimwearBox[2]) + (trackingBox[2] - trackingBox[0]) * 0.04f
        val bottom = maxOf(skinBox[3], swimwearBox[3]) + (trackingBox[3] - trackingBox[1]) * 0.04f
        return floatArrayOf(
            left.coerceAtLeast(0f),
            top.coerceAtLeast(0f),
            right.coerceAtLeast(left + 1f),
            bottom.coerceAtLeast(top + 1f)
        )
    }

    companion object {
        private const val VISIBILITY_THRESHOLD = 0.3f
        private const val MEMORY_HOLD_MS = 900L
        private const val SMOOTHING_WEIGHT = 0.65f
        private const val COLOR_DISTANCE_THRESHOLD = 90
        private const val HSV_DISTANCE_THRESHOLD = 92
        private const val HSV_SEARCH_THRESHOLD = 150
        private const val HSV_REGION_THRESHOLD = 56
        private const val DIRECT_COLOR_THRESHOLD = 42
        private const val DIRECT_COLOR_GLOBAL_THRESHOLD = 48
        private const val LOCKED_SWIMWEAR_THRESHOLD = 68
        private const val LOCKED_SKIN_THRESHOLD = 74
        private const val BACKGROUND_EXCLUSION_THRESHOLD = 42
        private const val DARK_COLOR_DISTANCE_THRESHOLD = 54
        private const val NEUTRAL_COLOR_DISTANCE_THRESHOLD = 62
        private const val MOTION_PREDICTION_WEIGHT = 0.92f
        private const val MAX_POSITION_JUMP_RATIO = 2.35f
        private const val MIN_BOX_SCALE_RATIO = 0.42f
        private const val MAX_BOX_SCALE_RATIO = 2.10f
        private const val TRACK_BOX_SMOOTHING = 0.36f
        private const val MIN_COMPONENT_DENSITY = 0.32f
        private const val MIN_REGION_MATCHES = 4
        private const val MIN_TEMPLATE_SAMPLES = 8
        private const val GLOBAL_RECOVERY_INTERVAL_MS = 360L
        private const val DIRECT_COLOR_LOCAL_COLS = 18
        private const val DIRECT_COLOR_LOCAL_ROWS = 20
        private const val CELL_EDGE_COLOR_THRESHOLD = 46
        private const val CELL_EDGE_LUMA_THRESHOLD = 34
        private const val COMPONENT_DISTANCE_NORMALIZER = 4800f
        private const val COMPONENT_SIZE_SCORE_WEIGHT = 1.25f
        private const val COMPONENT_AREA_PENALTY_WEIGHT = 18f
        private const val COMPONENT_IOU_SCORE_WEIGHT = 24f
        private const val MAX_TEMPLATE_COLORS = 5
        private const val PALETTE_DEDUP_THRESHOLD = 22
        private const val PALETTE_THRESHOLD_BONUS = 14
        private const val DEFAULT_HUE_TOLERANCE = 16f
        private const val TEMPLATE_SATURATION_MARGIN = 0.10f
        private const val TEMPLATE_VALUE_MARGIN = 0.18f
        private const val SEGMENT_GRID_COLS = 24
        private const val SEGMENT_GRID_ROWS = 16
        private const val MIN_SEGMENT_COMPONENT_CELLS = 2
        private const val SEGMENT_MERGE_COLOR_THRESHOLD = 30
        private const val SEGMENT_MERGE_LUMA_THRESHOLD = 24
        private const val SEGMENT_WATER_EXCLUSION_THRESHOLD = 34
        private const val SEGMENT_BACKGROUND_AREA_RATIO = 0.035f
        private const val SEGMENT_AREA_SCORE_WEIGHT = 0.0022f
        private const val SEGMENT_COLOR_PENALTY_WEIGHT = 1.1f
        private const val SEGMENT_IOU_SCORE_WEIGHT = 36f
        private const val SEGMENT_AREA_RATIO_PENALTY = 22f
        private const val SEGMENT_DISTANCE_NORMALIZER = 14000f
        private const val FAST_MOTION_MAX_COLOR_DISTANCE = 34
        private const val FAST_MOTION_MAX_AREA_RATIO = 2.9f
        private const val ANALYSIS_MAX_SIDE = 192
        private const val GLOBAL_RECOVERY_COLS = 8
        private const val GLOBAL_RECOVERY_ROWS = 10
    }
}
