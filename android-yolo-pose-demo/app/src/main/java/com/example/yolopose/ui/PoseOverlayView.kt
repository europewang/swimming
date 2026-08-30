package com.example.yolopose.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Path
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.example.yolopose.ml.DetectionMode
import com.example.yolopose.ml.KeyPoint
import com.example.yolopose.ml.PoseResult

class PoseOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#51E3A1")
        style = Paint.Style.STROKE
        strokeWidth = 12f
    }

    private val boxFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#2851E3A1")
        style = Paint.Style.FILL
    }

    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#37B2FF")
        style = Paint.Style.FILL
    }

    private val headPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFD166")
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val torsoFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#3351E3A1")
        style = Paint.Style.FILL
    }

    private val skeletonPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 8f
        alpha = 210
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private val skeletonShadowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#66000000")
        style = Paint.Style.STROKE
        strokeWidth = 14f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private val legLeftPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF6B6B")
        style = Paint.Style.STROKE
        strokeWidth = 12f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private val legRightPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#4ECDC4")
        style = Paint.Style.STROKE
        strokeWidth = 12f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private val footPointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFE66D")
        style = Paint.Style.FILL
    }

    private val legBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#99FFFFFF")
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val legLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
        isFakeBoldText = true
    }

    private val trackColorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.WHITE
    }

    private val anchorColorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.WHITE
    }

    private val anchorStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = Color.WHITE
    }

    private val skinBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.parseColor("#FFD166")
    }

    private val skinCellPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#44FFD166")
    }

    private val swimwearBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.parseColor("#FF6B6B")
    }

    private val swimwearCellPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#44FF6B6B")
    }

    private val comboBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 10f
        color = Color.parseColor("#FF4DDE")
    }

    private val comboFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#33FF4DDE")
    }

    private val boxLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 28f
        isFakeBoldText = true
    }

    private val memoryPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFB703")
        textSize = 34f
        isFakeBoldText = true
    }

    private val selectionPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00E5FF")
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    private var poseResult: PoseResult? = null
    private var sourceWidth = 1
    private var sourceHeight = 1
    private var detectionMode: DetectionMode = DetectionMode.LOST
    private var averageColor: Int? = null
    private var anchorColor: Int? = null
    private var trackingBox: FloatArray? = null
    private var skinBox: FloatArray? = null
    private var swimwearBox: FloatArray? = null
    private var skinContour: List<FloatArray>? = null
    private var swimwearContour: List<FloatArray>? = null
    private var skinCells: List<FloatArray>? = null
    private var swimwearCells: List<FloatArray>? = null
    private var colorComboBox: FloatArray? = null
    private var selectionPoint: FloatArray? = null

    fun updatePose(
        result: PoseResult?,
        sourceWidth: Int,
        sourceHeight: Int,
        mode: DetectionMode = DetectionMode.FULL_BODY,
        averageColor: Int? = null,
        anchorColor: Int? = null,
        trackingBox: FloatArray? = null,
        skinBox: FloatArray? = null,
        swimwearBox: FloatArray? = null,
        skinContour: List<FloatArray>? = null,
        swimwearContour: List<FloatArray>? = null,
        skinCells: List<FloatArray>? = null,
        swimwearCells: List<FloatArray>? = null,
        colorComboBox: FloatArray? = null
    ) {
        poseResult = result
        this.sourceWidth = sourceWidth
        this.sourceHeight = sourceHeight
        detectionMode = mode
        this.averageColor = averageColor
        this.anchorColor = anchorColor
        this.trackingBox = trackingBox
        this.skinBox = skinBox
        this.swimwearBox = swimwearBox
        this.skinContour = skinContour
        this.swimwearContour = swimwearContour
        this.skinCells = skinCells
        this.swimwearCells = swimwearCells
        this.colorComboBox = colorComboBox
        postInvalidateOnAnimation()
    }

    fun updateSelectionPoint(x: Float?, y: Float?) {
        selectionPoint = if (x != null && y != null) floatArrayOf(x, y) else null
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val scale = minOf(width / sourceWidth.toFloat(), height / sourceHeight.toFloat())
        val drawnWidth = sourceWidth * scale
        val drawnHeight = sourceHeight * scale
        val offsetX = (width - drawnWidth) / 2f
        val offsetY = (height - drawnHeight) / 2f
        val boxSource = trackingBox ?: poseResult?.boundingBox

        averageColor?.let {
            trackColorPaint.color = it
            canvas.drawCircle(36f, 36f, 18f, trackColorPaint)
        }
        anchorColor?.let {
            anchorColorPaint.color = it
            canvas.drawRect(60f, 18f, 96f, 54f, anchorColorPaint)
            canvas.drawRect(60f, 18f, 96f, 54f, anchorStrokePaint)
        }

        if (boxSource == null) {
            drawSelectionHint(canvas, scale, offsetX, offsetY)
            return
        }

        when (detectionMode) {
            DetectionMode.INITIALIZING,
            DetectionMode.FULL_BODY -> {
                drawBodyBox(canvas, boxSource, scale, offsetX, offsetY, solid = detectionMode != DetectionMode.TRACKED_MEMORY)
                drawAnchorRegion(canvas, boxSource, scale, offsetX, offsetY)
                drawColorBoxes(canvas, scale, offsetX, offsetY)
                poseResult?.let { result ->
                    drawLegHighlights(canvas, result, scale, offsetX, offsetY, faint = detectionMode != DetectionMode.FULL_BODY)
                    drawVisibleKeyPoints(canvas, result, scale, offsetX, offsetY, setOf(11, 12, 13, 14, 15, 16), faint = detectionMode != DetectionMode.FULL_BODY)
                }
            }
            DetectionMode.COLOR_TRACK,
            DetectionMode.TRACKED_MEMORY -> {
                drawColorTrackingFocus(canvas, scale, offsetX, offsetY)
                drawColorBoxes(canvas, scale, offsetX, offsetY)
            }
            DetectionMode.LEG_ONLY -> {
                drawBodyBox(canvas, boxSource, scale, offsetX, offsetY, solid = true)
                drawAnchorRegion(canvas, boxSource, scale, offsetX, offsetY)
                drawColorBoxes(canvas, scale, offsetX, offsetY)
                poseResult?.let { result ->
                    drawLegHighlights(canvas, result, scale, offsetX, offsetY, faint = false)
                    drawVisibleKeyPoints(canvas, result, scale, offsetX, offsetY, setOf(11, 12, 13, 14, 15, 16), faint = false)
                }
            }
            DetectionMode.FOOT_ONLY -> {
                drawBodyBox(canvas, boxSource, scale, offsetX, offsetY, solid = true)
                drawAnchorRegion(canvas, boxSource, scale, offsetX, offsetY)
                drawColorBoxes(canvas, scale, offsetX, offsetY)
                poseResult?.let { result -> drawFeetOnly(canvas, result, scale, offsetX, offsetY) }
            }
            DetectionMode.LOST -> Unit
        }

        if (detectionMode == DetectionMode.TRACKED_MEMORY) {
            canvas.drawText("跟踪延续", 24f, height - 24f, memoryPaint)
        }

        drawSelectionHint(canvas, scale, offsetX, offsetY)
    }

    private fun drawSelectionHint(canvas: Canvas, scale: Float, offsetX: Float, offsetY: Float) {
        selectionPoint?.let { point ->
            val x = point[0] * scale + offsetX
            val y = point[1] * scale + offsetY
            canvas.drawCircle(x, y, 24f, selectionPaint)
            canvas.drawLine(x - 32f, y, x + 32f, y, selectionPaint)
            canvas.drawLine(x, y - 32f, x, y + 32f, selectionPaint)
            canvas.drawText("已选目标", x + 20f, y - 20f, boxLabelPaint)
        }
    }

    private fun drawColorTrackingFocus(canvas: Canvas, scale: Float, offsetX: Float, offsetY: Float) {
        val selectedColor = anchorColor ?: averageColor
        selectedColor?.let {
            comboBoxPaint.color = it
            comboFillPaint.color = withAlpha(it, 70)
            swimwearBoxPaint.color = it
            swimwearCellPaint.color = withAlpha(it, 90)
        }
        val regionPath = buildCombinedRegionPath(scale, offsetX, offsetY)
        if (regionPath != null) {
            canvas.drawPath(regionPath, comboFillPaint)
            canvas.drawPath(regionPath, comboBoxPaint)
            val anchor = firstCombinedPoint(scale, offsetX, offsetY)
            if (anchor != null) {
                canvas.drawText("颜色主跟踪区域", anchor.first, (anchor.second - 14f).coerceAtLeast(36f), boxLabelPaint)
            }
            return
        }
        val combo = colorComboBox ?: trackingBox ?: return
        val rect = toRect(combo, scale, offsetX, offsetY)
        canvas.drawRoundRect(rect, 18f, 18f, comboFillPaint)
        canvas.drawRoundRect(rect, 18f, 18f, comboBoxPaint)
        canvas.drawText("颜色主跟踪区域", rect.left, (rect.top - 14f).coerceAtLeast(36f), boxLabelPaint)
    }

    private fun drawBodyBox(
        canvas: Canvas,
        boxSource: FloatArray,
        scale: Float,
        offsetX: Float,
        offsetY: Float,
        solid: Boolean
    ) {
        boxPaint.alpha = if (solid) 255 else 150
        val box = RectF(
            boxSource[0] * scale + offsetX,
            boxSource[1] * scale + offsetY,
            boxSource[2] * scale + offsetX,
            boxSource[3] * scale + offsetY
        )
        boxFillPaint.alpha = if (solid) 76 else 44
        canvas.drawRoundRect(box, 18f, 18f, boxFillPaint)
        canvas.drawRoundRect(box, 18f, 18f, boxPaint)
        canvas.drawText(labelForMode(), box.left, (box.top - 14f).coerceAtLeast(36f), boxLabelPaint)
        boxPaint.alpha = 255
    }

    private fun labelForMode(): String {
        return when (detectionMode) {
            DetectionMode.INITIALIZING -> "初始化框"
            DetectionMode.COLOR_TRACK -> "色块跟踪框"
            DetectionMode.FULL_BODY -> "人物框"
            DetectionMode.LEG_ONLY -> "腿部辅助框"
            DetectionMode.FOOT_ONLY -> "脚部辅助框"
            DetectionMode.TRACKED_MEMORY -> "延续跟踪框"
            DetectionMode.LOST -> "目标丢失"
        }
    }

    private fun drawAnchorRegion(canvas: Canvas, boxSource: FloatArray, scale: Float, offsetX: Float, offsetY: Float) {
        val left = boxSource[0] * scale + offsetX
        val top = boxSource[1] * scale + offsetY
        val right = boxSource[2] * scale + offsetX
        val bottom = boxSource[3] * scale + offsetY
        val anchorTop = top + (bottom - top) * 0.25f
        val anchorBottom = top + (bottom - top) * 0.75f
        val rect = RectF(left, anchorTop, right, anchorBottom)
        canvas.drawRoundRect(rect, 14f, 14f, anchorStrokePaint)
    }

    private fun drawColorBoxes(canvas: Canvas, scale: Float, offsetX: Float, offsetY: Float) {
        val selectedColor = anchorColor ?: averageColor
        selectedColor?.let {
            swimwearBoxPaint.color = it
            swimwearCellPaint.color = withAlpha(it, 90)
        }
        drawCellRegion(canvas, skinCells, skinContour, skinBox, scale, offsetX, offsetY, skinCellPaint, skinBoxPaint, "肤色区域")
        drawCellRegion(canvas, swimwearCells, swimwearContour, swimwearBox, scale, offsetX, offsetY, swimwearCellPaint, swimwearBoxPaint, "泳衣区域")
    }

    private fun drawCellRegion(
        canvas: Canvas,
        cells: List<FloatArray>?,
        contour: List<FloatArray>?,
        fallbackBox: FloatArray?,
        scale: Float,
        offsetX: Float,
        offsetY: Float,
        fillPaint: Paint,
        strokePaint: Paint,
        label: String
    ) {
        if (!cells.isNullOrEmpty()) {
            cells.forEach { cell ->
                canvas.drawRect(toRect(cell, scale, offsetX, offsetY), fillPaint)
            }
            val path = when {
                !contour.isNullOrEmpty() -> buildContourPath(contour, scale, offsetX, offsetY)
                else -> buildPathFromCells(cells, scale, offsetX, offsetY)
            }
            canvas.drawPath(path, strokePaint)
            val first = contour?.firstOrNull() ?: cells.first()
            canvas.drawText(label, first[0] * scale + offsetX, (first[1] * scale + offsetY - 8f).coerceAtLeast(36f), boxLabelPaint)
            return
        }
        fallbackBox?.let { box ->
            val rect = toRect(box, scale, offsetX, offsetY)
            canvas.drawRoundRect(rect, 12f, 12f, fillPaint)
            canvas.drawRoundRect(rect, 12f, 12f, strokePaint)
            canvas.drawText(label, rect.left, rect.top - 8f, boxLabelPaint)
        }
    }

    private fun buildCombinedRegionPath(scale: Float, offsetX: Float, offsetY: Float): Path? {
        val contourPath = buildCombinedContourPath(scale, offsetX, offsetY)
        if (contourPath != null) return contourPath
        val allCells = buildList {
            skinCells?.let { addAll(it) }
            swimwearCells?.let { addAll(it) }
        }
        if (allCells.isEmpty()) return null
        return buildPathFromCells(allCells, scale, offsetX, offsetY)
    }

    private fun buildCombinedContourPath(scale: Float, offsetX: Float, offsetY: Float): Path? {
        val contours = buildList {
            skinContour?.takeIf { it.isNotEmpty() }?.let { add(it) }
            swimwearContour?.takeIf { it.isNotEmpty() }?.let { add(it) }
        }
        if (contours.isEmpty()) return null
        val path = Path()
        contours.forEach { contour ->
            path.addPath(buildContourPath(contour, scale, offsetX, offsetY))
        }
        return path
    }

    private fun firstCombinedPoint(scale: Float, offsetX: Float, offsetY: Float): Pair<Float, Float>? {
        val cell = skinCells?.firstOrNull() ?: swimwearCells?.firstOrNull() ?: return null
        return Pair(cell[0] * scale + offsetX, cell[1] * scale + offsetY)
    }

    private fun buildPathFromCells(
        cells: List<FloatArray>,
        scale: Float,
        offsetX: Float,
        offsetY: Float
    ): Path {
        val path = Path()
        cells.forEach { cell ->
            val rect = toRect(cell, scale, offsetX, offsetY)
            path.addRoundRect(rect, 6f, 6f, Path.Direction.CW)
        }
        return path
    }

    private fun buildContourPath(
        contour: List<FloatArray>,
        scale: Float,
        offsetX: Float,
        offsetY: Float
    ): Path {
        val path = Path()
        contour.forEachIndexed { index, point ->
            val x = point[0] * scale + offsetX
            val y = point[1] * scale + offsetY
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        path.close()
        return path
    }

    private fun toRect(box: FloatArray, scale: Float, offsetX: Float, offsetY: Float): RectF {
        return RectF(
            box[0] * scale + offsetX,
            box[1] * scale + offsetY,
            box[2] * scale + offsetX,
            box[3] * scale + offsetY
        )
    }

    private fun withAlpha(color: Int, alpha: Int): Int {
        return Color.argb(
            alpha.coerceIn(0, 255),
            Color.red(color),
            Color.green(color),
            Color.blue(color)
        )
    }

    private fun drawVisibleKeyPoints(
        canvas: Canvas,
        result: PoseResult,
        scale: Float,
        offsetX: Float,
        offsetY: Float,
        onlyIndices: Set<Int>? = null,
        faint: Boolean = false
    ) {
        pointPaint.alpha = if (faint) 110 else 255
        result.keyPoints.forEachIndexed { index, point ->
            if ((onlyIndices == null || index in onlyIndices) && point.confidence > 0.3f) {
                canvas.drawCircle(point.x * scale + offsetX, point.y * scale + offsetY, 10f, pointPaint)
            }
        }
        pointPaint.alpha = 255
    }

    private fun drawLegHighlights(
        canvas: Canvas,
        result: PoseResult,
        scale: Float,
        offsetX: Float,
        offsetY: Float,
        faint: Boolean
    ) {
        legLeftPaint.alpha = if (faint) 120 else 255
        legRightPaint.alpha = if (faint) 120 else 255
        legBoxPaint.alpha = if (faint) 90 else 255
        drawSingleLeg(canvas, result.keyPoints.getOrNull(11), result.keyPoints.getOrNull(13), result.keyPoints.getOrNull(15), "左腿", legLeftPaint, scale, offsetX, offsetY)
        drawSingleLeg(canvas, result.keyPoints.getOrNull(12), result.keyPoints.getOrNull(14), result.keyPoints.getOrNull(16), "右腿", legRightPaint, scale, offsetX, offsetY)
        legLeftPaint.alpha = 255
        legRightPaint.alpha = 255
        legBoxPaint.alpha = 255
    }

    private fun drawFeetOnly(canvas: Canvas, result: PoseResult, scale: Float, offsetX: Float, offsetY: Float) {
        result.keyPoints.forEachIndexed { index, point ->
            if (index in setOf(15, 16) && point.confidence > 0.3f) {
                val x = point.x * scale + offsetX
                val y = point.y * scale + offsetY
                canvas.drawCircle(x, y, 18f, footPointPaint)
                canvas.drawText(if (index == 15) "左脚" else "右脚", x + 12f, y - 12f, legLabelPaint)
            }
        }
    }

    private fun drawSingleLeg(
        canvas: Canvas,
        hip: KeyPoint?,
        knee: KeyPoint?,
        ankle: KeyPoint?,
        label: String,
        paint: Paint,
        scale: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        if (!isVisible(hip) || !isVisible(knee) || !isVisible(ankle)) return
        val hipX = hip!!.x * scale + offsetX
        val hipY = hip.y * scale + offsetY
        val kneeX = knee!!.x * scale + offsetX
        val kneeY = knee.y * scale + offsetY
        val ankleX = ankle!!.x * scale + offsetX
        val ankleY = ankle.y * scale + offsetY
        val minX = minOf(hipX, kneeX, ankleX) - 20f
        val maxX = maxOf(hipX, kneeX, ankleX) + 20f
        val minY = minOf(hipY, kneeY, ankleY) - 20f
        val maxY = maxOf(hipY, kneeY, ankleY) + 20f
        canvas.drawRoundRect(RectF(minX, minY, maxX, maxY), 18f, 18f, legBoxPaint)
        canvas.drawLine(hipX, hipY, kneeX, kneeY, paint)
        canvas.drawLine(kneeX, kneeY, ankleX, ankleY, paint)
        canvas.drawCircle(ankleX, ankleY, 16f, footPointPaint)
        canvas.drawText(label, kneeX + 12f, kneeY - 12f, legLabelPaint)
    }

    private fun isVisible(point: KeyPoint?): Boolean = point != null && point.confidence > 0.3f
}
