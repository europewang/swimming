package com.example.swimming.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.swimming.model.PersonDetection
import java.nio.FloatBuffer
import kotlin.math.roundToInt
import kotlin.math.max
import kotlin.math.min

class PersonDetector(private val context: Context) {

    companion object {
        private const val MODEL_PATH = "yolo11n.onnx"
        private const val INPUT_SIZE = 640
        private const val PERSON_CLASS_INDEX = 0
        private const val CONFIDENCE_THRESHOLD = 0.22f
        private const val IOU_THRESHOLD = 0.45f
        private const val MAX_RESULTS = 10
    }

    private var environment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputName: String? = null
    private var lastDebugInfo: DetectionDebugInfo = DetectionDebugInfo(message = "detector not run")

    private data class LetterboxResult(
        val bitmap: Bitmap,
        val scale: Float,
        val padX: Float,
        val padY: Float
    )

    init {
        setupSession()
    }

    private fun setupSession() {
        try {
            environment = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open(MODEL_PATH).use { it.readBytes() }
            val options = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            session = environment?.createSession(modelBytes, options)
            inputName = session?.inputNames?.firstOrNull()
            lastDebugInfo = DetectionDebugInfo(message = "onnx model ready")
        } catch (t: Throwable) {
            t.printStackTrace()
            session = null
            inputName = null
            lastDebugInfo = DetectionDebugInfo(message = "onnx init failed: ${t.javaClass.simpleName}")
        }
    }

    fun detect(bitmap: Bitmap): PersonDetection? {
        val ortSession = session ?: return null
        val ortEnvironment = environment ?: return null
        val modelInputName = inputName ?: return null

        val letterboxed = letterbox(bitmap)
        val input = preprocess(letterboxed.bitmap)
        val inputTensor = OnnxTensor.createTensor(
            ortEnvironment,
            FloatBuffer.wrap(input),
            longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )

        inputTensor.use { tensor ->
            ortSession.run(mapOf(modelInputName to tensor)).use { outputs ->
                val output = outputs.firstOrNull()?.value ?: run {
                    lastDebugInfo = DetectionDebugInfo(message = "onnx output empty")
                    return null
                }
                val candidates = parseDetections(output, bitmap.width, bitmap.height, letterboxed)
                val selected = nonMaxSuppression(candidates)
                val best = selected.maxByOrNull { it.confidence }
                lastDebugInfo = DetectionDebugInfo(
                    candidateCount = candidates.size,
                    selectedCount = selected.size,
                    bestConfidence = best?.confidence ?: 0f,
                    message = if (best != null) {
                        "best detection selected scale=${"%.3f".format(letterboxed.scale)}"
                    } else {
                        "no box after nms scale=${"%.3f".format(letterboxed.scale)}"
                    }
                )
                return best
            }
        }
    }

    private fun letterbox(bitmap: Bitmap): LetterboxResult {
        val originalWidth = bitmap.width.toFloat()
        val originalHeight = bitmap.height.toFloat()
        val scale = min(INPUT_SIZE / originalWidth, INPUT_SIZE / originalHeight)
        val resizedWidth = (originalWidth * scale).roundToInt().coerceAtLeast(1)
        val resizedHeight = (originalHeight * scale).roundToInt().coerceAtLeast(1)
        val padX = (INPUT_SIZE - resizedWidth) / 2f
        val padY = (INPUT_SIZE - resizedHeight) / 2f

        val resized = Bitmap.createScaledBitmap(bitmap, resizedWidth, resizedHeight, true)
        val output = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        Canvas(output).apply {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(resized, padX, padY, null)
        }

        return LetterboxResult(
            bitmap = output,
            scale = scale,
            padX = padX,
            padY = padY
        )
    }

    private fun preprocess(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        val plane = INPUT_SIZE * INPUT_SIZE
        val result = FloatArray(3 * plane)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            result[i] = ((pixel shr 16) and 0xFF) / 255f
            result[plane + i] = ((pixel shr 8) and 0xFF) / 255f
            result[plane * 2 + i] = (pixel and 0xFF) / 255f
        }

        return result
    }

    private fun parseDetections(
        output: Any,
        originalWidth: Int,
        originalHeight: Int,
        letterbox: LetterboxResult
    ): List<PersonDetection> {
        val arr3d = output as? Array<*> ?: return emptyList()
        val channels = arr3d.firstOrNull() as? Array<*> ?: return emptyList()
        val rawChannels = channels.mapNotNull { it as? FloatArray }
        if (rawChannels.size < 5) {
            lastDebugInfo = DetectionDebugInfo(message = "unexpected output channels=${rawChannels.size}")
            return emptyList()
        }

        val personChannel = 4 + PERSON_CLASS_INDEX
        if (personChannel >= rawChannels.size) {
            lastDebugInfo = DetectionDebugInfo(message = "person channel missing")
            return emptyList()
        }

        val boxCount = rawChannels[0].size
        val detections = ArrayList<PersonDetection>(min(boxCount, MAX_RESULTS * 8))
        for (i in 0 until boxCount) {
            val confidence = rawChannels[personChannel][i]
            if (confidence < CONFIDENCE_THRESHOLD) continue
            val detection = createDetection(
                cx = rawChannels[0][i],
                cy = rawChannels[1][i],
                w = rawChannels[2][i],
                h = rawChannels[3][i],
                confidence = confidence,
                originalWidth = originalWidth,
                originalHeight = originalHeight,
                letterbox = letterbox
            ) ?: continue
            detections.add(detection)
        }

        if (detections.isEmpty()) {
            lastDebugInfo = DetectionDebugInfo(
                candidateCount = 0,
                selectedCount = 0,
                bestConfidence = 0f,
                message = "no candidate over threshold"
            )
        }

        return detections.sortedByDescending { it.confidence }.take(MAX_RESULTS * 3)
    }

    private fun createDetection(
        cx: Float,
        cy: Float,
        w: Float,
        h: Float,
        confidence: Float,
        originalWidth: Int,
        originalHeight: Int,
        letterbox: LetterboxResult
    ): PersonDetection? {
        val leftModel = (cx - w / 2f) * INPUT_SIZE
        val topModel = (cy - h / 2f) * INPUT_SIZE
        val rightModel = (cx + w / 2f) * INPUT_SIZE
        val bottomModel = (cy + h / 2f) * INPUT_SIZE

        val leftOriginal = ((leftModel - letterbox.padX) / letterbox.scale)
        val topOriginal = ((topModel - letterbox.padY) / letterbox.scale)
        val rightOriginal = ((rightModel - letterbox.padX) / letterbox.scale)
        val bottomOriginal = ((bottomModel - letterbox.padY) / letterbox.scale)

        val left = (leftOriginal / originalWidth).coerceIn(0f, 1f)
        val top = (topOriginal / originalHeight).coerceIn(0f, 1f)
        val right = (rightOriginal / originalWidth).coerceIn(0f, 1f)
        val bottom = (bottomOriginal / originalHeight).coerceIn(0f, 1f)

        val width = (right - left).coerceIn(0f, 1f)
        val height = (bottom - top).coerceIn(0f, 1f)

        if (width < 0.05f || height < 0.05f) return null

        return PersonDetection(
            centerX = (left + right) / 2f,
            centerY = (top + bottom) / 2f,
            width = right - left,
            height = bottom - top,
            confidence = confidence.coerceIn(0f, 1f),
            left = left,
            top = top,
            right = right,
            bottom = bottom
        )
    }

    private fun nonMaxSuppression(detections: List<PersonDetection>): List<PersonDetection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val selected = mutableListOf<PersonDetection>()

        while (sorted.isNotEmpty() && selected.size < MAX_RESULTS) {
            val current = sorted.removeAt(0)
            selected.add(current)
            sorted.removeAll { candidate -> iou(current, candidate) > IOU_THRESHOLD }
        }

        return selected
    }

    private fun iou(a: PersonDetection, b: PersonDetection): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        if (interRight <= interLeft || interBottom <= interTop) return 0f

        val intersection = (interRight - interLeft) * (interBottom - interTop)
        val union = a.width * a.height + b.width * b.height - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    fun close() {
        session?.close()
        session = null
        environment = null
    }

    fun isReady(): Boolean = session != null && inputName != null

    fun getLastDebugInfo(): DetectionDebugInfo = lastDebugInfo
}
