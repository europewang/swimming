package com.example.yolopose.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class YoloPoseDetector(
    context: Context,
    private val modelPath: String = MODEL_PATH
) {
    private val interpreter: Interpreter
    private val inputSize: Int
    private val outputTensorShape: IntArray

    init {
        val modelBuffer = loadModelFile(context, modelPath)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(modelBuffer, options)
        inputSize = interpreter.getInputTensor(0).shape()[1]
        outputTensorShape = interpreter.getOutputTensor(0).shape()
    }

    suspend fun estimatePose(bitmap: Bitmap): PoseResult? = withContext(Dispatchers.Default) {
        val scaled = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputBuffer = toInputBuffer(scaled)
        val output = Array(1) { Array(outputTensorShape[1]) { FloatArray(outputTensorShape[2]) } }
        interpreter.run(inputBuffer, output)
        decode(output, bitmap.width, bitmap.height)
    }

    fun close() {
        interpreter.close()
    }

    fun outputShape(): IntArray = outputTensorShape.copyOf()

    private fun decode(
        output: Array<Array<FloatArray>>,
        sourceWidth: Int,
        sourceHeight: Int
    ): PoseResult? {
        val rows = output[0].size
        val cols = output[0][0].size
        var best: PoseResult? = null

        if (rows == FEATURE_COUNT) {
            for (anchor in 0 until cols) {
                val candidate = decodeCandidate(
                    featureAt = { featureIndex: Int -> output[0][featureIndex][anchor] },
                    sourceWidth = sourceWidth,
                    sourceHeight = sourceHeight
                )
                if (candidate != null && (best == null || candidate.confidence > best.confidence)) {
                    best = candidate
                }
            }
        } else if (cols == FEATURE_COUNT) {
            for (anchor in 0 until rows) {
                val candidate = decodeCandidate(
                    featureAt = { featureIndex: Int -> output[0][anchor][featureIndex] },
                    sourceWidth = sourceWidth,
                    sourceHeight = sourceHeight
                )
                if (candidate != null && (best == null || candidate.confidence > best.confidence)) {
                    best = candidate
                }
            }
        } else {
            error("Unsupported output shape: " + outputTensorShape.joinToString(prefix = "[", postfix = "]"))
        }

        return best
    }

    private fun decodeCandidate(
        featureAt: (Int) -> Float,
        sourceWidth: Int,
        sourceHeight: Int
    ): PoseResult? {
        val confidence = featureAt(4)
        if (confidence < MIN_CONFIDENCE) {
            return null
        }

        val cx = featureAt(0)
        val cy = featureAt(1)
        val w = featureAt(2)
        val h = featureAt(3)

        val left = (cx - w / 2f) * sourceWidth / inputSize
        val top = (cy - h / 2f) * sourceHeight / inputSize
        val right = (cx + w / 2f) * sourceWidth / inputSize
        val bottom = (cy + h / 2f) * sourceHeight / inputSize

        val box = RectF(
            left.coerceIn(0f, sourceWidth.toFloat()),
            top.coerceIn(0f, sourceHeight.toFloat()),
            right.coerceIn(0f, sourceWidth.toFloat()),
            bottom.coerceIn(0f, sourceHeight.toFloat())
        )

        val keyPoints = ArrayList<KeyPoint>(KEYPOINT_COUNT)
        for (index in 0 until KEYPOINT_COUNT) {
            val base = 5 + index * 3
            keyPoints.add(
                KeyPoint(
                    x = normalizeCoordinate(featureAt(base), sourceWidth),
                    y = normalizeCoordinate(featureAt(base + 1), sourceHeight),
                    confidence = featureAt(base + 2)
                )
            )
        }

        return PoseResult(
            boundingBox = floatArrayOf(box.left, box.top, box.right, box.bottom),
            confidence = confidence,
            keyPoints = keyPoints
        )
    }

    private fun normalizeCoordinate(raw: Float, sourceDimension: Int): Float {
        val scaled = if (raw in 0f..1.5f) raw * sourceDimension else raw * sourceDimension / inputSize
        return scaled.coerceIn(0f, sourceDimension.toFloat())
    }

    private fun toInputBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
            .order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
            buffer.putFloat((pixel and 0xFF) / 255f)
        }
        buffer.rewind()
        return buffer
    }

    private fun loadModelFile(context: Context, path: String): ByteBuffer {
        val descriptor = context.assets.openFd(path)
        val inputStream = descriptor.createInputStream()
        return inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY,
            descriptor.startOffset,
            descriptor.declaredLength
        )
    }

    companion object {
        private const val KEYPOINT_COUNT = 17
        private const val FEATURE_COUNT = 56
        private const val MIN_CONFIDENCE = 0.35f
        const val MODEL_PATH = "models/yolo_pose.tflite"

        fun modelExists(context: Context, path: String = MODEL_PATH): Boolean {
            return try {
                context.assets.open(path).close()
                true
            } catch (_: FileNotFoundException) {
                false
            }
        }
    }
}
