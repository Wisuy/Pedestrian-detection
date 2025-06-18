package com.example.onnxmobileruntime

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.util.Collections

class YoloDetector(private val context: Context, private val modelPath: String) {

    @Volatile
    private var isClosed = false
    private val ortEnv = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null

    companion object {
        const val INPUT_SIZE = 640
        const val NMS_THRESHOLD = 0.45f
        const val IOU_THRESHOLD = 0.5f
        const val CONFIDENCE_THRESHOLD = 0.5f
        const val NUM_CLASSES = 1
        const val MODEL_OUTPUT_SIZE = 5 // x,y,w,h,conf
        private const val TAG = "YoloDetector"
    }

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            val modelBytes = context.assets.open(modelPath).readBytes()

            val sessionOptions = OrtSession.SessionOptions()
            // Use all available CPU cores
            val numCores = Runtime.getRuntime().availableProcessors()
            sessionOptions.setIntraOpNumThreads(numCores)

            ortSession = ortEnv.createSession(modelBytes, sessionOptions)
            Log.i(TAG, "ONNX model loaded successfully with $numCores threads: $modelPath")
            Log.i(TAG, "Input names: ${ortSession?.inputNames}")
            Log.i(TAG, "Output names: ${ortSession?.outputNames}")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading ONNX model from assets: ${e.message}", e)
            throw RuntimeException("Failed to load ONNX model: ${e.message}")
        }
    }

    fun detect(bitmap: Bitmap, callback: (List<Detection>) -> Unit) {
        if (isClosed || ortSession == null) {
            Log.w(TAG, "Tried to detect using a closed session.")
            callback(emptyList())
            return
        }

        val preprocessedBitmap = resizeAndNormalizeBitmap(bitmap)
        val floatBuffer = convertBitmapToFloatBuffer(preprocessedBitmap)

        val inputName = ortSession!!.inputNames.iterator().next()
        val inputShape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, inputShape)

        val results = mutableListOf<Detection>()

        try {
            val session = ortSession!!
            val runOptions = OrtSession.RunOptions()

            // Start Inference Time
            val inferenceStartTime = System.currentTimeMillis()
            val outputs = session.run(Collections.singletonMap(inputName, inputTensor), runOptions)
            val inferenceEndTime = System.currentTimeMillis()
            val inferenceTime = inferenceEndTime - inferenceStartTime
            Log.d(TAG, "Inference Time: ${inferenceTime}ms")
            // End Inference Time

            outputs.use {
                @Suppress("UNCHECKED_CAST")
                val outputTensor = it[0].value as Array<Array<FloatArray>> // shape: [1, 5, 8400]
                val batch = outputTensor.size // should be 1
                val channels = outputTensor[0].size // should be 5
                val numBoxes = outputTensor[0][0].size // 8400

                // Start Post-processing Time
                val postProcessingStartTime = System.currentTimeMillis()

                // Iterate over all boxes
                for (i in 0 until numBoxes) {
                    val x_center = outputTensor[0][0][i]
                    val y_center = outputTensor[0][1][i]
                    val width = outputTensor[0][2][i]
                    val height = outputTensor[0][3][i]
                    val confidence = outputTensor[0][4][i]

                    if (confidence > CONFIDENCE_THRESHOLD) {
                        val x1 = x_center - width / 2
                        val y1 = y_center - height / 2
                        val x2 = x_center + width / 2
                        val y2 = y_center + height / 2

                        results.add(
                            Detection(
                                x1, y1, x2, y2, confidence, 0
                            )
                        )
                    }
                }

                // Apply Non-Maximum Suppression (NMS)
                val finalDetections = nonMaxSuppression(results)
                val postProcessingEndTime = System.currentTimeMillis()
                val postProcessingTime = postProcessingEndTime - postProcessingStartTime
                Log.d(TAG, "Post-processing Time: ${postProcessingTime}ms")
                // End Post-processing Time

                callback(finalDetections)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during ONNX inference: ${e.message}", e)
            callback(emptyList())
        } finally {
            inputTensor.close()
        }
    }

    private fun resizeAndNormalizeBitmap(bitmap: Bitmap): Bitmap {
        val matrix = Matrix()
        val scaleX = INPUT_SIZE.toFloat() / bitmap.width
        val scaleY = INPUT_SIZE.toFloat() / bitmap.height
        matrix.postScale(scaleX, scaleY)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun convertBitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val imgData = ByteBuffer.allocateDirect(1 * 3 * INPUT_SIZE * INPUT_SIZE * 4)
        imgData.order(java.nio.ByteOrder.nativeOrder())
        val floatBuffer = imgData.asFloatBuffer()

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (channel in 0 until 3) {
            for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
                val pixel = pixels[i]
                val value = when (channel) {
                    0 -> (pixel shr 16 and 0xFF) / 255.0f // R
                    1 -> (pixel shr 8 and 0xFF) / 255.0f  // G
                    else -> (pixel and 0xFF) / 255.0f     // B
                }
                floatBuffer.put(value)
            }
        }

        floatBuffer.rewind()
        return floatBuffer
    }

    private fun nonMaxSuppression(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val sortedDetections = detections.sortedByDescending { it.confidence }
        val finalDetections = mutableListOf<Detection>()
        val suppressed = BooleanArray(sortedDetections.size) { false }

        for (i in sortedDetections.indices) {
            if (suppressed[i]) continue

            val currentDetection = sortedDetections[i]
            finalDetections.add(currentDetection)

            for (j in (i + 1) until sortedDetections.size) {
                if (suppressed[j]) continue

                val otherDetection = sortedDetections[j]
                val iou = calculateIoU(currentDetection.boundingBox, otherDetection.boundingBox)

                if (iou > IOU_THRESHOLD) {
                    suppressed[j] = true
                }
            }
        }
        return finalDetections
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val xA = maxOf(box1.x1, box2.x1)
        val yA = maxOf(box1.y1, box2.y1)
        val xB = minOf(box1.x2, box2.x2)
        val yB = minOf(box1.y2, box2.y2)

        val intersectionArea = maxOf(0f, xB - xA) * maxOf(0f, yB - yA)
        val box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        val box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    fun close() {
        isClosed = true
        ortSession?.close()
        ortEnv.close()
        Log.i(TAG, "ONNX session and environment closed.")
    }
}

data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classId: Int
) {
    val boundingBox: BoundingBox
        get() = BoundingBox(x1, y1, x2, y2)
}

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
)