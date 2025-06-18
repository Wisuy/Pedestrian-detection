package com.example.onnxmobileruntime

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.View
import kotlin.math.max
import kotlin.math.min

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<Detection> = emptyList()
    private var modelInputWidth: Int = 0 // 640
    private var modelInputHeight: Int = 0 // 640

    // Renamed from cameraPreviewRealWidth/Height to reflect the actual ImageProxy dimensions
    private var imageProxyProcessedWidth: Int = 0 // e.g., 1088 (actual size of bitmap from ImageProxy)
    private var imageProxyProcessedHeight: Int = 0 // e.g., 1088

    private var modelInputScaledWidth: Int = 0 // actual width of content in 640x640 (e.g., 640)
    private var modelInputScaledHeight: Int = 0 // actual height of content in 640x640 (e.g., 480)
    private var modelInputPaddingX: Int = 0 // padding on X in 640x640 (e.g., 0)
    private var modelInputPaddingY: Int = 0 // padding on Y in 640x640 (e.g., 80)


    // Paint for bounding box
    private val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    // Paint for text label
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        style = Paint.Style.FILL
    }

    // Separate Paint for label background to avoid modifying boxPaint dynamically
    private val labelBgPaint = Paint().apply {
        color = Color.BLUE
        style = Paint.Style.FILL
    }

    // Set detections and all relevant image sizes, then redraw
    fun setDetections(
        detections: List<Detection>,
        modelInputW: Int,
        modelInputH: Int,
        imageProxyProcessedW: Int, // Changed parameter name
        imageProxyProcessedH: Int, // Changed parameter name
        modelScaledW: Int,
        modelScaledH: Int,
        modelPadX: Int,
        modelPadY: Int
    ) {
        this.detections = detections
        this.modelInputWidth = modelInputW
        this.modelInputHeight = modelInputH
        this.imageProxyProcessedWidth = imageProxyProcessedW // Assigned new parameter
        this.imageProxyProcessedHeight = imageProxyProcessedH // Assigned new parameter
        this.modelInputScaledWidth = modelScaledW
        this.modelInputScaledHeight = modelScaledH
        this.modelInputPaddingX = modelPadX
        this.modelInputPaddingY = modelPadY
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (detections.isEmpty() || modelInputWidth == 0 || modelInputHeight == 0 ||
            imageProxyProcessedWidth == 0 || imageProxyProcessedHeight == 0 || // Changed condition
            modelInputScaledWidth == 0 || modelInputScaledHeight == 0) {
            return
        }

        Log.d(TAG, "Canvas size: ${width}x$height")
        Log.d(TAG, "Model Input size: ${modelInputWidth}x${modelInputHeight}")
        Log.d(TAG, "ImageProxy Processed size: ${imageProxyProcessedWidth}x${imageProxyProcessedHeight}") // Changed log
        Log.d(TAG, "Model Letterboxed Content Size: ${modelInputScaledWidth}x${modelInputScaledHeight} with padding X=${modelInputPaddingX}, Y=${modelInputPaddingY}")


        // Calculate the actual content area within the OverlayView (excluding red bars)
        // This simulates PreviewView.ScaleType.FIT_CENTER
        val viewAspectRatio = width.toFloat() / height.toFloat()
        // Use the imageProxyProcessed dimensions for aspect ratio calculation
        val cameraAspectRatio = imageProxyProcessedWidth.toFloat() / imageProxyProcessedHeight.toFloat()

        var displayedImageWidth: Float
        var displayedImageHeight: Float
        var offsetX = 0f
        var offsetY = 0f

        if (viewAspectRatio > cameraAspectRatio) { // View is wider than camera feed (pillarboxing/red bars on sides)
            displayedImageHeight = height.toFloat()
            displayedImageWidth = displayedImageHeight * cameraAspectRatio
            offsetX = (width - displayedImageWidth) / 2f
            offsetY = 0f
        } else { // View is taller than camera feed (letterboxing/red bars on top/bottom) - less likely given your observation
            displayedImageWidth = width.toFloat()
            displayedImageHeight = displayedImageWidth / cameraAspectRatio
            offsetY = (height - displayedImageHeight) / 2f
            offsetX = 0f
        }

        Log.d(TAG, "Displayed Image Area: ${displayedImageWidth}x${displayedImageHeight} at offset (${offsetX}, ${offsetY})")

        // Scale factors from imageProxyProcessed dimensions to the displayed image dimensions
        val scaleFactorDisplayX = displayedImageWidth / imageProxyProcessedWidth.toFloat() // Changed divisor
        val scaleFactorDisplayY = displayedImageHeight / imageProxyProcessedHeight.toFloat() // Changed divisor


        for (detection in detections) {
            val boundingBox = detection.boundingBox

            // Step 1: Adjust bounding box for padding within the model's 640x640 input
            // The detections are normalized (0-1) to the 640x640 input, but the *content* is smaller
            // and offset by padding.
            val x1_adjusted = boundingBox.x1 * modelInputWidth - modelInputPaddingX
            val y1_adjusted = boundingBox.y1 * modelInputHeight - modelInputPaddingY
            val x2_adjusted = boundingBox.x2 * modelInputWidth - modelInputPaddingX
            val y2_adjusted = boundingBox.y2 * modelInputHeight - modelInputPaddingY

            // Step 2: Normalize these adjusted coordinates to the *actual content size* within the 640x640
            // These should now be 0-1 values relative to the scaled image content.
            val x1_norm_content = x1_adjusted / modelInputScaledWidth
            val y1_norm_content = y1_adjusted / modelInputScaledHeight
            val x2_norm_content = x2_adjusted / modelInputScaledWidth
            val y2_norm_content = y2_adjusted / modelInputScaledHeight

            // Step 3: Scale these normalized content coordinates to the ImageProxy's processed dimensions (e.g., 1088x1088)
            val final_x1_real = x1_norm_content * imageProxyProcessedWidth // Changed multiplier
            val final_y1_real = y1_norm_content * imageProxyProcessedHeight // Changed multiplier
            val final_x2_real = x2_norm_content * imageProxyProcessedWidth // Changed multiplier
            val final_y2_real = y2_norm_content * imageProxyProcessedHeight // Changed multiplier

            // Step 4: Apply the scaling and offset for the `fitCenter` display on the OverlayView
            val left = (final_x1_real * scaleFactorDisplayX) + offsetX
            val top = (final_y1_real * scaleFactorDisplayY) + offsetY
            val right = (final_x2_real * scaleFactorDisplayX) + offsetX
            val bottom = (final_y2_real * scaleFactorDisplayY) + offsetY

            Log.d(TAG, "Raw bounding box coords: $boundingBox")
            Log.d(TAG, "Adjusted for model padding: ($x1_adjusted, $y1_adjusted) - ($x2_adjusted, $y2_adjusted)")
            Log.d(TAG, "Normalized to content: ($x1_norm_content, $y1_norm_content) - ($x2_norm_content, $y2_norm_content)")
            Log.d(TAG, "Scaled to ImageProxy processed pixels: ($final_x1_real, $final_y1_real) - ($final_x2_real, $final_y2_real)") // Changed log
            Log.d(TAG, "Scaled and offset for display: ($left, $top) - ($right, $bottom)")

            // Draw bounding box
            val rectF = RectF(left, top, right, bottom)
            canvas.drawRect(rectF, boxPaint)

            // Prepare label
            val label = "Pedestrian: ${String.format("%.2f", detection.confidence * 100)}%"
            val textWidth = textPaint.measureText(label)
            val textHeight = textPaint.descent() - textPaint.ascent()

            // Draw background rectangle for label (above the box)
            canvas.drawRect(left, top - textHeight, min(left + textWidth, width.toFloat()), top, labelBgPaint)

            // Draw text label
            canvas.drawText(label, left, top - textPaint.descent(), textPaint)
        }
    }

    companion object {
        private const val TAG = "OverlayView"
    }
}