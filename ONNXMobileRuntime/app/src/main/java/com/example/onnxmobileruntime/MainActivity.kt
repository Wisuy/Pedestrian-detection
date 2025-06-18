package com.example.onnxmobileruntime

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.onnxmobileruntime.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Color
import android.os.SystemClock // Import for SystemClock.elapsedRealtime()

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var yoloDetector: YoloDetector

    // Flag to avoid detection calls after detector is closed
    @Volatile
    private var isDetectorActive = false

    private var imageProxyProcessedWidth: Int = 0
    private var imageProxyProcessedHeight: Int = 0

    private var modelInputScaledWidth: Int = 0
    private var modelInputScaledHeight: Int = 0
    private var modelInputPaddingX: Int = 0
    private var modelInputPaddingY: Int = 0

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            Log.d(TAG, "Camera permission granted")
            startCamera()
        } else {
            Log.e(TAG, "Camera permission denied")
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        try {
            yoloDetector = YoloDetector(this, "model.onnx")
            isDetectorActive = true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize YoloDetector: ${e.message}", e)
            Toast.makeText(this, "Failed to load model: ${e.message}", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(YoloDetector.INPUT_SIZE, YoloDetector.INPUT_SIZE)) // Model input size
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImageProxy(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                val camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)

                binding.viewFinder.post {
                    preview.resolutionInfo?.resolution?.let { resolution ->
                        Log.d(TAG, "PreviewView Surface Resolution: ${resolution.width}x${resolution.height}")
                    }
                }

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImageProxy(imageProxy: ImageProxy) {
        if (!isDetectorActive) {
            imageProxy.close()
            return
        }

        val startTime = SystemClock.elapsedRealtime()

        val bitmap: Bitmap
        try {
            val bitmapConversionStart = SystemClock.elapsedRealtime()
            bitmap = imageProxy.toBitmap()
            val bitmapConversionEnd = SystemClock.elapsedRealtime()
            Log.d(TAG, "Time to convert ImageProxy to Bitmap: ${bitmapConversionEnd - bitmapConversionStart}ms")

            imageProxyProcessedWidth = bitmap.width
            imageProxyProcessedHeight = bitmap.height
            Log.d(TAG, "Bitmap created from ImageProxy size (ImageAnalysis output): ${imageProxyProcessedWidth}x${imageProxyProcessedHeight}")
        } catch (e: Exception) {
            Log.e(TAG, "Error converting ImageProxy to Bitmap: ${e.message}", e)
            imageProxy.close()
            return
        }

        if (!isDetectorActive) {
            bitmap.recycle()
            imageProxy.close()
            return
        }

        val letterboxStart = SystemClock.elapsedRealtime()
        val preparedBitmap = letterboxBitmap(bitmap, YoloDetector.INPUT_SIZE)
        val letterboxEnd = SystemClock.elapsedRealtime()
        Log.d(TAG, "Time to letterbox bitmap: ${letterboxEnd - letterboxStart}ms")

        bitmap.recycle() // Recycle the original bitmap immediately

        val detectionStart = SystemClock.elapsedRealtime()
        yoloDetector.detect(preparedBitmap) { detections ->
            val detectionEnd = SystemClock.elapsedRealtime()
            Log.d(TAG, "Time for YoloDetector.detect() (inference + post-process): ${detectionEnd - detectionStart}ms")

            if (!isDetectorActive || this@MainActivity.isFinishing || this@MainActivity.isDestroyed) {
                preparedBitmap.recycle()
                return@detect
            }

            // Pass the model's expected input size (640x640),
            // the *actual* dimensions of the bitmap that ImageAnalysis provided (e.g., 1088x1088),
            // AND the actual dimensions and padding used for the letterboxed image for the model
            runOnUiThread {
                binding.overlayView.setDetections(
                    detections,
                    YoloDetector.INPUT_SIZE, // modelInputW (640)
                    YoloDetector.INPUT_SIZE, // modelInputH (640)
                    imageProxyProcessedWidth, // imageProxyProcessedW (e.g., 1088)
                    imageProxyProcessedHeight, // imageProxyProcessedH (e.g., 1088)
                    modelInputScaledWidth, // actual content width within the 640x640 (e.g., 640)
                    modelInputScaledHeight, // actual content height within the 640x640 (e.g., 480)
                    modelInputPaddingX, // padding on X axis for letterbox (e.g., 0)
                    modelInputPaddingY // padding on Y axis for letterbox (e.g., 80)
                )
            }
        }
        preparedBitmap.recycle()
        imageProxy.close()

        val totalProcessTime = SystemClock.elapsedRealtime() - startTime
        Log.d(TAG, "Total processImageProxy execution time: ${totalProcessTime}ms")
    }

    /**
     * Resizes the original bitmap to fit within the targetSize x targetSize
     * while maintaining aspect ratio, and pads the remaining space with black.
     * Stores the actual scaled dimensions and padding for later use.
     */
    private fun letterboxBitmap(originalBitmap: Bitmap, targetSize: Int): Bitmap {
        val originalWidth = originalBitmap.width
        val originalHeight = originalBitmap.height

        val scale: Float
        val scaledWidth: Int
        val scaledHeight: Int
        val paddingX: Int
        val paddingY: Int

        // Calculate scaling factor to fit into targetSize while preserving aspect ratio
        if (originalWidth > originalHeight) { // Landscape or square original
            scale = targetSize.toFloat() / originalWidth.toFloat()
            scaledWidth = targetSize
            scaledHeight = (originalHeight * scale).toInt()
        } else { // Portrait original
            scale = targetSize.toFloat() / originalHeight.toFloat()
            scaledHeight = targetSize
            scaledWidth = (originalWidth * scale).toInt()
        }

        // Calculate padding to center the scaled image in the target square
        paddingX = (targetSize - scaledWidth) / 2
        paddingY = (targetSize - scaledHeight) / 2

        // Store these values to pass to OverlayView
        modelInputScaledWidth = scaledWidth
        modelInputScaledHeight = scaledHeight
        modelInputPaddingX = paddingX
        modelInputPaddingY = paddingY

        Log.d(TAG, "Letterboxing: Original ${originalWidth}x${originalHeight} -> Scaled ${scaledWidth}x${scaledHeight} with padding X=${paddingX}, Y=${paddingY} into ${targetSize}x${targetSize}")

        // Create a new bitmap of targetSize x targetSize (e.g., 640x640) with black background
        val letterboxedBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(letterboxedBitmap)
        canvas.drawColor(Color.BLACK) // Fill with black

        // Draw the scaled original bitmap onto the new canvas at the calculated offsets
        val matrix = Matrix()
        matrix.postScale(scale, scale)
        matrix.postTranslate(paddingX.toFloat(), paddingY.toFloat())
        canvas.drawBitmap(originalBitmap, matrix, null)

        return letterboxedBitmap
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer
        val vuBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val vuSize = vuBuffer.remaining()

        val nv21 = ByteArray(ySize + vuSize)
        yBuffer.get(nv21, 0, ySize)
        vuBuffer.get(nv21, ySize, vuSize)

        val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val imageBytes = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    override fun onDestroy() {
        isDetectorActive = false

        cameraExecutor.shutdown()

        try {
            yoloDetector.close()
        } catch (e: Exception) {
            Log.w(TAG, "Exception while closing YoloDetector: ${e.message}")
        }

        super.onDestroy()
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}