package com.surendramaran.yolov11instancesegmentation

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.surendramaran.yolov11instancesegmentation.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), InstanceSegmentation.InstanceSegmentationListener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var previewView: PreviewView
    private lateinit var drawImages: DrawImages
    private lateinit var imageCapture: ImageCapture

    private var instanceSegmentation: InstanceSegmentation? = null
    private var isModelReady = false
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    @Volatile
    private var latestBox: Output0? = null

    // Stability & Capture State
    private var stableFrameCount = 0
    private var lastArea = 0.0
    private var isLocked = false
    private var isCapturing = false

    private val AREA_THRESHOLD = 0.10
    private val REQUIRED_STABLE_FRAMES = 15 // Adjusted for better balance

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        previewView = binding.previewView
        drawImages = DrawImages(applicationContext)

        showLoadingUI(true)

        lifecycleScope.launch(Dispatchers.IO) {
            val loaded = OpenCVLoader.initDebug()
            instanceSegmentation = InstanceSegmentation(
                context = applicationContext,
                modelPath = "v6.tflite",
                labelPath = null,
                instanceSegmentationListener = this@MainActivity,
                message = {}
            )
            isModelReady = true

            withContext(Dispatchers.Main) {
                showLoadingUI(false)
                checkPermission()
            }
        }
    }

    private fun showLoadingUI(show: Boolean) {
        binding.loadingText.visibility = if (show) View.VISIBLE else View.GONE
        binding.loadingProgress.visibility = if (show) View.VISIBLE else View.GONE
        binding.previewView.visibility = if (show) View.GONE else View.VISIBLE
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val aspectRatio = AspectRatio.RATIO_4_3

            val preview = Preview.Builder()
                .setTargetAspectRatio(aspectRatio)
                .build()
                .also { it.surfaceProvider = previewView.surfaceProvider }

            // Friend's Fix: High-res ImageCapture use case
            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(aspectRatio)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()

            val analyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(aspectRatio)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ImageAnalyzer()) }

            provider.unbindAll()
            try {
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analyzer, imageCapture)
            } catch (exc: Exception) {
                Log.e("CameraX", "Binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // Original Logic: Restored Rotation Handling
    inner class ImageAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(image: ImageProxy) {
            if (!isModelReady || instanceSegmentation == null || isCapturing) {
                image.close()
                return
            }

            val bitmapBuffer = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
            image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

            // Rotate bitmap to match screen orientation (Portrait)
            val matrix = Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }
            val rotatedBitmap = Bitmap.createBitmap(bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true)

            instanceSegmentation?.invoke(rotatedBitmap)
            image.close()
        }
    }

    override fun onDetect(interfaceTime: Long, results: List<SegmentationResult>, preProcessTime: Long, postProcessTime: Long) {
        if (results.isNotEmpty()) latestBox = results.first().box

        val state = drawImages.invoke(results, isLocked)

        runOnUiThread {
            updateStability(state, results)
            binding.ivTop.setImageBitmap(state.bitmap)
        }
    }

    private fun updateStability(state: DetectionState, results: List<SegmentationResult>) {
        if (isCapturing) return

        if (state.isQuadFound) {
            val diff = if (lastArea > 0) Math.abs(state.area - lastArea) / lastArea else 0.0
            if (diff <= AREA_THRESHOLD) stableFrameCount++ else stableFrameCount = 0
            lastArea = state.area
        } else {
            stableFrameCount = 0
            lastArea = 0.0
        }

        isLocked = stableFrameCount >= REQUIRED_STABLE_FRAMES

        if (isLocked && results.isNotEmpty()) {
            captureDocument()
        }
    }

    private fun captureDocument() {
        if (isCapturing) return
        isCapturing = true

        // Play Shutter Sound
        window.decorView.playSoundEffect(android.view.SoundEffectConstants.CLICK)

        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, "SCAN_${System.currentTimeMillis()}.jpg")
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
        }

        val outputOptions = ImageCapture.OutputFileOptions.Builder(contentResolver, MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values).build()

        imageCapture.takePicture(outputOptions, cameraExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                val uri = result.savedUri ?: return
                lifecycleScope.launch(Dispatchers.IO) {
                    val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                    val box = latestBox ?: return@launch

                    // Corrected Crop: Maps rotated AI coordinates to the captured bitmap
                    val cropped = crop(bitmap, box)

                    contentResolver.openOutputStream(uri)?.use {
                        cropped.compress(Bitmap.CompressFormat.JPEG, 100, it)
                    }

                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Document Saved!", Toast.LENGTH_LONG).show()
                        isCapturing = false
                        stableFrameCount = 0 // Reset for next scan
                    }
                }
            }

            override fun onError(exception: ImageCaptureException) {
                isCapturing = false
                Log.e("Capture", "Error: ${exception.message}")
            }
        })
    }

    private fun crop(bitmap: Bitmap, box: Output0): Bitmap {

        val left = (box.x1 * bitmap.width).toInt().coerceAtLeast(0)
        val top = (box.y1 * bitmap.height).toInt().coerceAtLeast(0)
        val right = (box.x2 * bitmap.width).toInt().coerceAtMost(bitmap.width)
        val bottom = (box.y2 * bitmap.height).toInt().coerceAtMost(bitmap.height)

        val width = (right - left).coerceAtLeast(1)
        val height = (bottom - top).coerceAtLeast(1)

        return Bitmap.createBitmap(bitmap, left, top, width, height)
    }

    private fun checkPermission() {
        val granted = REQUIRED_PERMISSIONS.all { ActivityCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }
        if (granted) startCamera() else requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { if (it.all { p -> p.value }) startCamera() }

    override fun onError(error: String) {}
    override fun onEmpty() {}
    override fun onDestroy() {
        super.onDestroy()
        instanceSegmentation?.close()
        cameraExecutor.shutdown()
    }

    companion object {
        val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}