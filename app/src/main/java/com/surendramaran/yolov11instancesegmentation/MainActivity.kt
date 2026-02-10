package com.surendramaran.yolov11instancesegmentation

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.surendramaran.yolov11instancesegmentation.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import org.opencv.android.OpenCVLoader
import java.io.IOException

class MainActivity : AppCompatActivity(), InstanceSegmentation.InstanceSegmentationListener {
    private lateinit var binding: ActivityMainBinding
    private var instanceSegmentation: InstanceSegmentation? = null
    private lateinit var drawImages: DrawImages
    private lateinit var previewView: PreviewView
    private var isModelReady = false

    // Stability Tracker Variables
    private var stableFrameCount = 0
    private var lastArea = 0.0
    private var isLocked = false

    private val AREA_THRESHOLD = 0.10 // 10% allowed variance in area
    private val REQUIRED_STABLE_FRAMES = 10 // Approx 1 second at 30fps
    private var isCapturing = false
    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        previewView = binding.previewView
        drawImages = DrawImages(applicationContext)

        //  CRITICAL FIX: Initialize models in background thread
        showLoadingUI(true)
        // In onCreate or similar initialization block
        binding.ivTop.setOnClickListener {
            isCapturing = false
            isLocked = false
            stableFrameCount = 0
            startCamera() // Re-bind the camera preview and analyzer
            Toast.makeText(this, "Scanner Ready", Toast.LENGTH_SHORT).show()
        }
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Load OpenCV in background
                val opencvLoaded = OpenCVLoader.initDebug()
                
                withContext(Dispatchers.Main) {
                    if (!opencvLoaded) {
                        Toast.makeText(this@MainActivity, "OpenCV load failed!", Toast.LENGTH_LONG).show()
                    }
                }
                
                // Load model in background (this takes 6-15s)
                val segmentation = InstanceSegmentation(
                    context = applicationContext,
                    modelPath = "v6.tflite",
                    labelPath = null,
                    instanceSegmentationListener = this@MainActivity,
                    message = { msg ->
                        runOnUiThread {
                            Toast.makeText(applicationContext, msg, Toast.LENGTH_SHORT).show()
                        }
                    },
                )
                
                instanceSegmentation = segmentation
                isModelReady = true
                
                withContext(Dispatchers.Main) {
                    showLoadingUI(false)
                    Toast.makeText(this@MainActivity, "Model loaded! Starting camera...", Toast.LENGTH_SHORT).show()
                    checkPermission()
                }
                
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Initialization failed: ${e.message}", Toast.LENGTH_LONG).show()
                    showLoadingUI(false)
                }
                Log.e("MainActivity", "Init failed", e)
            }
        }
    }

    private fun showLoadingUI(show: Boolean) {
        binding.loadingText.visibility = if (show) View.VISIBLE else View.GONE
        binding.loadingProgress.visibility = if (show) View.VISIBLE else View.GONE
        binding.previewView.visibility = if (show) View.GONE else View.VISIBLE
    }

    private fun startCamera() {
        if (!isModelReady) {
            Toast.makeText(this, "Model not ready yet...", Toast.LENGTH_SHORT).show()
            return
        }
        
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val aspectRatio = AspectRatio.RATIO_4_3

            val preview = Preview.Builder()
                .setTargetAspectRatio(aspectRatio)
                .build()
                .also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(aspectRatio)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(Executors.newSingleThreadExecutor(), ImageAnalyzer())
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e("CameraX", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    inner class ImageAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            if (!isModelReady || instanceSegmentation == null) {
                imageProxy.close()
                return
            }

            val bitmapBuffer =
                Bitmap.createBitmap(
                    imageProxy.width,
                    imageProxy.height,
                    Bitmap.Config.ARGB_8888
                )
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
            )
            instanceSegmentation?.invoke(rotatedBitmap)
        }
    }

    private fun checkPermission() = lifecycleScope.launch(Dispatchers.IO) {
        val isGranted = REQUIRED_PERMISSIONS.all {
            ActivityCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
        }
        if (isGranted) {
            withContext(Dispatchers.Main) {
                startCamera()
            }
        } else {
            withContext(Dispatchers.Main) {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) { map ->
            if(map.all { it.value }) {
                startCamera()
            } else {
                Toast.makeText(baseContext, "Permission required", Toast.LENGTH_LONG).show()
            }
        }

    override fun onError(error: String) {
        runOnUiThread {
            Toast.makeText(applicationContext, error, Toast.LENGTH_SHORT).show()
            binding.ivTop.setImageResource(0)
        }
    }

    override fun onDetect(
        interfaceTime: Long,
        results: List<SegmentationResult>,
        preProcessTime: Long,
        postProcessTime: Long
    ) {
        // Step 1: Pass the current 'isLocked' state to the drawer
        val detectionState = drawImages.invoke(results, isLocked)

        runOnUiThread {
            // Step 2: Run Stability Logic
            updateStability(detectionState, results)

            // Step 3: Update UI
            binding.tvPreprocess.text = "Pre: ${preProcessTime}ms"
            binding.tvInference.text = "Inf: ${interfaceTime}ms"
            binding.tvPostprocess.text = "Post: ${postProcessTime}ms"
            binding.ivTop.setImageBitmap(detectionState.bitmap)
        }
    }
    private fun updateStability(state: DetectionState, results: List<SegmentationResult>) {
        if (isCapturing) return
        if (state.isQuadFound) {
            // Check Area Stability: current area vs last frame's area
            val areaDiff = if (lastArea > 0) Math.abs(state.area - lastArea) / lastArea else 0.0

            if (areaDiff <= AREA_THRESHOLD) {
                stableFrameCount++
                Log.d("StabilityTracker", "Stabilizing: $stableFrameCount/$REQUIRED_STABLE_FRAMES")
            } else {
                // Reset if the document moved too much
                stableFrameCount = 0
                Log.d("StabilityTracker", "Reset: Movement (Diff: ${String.format("%.2f", areaDiff)})")
            }

            lastArea = state.area
        } else {
            if (stableFrameCount > 0) Log.d("StabilityTracker", "Reset: Quad lost")
            // Reset if no quad is detected (Geometric Validity failure)
            stableFrameCount = 0
            lastArea = 0.0
        }
        val wasLocked = isLocked
        // Final Stability Check
        isLocked = stableFrameCount >= REQUIRED_STABLE_FRAMES
        if (isLocked && !wasLocked) {
            Log.i("StabilityTracker", "🟢 DOCUMENT LOCKED - STABILITY ACHIEVED")
        }
        if (isLocked && !isCapturing && results.isNotEmpty()) {
            captureDocument(state, results.first())
        }
    }
    private fun captureDocument(state: DetectionState, result: SegmentationResult) {
        if (isCapturing) return
        isCapturing = true

        // 1. Stop the Camera Analyzer immediately to "freeze" the frame
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            cameraProvider.unbindAll() // This stops the preview and analyzer
        }, ContextCompat.getMainExecutor(this))

        // 2. Play Sound
        val view = window.decorView
        view.playSoundEffect(android.view.SoundEffectConstants.CLICK)

        // 3. Save and Crop
        lifecycleScope.launch(Dispatchers.IO) {
            val uri = saveAndCropBitmap(this@MainActivity, state.bitmap, result.box)

            withContext(Dispatchers.Main) {
                if (uri != null) {
                    Toast.makeText(this@MainActivity, "Saved to Gallery", Toast.LENGTH_LONG).show()
                    // Show the final cropped version on top
                    binding.ivTop.setImageBitmap(state.bitmap)
                }
            }
        }
    }
    fun saveAndCropBitmap(context: Context, fullBitmap: Bitmap, box: Output0): Uri? {
        // 1. Calculate Crop Coordinates (converting normalized 0.0-1.0 to pixel values)
        val left = (box.x1 * fullBitmap.width).toInt().coerceAtLeast(0)
        val top = (box.y1 * fullBitmap.height).toInt().coerceAtLeast(0)
        val width = ((box.x2 - box.x1) * fullBitmap.width).toInt().coerceAtMost(fullBitmap.width - left)
        val height = ((box.y2 - box.y1) * fullBitmap.height).toInt().coerceAtMost(fullBitmap.height - top)

        val croppedBitmap = Bitmap.createBitmap(fullBitmap, left, top, width, height)

        // 2. Save to Gallery using MediaStore
        val filename = "SCAN_${System.currentTimeMillis()}.jpg"
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
        }

        val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
// In saveAndCropBitmap function
        uri?.let {
            // Add the safe call ?. before use
            context.contentResolver.openOutputStream(it)?.use { stream ->
                croppedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream)
            } ?: throw IOException("Failed to open output stream.")
        }
        return uri
    }

    override fun onEmpty() {
        runOnUiThread {
            binding.ivTop.setImageResource(0)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        instanceSegmentation?.close()
    }

    companion object {
        val REQUIRED_PERMISSIONS = mutableListOf (
            Manifest.permission.CAMERA
        ).toTypedArray()
    }
}
