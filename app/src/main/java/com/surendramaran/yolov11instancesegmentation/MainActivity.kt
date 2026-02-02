package com.surendramaran.yolov11instancesegmentation

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
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

class MainActivity : AppCompatActivity(), InstanceSegmentation.InstanceSegmentationListener {
    private lateinit var binding: ActivityMainBinding
    private var instanceSegmentation: InstanceSegmentation? = null
    private lateinit var drawImages: DrawImages
    private lateinit var previewView: PreviewView
    private var isModelReady = false

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
        val image = drawImages.invoke(results)
        runOnUiThread {
            binding.tvPreprocess.text = preProcessTime.toString()
            binding.tvInference.text = interfaceTime.toString()
            binding.tvPostprocess.text = postProcessTime.toString()
            binding.ivTop.setImageBitmap(image)
        }
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
