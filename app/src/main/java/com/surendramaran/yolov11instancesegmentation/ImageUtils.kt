package com.surendramaran.yolov11instancesegmentation

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import java.io.IOException
import kotlin.math.exp

object ImageUtils {
    fun List<Array<FloatArray>>.clone(): List<Array<FloatArray>> {
        return this.map { array -> array.map { it.clone() }.toTypedArray() }
    }

    fun Array<FloatArray>.scaleMask(targetWidth: Int, targetHeight: Int): Array<FloatArray> {
        val originalHeight = this.size
        val originalWidth = this[0].size

        val xRatio = (originalWidth shl 16) / targetWidth
        val yRatio = (originalHeight shl 16) / targetHeight

        val output = Array(targetHeight) { FloatArray(targetWidth) }

        for (y in 0 until targetHeight) {
            val origY = (y * yRatio) ushr 16
            for (x in 0 until targetWidth) {
                val origX = (x * xRatio) ushr 16
                output[y][x] = this[origY][origX]
            }
        }

        return output
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

    fun Array<FloatArray>.toMask(): Array<IntArray> =
        map { row -> row.map { if (it > 0) 1 else 0 }.toIntArray() }.toTypedArray()


    fun Array<IntArray>.smooth(kernel: Int) : Array<IntArray> {
        // Using Array because it is faster then List
        val maskFloat = Array(this.size) { i ->
            FloatArray(this[i].size) { j ->
                if (this[i][j] > 0) 1F else 0F
            }
        }
        val gaussianKernel = createGaussianKernel(kernel)
        val blurredImage = applyGaussianBlur(maskFloat, gaussianKernel)
        return thresholdImage(blurredImage)
    }

    private fun createGaussianKernel(size: Int): Array<FloatArray> {
        val sigma = 2F
        val kernel = Array(size) { FloatArray(size) }
        val mean = size / 2
        var sum = 0F

        for (x in 0 until size) {
            for (y in 0 until size) {
                kernel[x][y] = (1F / (2F * Math.PI.toFloat() * sigma * sigma)) * exp(
                    -((x - mean) * (x - mean) + (y - mean) * (y - mean)) / (2F * sigma * sigma)
                )
                sum += kernel[x][y]
            }
        }

        for (x in 0 until size) {
            for (y in 0 until size) {
                kernel[x][y] /= sum
            }
        }

        return kernel
    }

    private fun applyGaussianBlur(image: Array<FloatArray>, kernel: Array<FloatArray>): Array<FloatArray> {
        val height = image.size
        val width = image[0].size
        val kernelSize = kernel.size
        val offset = kernelSize / 2
        val blurredImage = Array(height) { FloatArray(width) }

        for (i in image.indices) {
            for (j in image[i].indices) {
                if (i < offset || j < offset || i >= height - offset || j >= width - offset) {
                    blurredImage[i][j] = image[i][j]
                    continue
                }

                var sum = 0F
                for (ki in kernel.indices) {
                    for (kj in kernel[ki].indices) {
                        sum += image[i - offset + ki][j - offset + kj] * kernel[ki][kj]
                    }
                }
                blurredImage[i][j] = sum
            }
        }

        return blurredImage
    }

    private fun thresholdImage(image: Array<FloatArray>): Array<IntArray> {
        val height = image.size
        val width = image[0].size
        return Array(height) { i ->
            IntArray(width) { j ->
                if (image[i][j] > 0.9F) 1 else 0
            }
        }
    }
}