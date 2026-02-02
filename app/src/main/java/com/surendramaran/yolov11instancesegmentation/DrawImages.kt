package com.surendramaran.yolov11instancesegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import androidx.core.content.ContextCompat
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.imgproc.Imgproc
import java.util.ArrayList

class DrawImages(private val context: Context) {

    // Using a single color for your specific class
    private val orangeColorInt by lazy {
        ContextCompat.getColor(context, R.color.overlay_orange)
    }

    fun invoke(results: List<SegmentationResult>) : Bitmap {
        if (results.isEmpty()) return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)

        val width = results.first().mask[0].size
        val height = results.first().mask.size
        val combined = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(combined)

        results.forEach { result ->
            // --- PART 1: Draw the Semi-Transparent Mask ---
            for (y in 0 until height) {
                for (x in 0 until width) {
                    if (result.mask[y][x] > 0.5f) {
                        combined.setPixel(x, y, applyTransparentOverlayColor(orangeColorInt))
                    }
                }
            }

            // --- PART 2: Draw the Approximated Quad Outline ---
            val maskMat = floatArrayToMat(result.mask, width, height)
            val contours = ArrayList<MatOfPoint>()
            val hierarchy = Mat()

            Imgproc.findContours(
                maskMat,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
            )

            contours.sortByDescending { Imgproc.contourArea(it) }

            for (contour in contours) {
                if (Imgproc.contourArea(contour) < 2000.0) {
                    contour.release()
                    continue
                }

                val contour2f = MatOfPoint2f(*contour.toArray())
                val perimeter = Imgproc.arcLength(contour2f, true)
                val approx = MatOfPoint2f()

                Imgproc.approxPolyDP(contour2f, approx, 0.02 * perimeter, true)

                if (approx.total() == 4L) {
                    // Draw the outline using the orange color
                    drawOpenCVPoly(canvas, approx)

                    contour2f.release()
                    approx.release()
                    contour.release()
                    break
                }

                contour2f.release()
                approx.release()
                contour.release()
            }

            maskMat.release()
            hierarchy.release()
        }

        return combined
    }

    private fun drawOpenCVPoly(canvas: Canvas, poly: MatOfPoint2f) {
        val points = poly.toArray()
        if (points.isEmpty()) return

        val paint = Paint().apply {
            color = orangeColorInt // Fixed to orange
            strokeWidth = 8F
            style = Paint.Style.STROKE
            isAntiAlias = true
        }

        val path = Path()
        path.moveTo(points[0].x.toFloat(), points[0].y.toFloat())
        for (i in 1 until points.size) {
            path.lineTo(points[i].x.toFloat(), points[i].y.toFloat())
        }
        path.close()

        canvas.drawPath(path, paint)
    }

    private fun applyTransparentOverlayColor(color: Int): Int {
        val alpha = 100
        val red = Color.red(color)
        val green = Color.green(color)
        val blue = Color.blue(color)

        return Color.argb(alpha, red, green, blue)
    }

    private fun floatArrayToMat(mask: Array<FloatArray>, width: Int, height: Int): Mat {
        val mat = Mat(height, width, CvType.CV_8UC1)
        val data = ByteArray(width * height)
        var index = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                data[index++] = if (mask[y][x] > 0.5f) 255.toByte() else 0
            }
        }
        mat.put(0, 0, data)
        return mat
    }
}