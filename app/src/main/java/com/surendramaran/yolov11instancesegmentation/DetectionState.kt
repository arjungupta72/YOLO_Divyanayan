package com.surendramaran.yolov11instancesegmentation

import android.graphics.Bitmap
import org.opencv.core.Point

data class DetectionState(
    val bitmap: Bitmap,
    val isQuadFound: Boolean,
    val area: Double,
    val quadPoints: List<Point>? = null
)