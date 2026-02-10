package com.surendramaran.yolov11instancesegmentation

import android.graphics.Bitmap

data class DetectionState(
    val bitmap: Bitmap,
    val isQuadFound: Boolean,
    val area: Double
)