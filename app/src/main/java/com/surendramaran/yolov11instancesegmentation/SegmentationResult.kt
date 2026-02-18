package com.surendramaran.yolov11instancesegmentation

// In SegmentationResult.kt
data class SegmentationResult(
    val box: Output0,
    val mask: Array<FloatArray>,
    val quadPoints: List<org.opencv.core.Point>? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as SegmentationResult

        return mask.contentDeepEquals(other.mask)
    }

    override fun hashCode(): Int {
        return mask.contentDeepHashCode()
    }
}