package org.openvino.java.enums;

/**
 * This enum contains enumerations for color format.
 */
public enum ColorFormat {
    /**
     * Undefine color format
     */
    UNDEFINE,
    /**
     * Image in NV12 format as single tensor
     */
    NV12_SINGLE_PLANE,
    /**
     *Image in NV12 format represented as separate tensors for Y and UV planes.
     */
    NV12_TWO_PLANES,
    /**
     * Image in I420 (YUV) format as single tensor
     */
    I420_SINGLE_PLANE,
    /**
     * Image in I420 format represented as separate tensors for Y, U and V planes.
     */
    I420_THREE_PLANES,
    /**
     * Image in RGB interleaved format (3 channels)
     */
    RGB,
    /**
     * Image in BGR interleaved format (3 channels)
     */
    BGR,
    /**
     * Image in GRAY format (1 channel)
     */
    GRAY,
    /**
     * Image in RGBX interleaved format (4 channels)
     */
    RGBX,
    /**
     * Image in BGRX interleaved format (4 channels)
     */
    BGRX
}
