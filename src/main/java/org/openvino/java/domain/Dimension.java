package org.openvino.java.domain;

import org.openvino.java.OpenVINO;

/**
 * Class representing a dimension, which may be dynamic (undetermined until runtime),in a shape or shape-like object.
 * Static dimensions may be implicitly converted from value_type.
 * A dynamic dimension is constructed with Dimension() or Dimension::dynamic().
 */
public class Dimension {

    /**
     * The ov_dimension struct.
     */
    OvDimension mDimension;

    /**
     * Construct a static dimension.
     *
     * @param dimension Value of the dimension.
     */
    public Dimension(long dimension) {
        mDimension = new OvDimension();
        mDimension.min = dimension;
        mDimension.max = dimension;
    }

    /**
     * Construct a dynamic dimension with ov_dimension struct.
     *
     * @param ovDimension The ov_dimension struct.
     */
    public Dimension(OvDimension ovDimension) {
        this.mDimension = ovDimension;
    }

    /**
     * Construct a dynamic dimension with bounded range
     *
     * @param min_dimension The lower inclusive limit for the dimension
     * @param max_dimension The upper inclusive limit for the dimension
     */
    public Dimension(long min_dimension, long max_dimension) {
        mDimension = new OvDimension();
        mDimension.min = min_dimension;
        mDimension.max = max_dimension;
    }

    /**
     * Get ov_dimension struct.
     *
     * @return Return ov_dimension struct.
     */
    public OvDimension getDimension() {
        return mDimension;
    }

    /**
     * Get max.
     *
     * @return Dimension max.
     */
    public long getMax() {
        return mDimension.max;
    }

    /**
     * Get min.
     *
     * @return Dimension min.
     */
    public long getMin() {
        return mDimension.min;
    }

    /**
     * Check this dimension whether is dynamic
     *
     * @return Boolean, true is dynamic and false is static.
     */
    public boolean isDynamic() {
        return OpenVINO.getCore().ov_dimension_is_dynamic(mDimension);
    }
}
