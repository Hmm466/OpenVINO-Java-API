package org.openvino.java.domain;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

/**
 * This is a structure interface equal to ov::Dimension
 */
public class OvDimension extends Structure {

    /**
     * The lower inclusive limit for the dimension.
     */
    public long min;

    /**
     * The upper inclusive limit for the dimension.
     */
    public long max;

    public OvDimension() {
        super();
    }

    public OvDimension(Pointer pointer) {
        super(pointer);
    }
}
