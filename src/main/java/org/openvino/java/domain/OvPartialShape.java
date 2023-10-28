package org.openvino.java.domain;

import com.sun.jna.Pointer;

/**
 * It represents a shape that may be partially or totally dynamic.
 * Dynamic rank. (Informal notation: `?`)
 * Static rank, but dynamic dimensions on some or all axes.
 * (Informal notation examples: `{1,2,?,4}`, `{?,?,?}`)
 * Static rank, and static dimensions on all axes.
 * (Informal notation examples: `{1,2,3,4}`, `{6}`, `{}`)
 */
public class OvPartialShape {

    /**
     * The rank
     */
    public OvDimension rank;

    /**
     * The dimension
     */
    public Pointer dims;

}
