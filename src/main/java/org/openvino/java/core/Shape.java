package org.openvino.java.core;

import com.sun.jna.ptr.LongByReference;
import org.openvino.java.OpenVINO;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.domain.OvShape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Shape for a tensor.
 */
public class Shape extends OpenVINOCls {

    /**
     * [struct] The shape ov_shape
     */
    private OvShape shape;

    /**
     *
     */
    private long rank;

    /**
     *
     */
    private List<Long> dims;

    /**
     * Constructs Shape from the initialized IntPtr.
     *
     * @param shapeT Initialized IntPtr
     */
    public Shape(OvShape shapeT) {
        super("Shape");
        shape = shapeT;
        getVinoObj().setPointer(shape.getPointer());
        dims = new ArrayList<>();
        rank = shapeT.rank;
        long[] data = shapeT.dims.getPointer().getLongArray(0, (int) shapeT.rank);
        for (int i = 0; i < data.length; i++) {
            dims.add(data[i]);
        }
    }

    /**
     * Constructs Shape from the list.
     *
     * @param axisLengths Initialized list
     */
    public Shape(List<Long> axisLengths) {
        super("Shape", null);
        dims = new ArrayList<>();
        dims.addAll(axisLengths);
        shape = new OvShape();
        LongByReference longByReference = new LongByReference();
        OpenVINO.getCore().ov_shape_create(axisLengths.size(), longByReference, shape);
    }

    /**
     * Constructs Shape from the initialized array.
     *
     * @param axisLengths Initialized array
     */
//    public Shape(long[] axisLengths) {
//        dims = new ArrayList<>();
//        dims.addAll(Arrays.asList(axisLengths));
//    }
    @Override
    public String toString() {
        return "Shape{" +
                ", rank=" + rank +
                ", dims=" + dims.stream().map(String::valueOf).collect(Collectors.joining(",")) +
                '}';
    }

    @Override
    protected void dispose() {
        if (!isNull()) {
            getVino().ov_shape_free(getValue());
            setVinoObj(null);
        }
    }

    public List<Long> getDims() {
        return dims;
    }

    public long getRank() {
        return rank;
    }

    public OvShape getShape() {
        return shape;
    }
}
