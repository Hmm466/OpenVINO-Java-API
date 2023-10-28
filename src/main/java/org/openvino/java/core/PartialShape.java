package org.openvino.java.core;

import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.domain.*;

import java.util.Arrays;
import java.util.List;

/**
 * Class representing a shape that may be partially or totally dynamic.
 * Dynamic rank. (Informal notation: `?`)
 * Static rank, but dynamic dimensions on some or all axes.
 * (Informal notation examples: `{1,2,?,4}`, `{?,?,?}`)
 * Static rank, and static dimensions on all axes.
 * (Informal notation examples: `{1,2,3,4}`, `{6}`, `{}`)
 */
public class PartialShape extends OpenVINOCls {

    OvPartialShape partialShape;

    /**
     * PartialShape rank.
     */
    private Dimension rank;

    /**
     * PartialShape dimensions.
     */
    private Dimension[] dimensions;

    public PartialShape(OvPartialShape partialShape) {
        super("PartialShape");
        Dimension rankTemp = new Dimension(partialShape.rank);
        if (!rankTemp.isDynamic()){
            rank = rankTemp;
            dimensions = new Dimension[(int)rank.getMin()];
            for (int i = 0; i < rank.getMin(); ++i) {
                Dimension dim = new Dimension(new OvDimension(partialShape.dims.getPointer(i)));
                dimensions[i] = dim;
            }
        }
        else {
            rank = rankTemp;
        }
    }

    /**
     * Constructing partial shape by dimensions.
     * @param dimensions The partial shape dimensions array.
     */
    public PartialShape(Dimension[] dimensions) {
        super("PartialShape");
        OvDimension[] ds = new OvDimension[dimensions.length];
        for (int i = 0; i < dimensions.length; ++i) {
            ds[i] = dimensions[i].getDimension();
        }
        partialShape = new OvPartialShape();
        verifyExceptionStatus(getVino().ov_partial_shape_create(dimensions.length,ds[0],partialShape));
        this.dimensions = dimensions;
        rank = new Dimension(dimensions.length, dimensions.length);
    }

    /**
     * Constructing partial shape by dimensions.
     * @param dimensions The partial shape dimensions list.
     */
    public PartialShape(List<Dimension> dimensions){
        this((Dimension[]) dimensions.toArray());
    }

    /**
     * Constructing dynamic partial shape by dimensions.
     * @param rank The partial shape rank.
     * @param dimensions The partial shape dimensions array.
     */
    public PartialShape(Dimension rank, Dimension[] dimensions) {
        super("PartialShape");
        OvDimension[] ds = new OvDimension[dimensions.length];
        for (int i = 0; i < dimensions.length; ++i)
        {
            ds[i] = dimensions[i].getDimension();
        }
        partialShape = new OvPartialShape();
//        rank.getDimension().
        OvRank rank1 = new OvRank();
//        shape.rank = rank.
        verifyExceptionStatus(getVino().ov_partial_shape_create_dynamic(rank1,ds[0],partialShape));
        this.dimensions = dimensions;
        this.rank = rank;
    }

    /**
     * Constructing dynamic partial shape by dimensions.
     * @param rank The partial shape rank.
     * @param dimensions The partial shape dimensions list.
     */
    public PartialShape(Dimension rank, List<Dimension> dimensions){
        this(rank, (Dimension[]) dimensions.toArray());
    }

    /**
     * Constructing static partial shape by dimensions.
     * @param rank The partial shape rank.
     * @param dimensions The partial shape dimensions array.
     */
    public PartialShape(long rank, long[] dimensions) {
        super("PartialShape");
        LongByReference longByReference = new LongByReference();
        partialShape = new OvPartialShape();
        verifyExceptionStatus(getVino().ov_partial_shape_create_static(rank, longByReference, partialShape));
        this.rank = new Dimension(rank);
        this.dimensions = new Dimension[dimensions.length];
        for (int i = 0; i < dimensions.length; ++i) {
            this.dimensions[i] = new Dimension(dimensions[i]);
        }
    }

    /**
     * Constructing static partial shape by dimensions.
     * @param rank The partial shape rank.
     * @param dimensions The partial shape dimensions list.
     */
    public PartialShape(long rank, List<Long> dimensions) {
        this(rank, listTo(dimensions));
    }

    public static long[] listTo(List<Long> list) {
        long[] result = new long[list.size()];
        for (int i = 0; i < list.size(); i++) {
            result[i] = list.get(i);
        }
        return result;
    }

    /**
     * Constructing static partial shape by shape.
     * @param shape The shape
     */
    public PartialShape(Shape shape) {
        super("PartialShape",null);
        partialShape = new OvPartialShape();
        verifyExceptionStatus(getVino().ov_shape_to_partial_shape(shape.getOvShapeT(), partialShape));
        this.rank = new Dimension(shape.getRank());
        dimensions = new Dimension[(int) shape.getRank()];
        for (int i = 0; i < dimensions.length; ++i) {
            this.dimensions[i] = new Dimension(shape.getDims().get(i));
        }
    }

    @Override
    protected void dispose() {
        if (partialShape != null) {
            getVino().ov_partial_shape_free(partialShape);
            partialShape = null;
        }
    }

    /// <summary>
    /// Get ov_partial_shape
    /// </summary>
    /// <returns>return ov_partial_shape.</returns>

    /**
     * Get ov_partial_shape
     * @return return ov_partial_shape.
     */
    public OvPartialShape getPartialShape() {
        OvPartialShape partialShape = new OvPartialShape();
        partialShape.rank = rank.getDimension();
        PointerByReference data = new PointerByReference();
        for (int i = 0; i < rank.getMax(); ++i) {
            data.getPointer().setPointer(i,dimensions[i].getDimension().getPointer());
        }
        partialShape.dims = data.getPointer();
        return partialShape;
    }

    /**
     * Get dimensions.
     * @return Dimension[
     */
    public Dimension[] get_dimensions() {
        return dimensions;
    }

    /**
     * Convert partial shape without dynamic data to a static shape.
     * @return The shape.
     */
    public Shape toShape() {
        OvShape shape = new OvShape();
        verifyExceptionStatus(getVino().ov_partial_shape_to_shape(getPartialShape(),shape));
        return new Shape(shape);
    }

    /**
     * Check if this shape is static.
     * A shape is considered static if it has static rank, and all dimensions of the shape are static.
     * @return `true` if this shape is static, else `false`.
     */
    public boolean isStatic() {
        return !isDynamic();
    }

    /**
     * Check if this shape is dynamic.
     * A shape is considered static if it has static rank, and all dimensions of the shape
     * @return `false` if this shape is static, else `true`.
     */
    public boolean isDynamic() {
        return getVino().ov_partial_shape_is_dynamic(getPartialShape());
    }

    /**
     * Get partial shape string.
     * @return
     */
    @Override
    public String toString() {
        String  s = "Shape : {";
        if (rank.isDynamic()) {
            s += "?";
        }
        else {
            for (int i = 0; i < rank.getMax(); ++i) {
                if (dimensions[i].isDynamic()) {
                    s += "?,";
                }
                else {
                    s += dimensions[i].getDimension().max + ",";
                }
            }
        }
        s = s.substring(0, s.length() - 1);
        s += "}";
        return s;
    }
}
