package org.openvino.java.core;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;
import org.opencv.core.Mat;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.domain.OvShape;
import org.openvino.java.enums.ElementType;

/**
 * Tensor API holding host memory.
 * It can throw exceptions safely for the application, where it is properly handled.
 */
public class Tensor extends OpenVINOCls {

    /**
     * Constructs Tensor from the initialized pointer.
     * @param tensor Tensor pointer.
     */
    public Tensor(PointerByReference tensor) {
        super("Tensor",tensor);
    }

    /**
     * Constructs Tensor using element type ,shape and image data.
     * @param type Tensor element type<
     * @param shape Tensor shape
     * @param mat Image data
     */
    public Tensor(ElementType type, Shape shape, Mat mat) {
        super("Tensor");
        Pointer pointer = new Pointer(mat.dataAddr());
        verifyExceptionStatus(getVino().ov_tensor_create_from_host_ptr(type.hashCode(),shape.getShape(),pointer,getVinoObj()));
    }

    /**
     * Constructs Tensor using element type and shape. Wraps allocated host memory.Does not perform memory allocation internally.
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param host_ptr Pointer to pre-allocated host memory
     */
    public Tensor(ElementType type, Shape shape, Pointer host_ptr) {
        super("Tensor");
        verifyExceptionStatus(getVino().ov_tensor_create_from_host_ptr(type.ordinal(),shape.getShape(),host_ptr,getVinoObj()));
    }

    /**
     * Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
     * @param type Tensor element type
     * @param shape Tensor shape
     */
    public Tensor(ElementType type, Shape shape) {
        super("Tensor");
        verifyExceptionStatus(getVino().ov_tensor_create(type.ordinal(),shape.getShape(),getVinoObj()));
    }

    /**
     * Default copy constructor
     * @param tensor other Tensor object
     */
    public Tensor(Tensor tensor) {
        super("Tensor");
        verifyExceptionStatus(getVino().ov_tensor_create_from_host_ptr(tensor.getElementType().ordinal(),tensor.getShape().getShape(),tensor.getData().getPointer(),getVinoObj()));
    }

    /**
     * Set new shape for tensor, deallocate/allocate if new total size is bigger than previous one.
     * @param shape A new shape
     */
    public void setShape(Shape shape) {
        verifyExceptionStatus(getVino().ov_tensor_set_shape(getValue(),shape.getShape()));
    }

    /**
     * Get tensor shape
     * @return A tensor shape
     */
    public Shape getShape() {
        OvShape shapeT = new OvShape();
        verifyExceptionStatus(getVino().ov_tensor_get_shape(getValue(),shapeT));
        return new Shape(shapeT);
    }

    /**
     * Get tensor element type
     * @return A tensor element type
     */
    public ElementType getElementType() {
        IntByReference intByReference = new IntByReference();
        verifyExceptionStatus(getVino().ov_tensor_get_element_type(getValue(),intByReference));
        return ElementType.values()[intByReference.getValue()];
    }

    /**
     * Returns the total number of elements (a product of all the dims or 1 for scalar).
     * @return The total number of elements.
     */
    public long getSize() {
        LongByReference longByReference = new LongByReference();
        verifyExceptionStatus(getVino().ov_tensor_get_size(getValue(),longByReference));
        return longByReference.getValue();
    }

    /**
     * Returns the size of the current Tensor in bytes.
     * @return Tensor's size in bytes
     */
    public long getByteSize() {
        LongByReference longByReference = new LongByReference();
        verifyExceptionStatus(getVino().ov_tensor_get_byte_size(getValue(),longByReference));
        return longByReference.getValue();
    }

    /**
     * Copy tensor, destination tensor should have the same element type and shape
     * @param dst
     */
    public <T> void copyTo(Class cls,Tensor dst) {
        long length = getSize();
        T[] data = this.getData(cls,(int)length);
        dst.setData(data);
    }

    /**
     * Provides an access to the underlaying host memory.
     * @return A host pointer to tensor memory.
     */
    public PointerByReference data() {
        PointerByReference data = new PointerByReference();
        verifyExceptionStatus(getVino().ov_tensor_data(getValue(),data));
        return data;
    }

    /**
     * Load the specified type of data into the underlying host memory.
     * @param data Data to be loaded.
     * @param <T> data type
     */
    public <T> void setData(T data) {
        PointerByReference pointerByReference = new PointerByReference(new Memory(((float[]) data).length));
        verifyExceptionStatus(getVino().ov_tensor_data(getValue(),pointerByReference));
        if (data.getClass().getName().equals("[F")) {
            pointerByReference.getValue().write(0,(float[]) data,0,((float[])data).length);
        } else if (data.getClass().getName().equals("[B")) {
            pointerByReference.getValue().write(0,(byte[]) data,0,((byte[])data).length);
        } if (data.getClass().getName().equals("[I")) {
            pointerByReference.getValue().write(0,(int[]) data,0,((int[])data).length);
        } else if (data.getClass().getName().equals("[D")) {
            pointerByReference.getValue().write(0,(double[]) data,0,((double[])data).length);
        }
    }


    /**
     * Read data of the specified type from the underlying host memory.
     * @param cls Type of data read.
     * @param outputLength The length of the read data.
     * @param <T> Type of data read.
     * @return
     */
    public <T> T getData(Class cls, int outputLength) {
        PointerByReference data = getData();
        if (cls.getName().equals("[F")) {
            return (T) data.getValue().getFloatArray(0, outputLength);
        } else if (cls.getName().equals("[S")) {
            return (T) data.getValue().getStringArray(0, outputLength);
        } else if (cls.getName().equals("[B")) {
            return (T) data.getValue().getByteArray(0, outputLength);
        } else if (cls.getName().equals("[C")) {
            return (T) data.getValue().getCharArray(0, outputLength);
        } else if (cls.getName().equals("[D")) {
            return (T) data.getValue().getDoubleArray(0, outputLength);
        } else if (cls.getName().equals("[S")) {
            return (T) data.getValue().getShortArray(0, outputLength);
        } else if (cls.getName().equals("[L")) {
            return (T) data.getValue().getLongArray(0, outputLength);
        } else if (cls.getName().equals("[I")) {
            return (T) data.getValue().getIntArray(0, outputLength);
        } else {
            return null;
        }
    }

    /**
     * Provides an access to the underlaying host memory.
     * @return A host pointer to tensor memory.
     */
    private PointerByReference getData() {
        PointerByReference data = new PointerByReference();
        verifyExceptionStatus(getVino().ov_tensor_data(getValue(),data));
        return data;
    }

    @Override
    protected void dispose() {
        if (!isNull()) {
            getVino().ov_tensor_free(getValue());
            setVinoObj(null);
        }
    }
}
