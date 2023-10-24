package org.openvino.java.core;

import com.sun.jna.Memory;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.core.structures.OvShape;
import org.openvino.java.utils.Console;

public class Tensor extends OpenVINOCls {

    public Tensor(PointerByReference tensor) {
        super("Tensor",tensor);
    }

    public int getElementType() {
        return -1;
    }

    public Shape getShape() {
        OvShape shapeT = new OvShape();
        verifyExceptionStatus(getVino().ov_tensor_get_shape(getValue(),shapeT));
        return new Shape(shapeT);
    }

    public long getSize() {
        LongByReference longByReference = new LongByReference();
        verifyExceptionStatus(getVino().ov_tensor_get_size(getValue(),longByReference));
        return longByReference.getValue();
    }

    public <T> void setData(T data) {
        PointerByReference pointerByReference = new PointerByReference(new Memory(((float[]) data).length));
        verifyExceptionStatus(getVino().ov_tensor_data(getValue(),pointerByReference));
        if (data.getClass().getName().equals("[F")) {
            pointerByReference.getValue().write(0,(float[]) data,0,((float[])data).length);
        }
    }


    public <T> T getData(Class cls, int outputLength) {
        PointerByReference data = getData();
        if (cls.getName().equals("[F")) {
            return (T) data.getValue().getFloatArray(0, outputLength);
        } else {
            return null;
        }
    }

    private PointerByReference getData() {
        PointerByReference data = new PointerByReference();
        verifyExceptionStatus(getVino().ov_tensor_data(getValue(),data));
        return data;
    }

    @Override
    protected void dispose() {

    }
}
