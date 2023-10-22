package org.openvino.java.core;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;

public class InferRequest extends OpenVINOCls {
    public InferRequest(PointerByReference request) {
        super("InferRequest",request);

    }

    /**
     * Gets an input tensor for inference.
     * @return The input tensor for the model. If model has several inputs, an exception is thrown.
     */
    public Tensor getInputTensor() {
        PointerByReference tensor = new PointerByReference();
        verifyExceptionStatus(getVino().ov_infer_request_get_input_tensor(getValue(),tensor));
        return new Tensor(tensor);
    }

    public Tensor getOutputTensor() {
        PointerByReference tensor = new PointerByReference();
        verifyExceptionStatus(getVino().ov_infer_request_get_output_tensor(getValue(),tensor));
        return new Tensor(tensor);
    }

    @Override
    protected void dispose() {

    }

    public void infer() {
        verifyExceptionStatus(getVino().ov_infer_request_infer(getValue()));
    }

    /**
     * Gets an input/output tensor for inference by tensor name.
     * @param tensorName Name of a tensor to get.
     * @return The tensor with name @p tensor_name. If the tensor is not found, an exception is thrown.
     */
    public Tensor getTensor(String tensorName) {
        PointerByReference tensor = new PointerByReference();
        verifyExceptionStatus(getVino().ov_infer_request_get_tensor(getValue(),tensorName,tensor));
        return new Tensor(tensor);
    }
}
