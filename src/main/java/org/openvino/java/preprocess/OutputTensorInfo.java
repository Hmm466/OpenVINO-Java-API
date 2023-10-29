package org.openvino.java.preprocess;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.enums.ElementType;

/**
 * Information about user's desired output tensor. By default, it will be initialized to same data
 * (type/shape/etc) as model's output parameter. User application can override particular parameters (like
 * 'element_type') according to application's data and specify appropriate conversions in post-processing steps
 */
public class OutputTensorInfo extends OpenVINOCls {

    /**
     * Default construction through OutputTensorInfo pointer.
     * @param ptr OutputTensorInfo pointer.
     */
    public OutputTensorInfo(PointerByReference ptr) {
        super("OutputTensorInfo", ptr);
    }

    @Override
    protected void dispose() {
        if (!isNull()) {
            getVino().ov_preprocess_output_tensor_info_free(getValue());
            setVinoObj(null);
        }
    }

    /**
     * Set element type for user's desired output tensor.
     * @param type Element type for user's output tensor.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public OutputTensorInfo setElementType(ElementType type) {
        getVino().ov_preprocess_output_set_element_type(getValue(), type.ordinal());
        return this;
    }
}
