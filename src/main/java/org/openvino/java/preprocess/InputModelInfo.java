package org.openvino.java.preprocess;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.core.Layout;

/**
 * Information about model's input tensor. If all information is already included to loaded model, this info
 * may not be needed. However it can be set to specify additional information about model, like 'layout'.
 * Example of usage of model 'layout':
 *     Support model has input parameter with shape {1, 3, 224, 224} and user needs to resize input image to model's
 *     dimensions. It can be done like this
 */
public class InputModelInfo extends OpenVINOCls {

    /**
     * Default construction through InputModelInfo pointer.
     * @param ptr InputModelInfo pointer.
     */
    public InputModelInfo(PointerByReference ptr) {
        super("InputModelInfo", ptr);
    }

    @Override
    protected void dispose() {
        if (!isNull()) {
            getVino().ov_preprocess_input_model_info_free(getValue());
            setVinoObj(null);
        }
    }

    /**
     * Set layout for model's input tensor. This version allows chaining for Lvalue objects
     * @param layout Layout for model's input tensor.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner
     */
    public InputModelInfo setLayout(Layout layout) {
        getVino().ov_preprocess_input_model_info_set_layout(getValue(),layout.getValue());
        return this;
    }
}
