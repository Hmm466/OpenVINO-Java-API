package org.openvino.java.preprocess;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;

/**
 * Class holding preprocessing information for one input
 * From preprocessing pipeline perspective, each input can be represented as:
 * - User's input parameter info (InputInfo::tensor)
 * - Preprocessing steps applied to user's input (InputInfo::preprocess)
 * - Model's input info, which is a final input's info after preprocessing (InputInfo::model)
 */
public class InputInfo extends OpenVINOCls {

    /**
     * Default construction through InputInfo pointer.
     *
     * @param ptr InputInfo pointer.
     */
    public InputInfo(PointerByReference ptr) {
        super("InputInfo", ptr);
    }

    @Override
    protected void dispose() {
        if (!isNull()) {
            getVino().ov_preprocess_input_info_free(getValue());
            setVinoObj(null);
        }
    }

    /**
     * Get current input tensor information with ability to change specific data
     *
     * @return Reference to current input tensor structure
     */
    public InputTensorInfo tensor() {
        PointerByReference info = new PointerByReference();
        getVino().ov_preprocess_input_info_get_tensor_info(getValue(), info);
        return new InputTensorInfo(info);
    }

    /**
     * Get current input preprocess information with ability to add more preprocessing steps
     *
     * @return Reference to current preprocess steps structure.
     */
    public PreProcessSteps preprocess() {
        PointerByReference ptr = new PointerByReference();
        getVino().ov_preprocess_input_info_get_preprocess_steps(getValue(), ptr);
        return new PreProcessSteps(ptr);
    }

    /**
     * Get current input model information with ability to change original model's input data
     *
     * @return Reference to current model's input information structure.
     */
    public InputModelInfo model() {
        PointerByReference model = new PointerByReference();
        getVino().ov_preprocess_input_info_get_model_info(getValue(), model);
        return new InputModelInfo(model);
    }
}
