package org.openvino.java.preprocess;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;

/**
 * Class holding postprocessing information for one output
 * From postprocessing pipeline perspective, each output can be represented as:
 * - Model's output info,  (OutputInfo::model)
 * - Postprocessing steps applied to user's input (OutputInfo::postprocess)
 * - User's desired output parameter information, which is a final one after preprocessing (OutputInfo::tensor)
 */
public class OutputInfo extends OpenVINOCls {

    /**
     * Default construction through OutputInfo pointer.
     *
     * @param ptr OutputInfo pointer.
     */
    public OutputInfo(PointerByReference ptr) {
        super("OutputInfo", ptr);
    }

    @Override
    protected void dispose() {
        if (!isNull()) {
            getVino().ov_preprocess_output_info_free(getValue());
            setVinoObj(null);
        }
    }

    /**
     * Get current output tensor information with ability to change specific data
     *
     * @return Reference to current output tensor structure
     */
    public OutputTensorInfo tensor() {
        PointerByReference tensor = new PointerByReference();
        getVino().ov_preprocess_output_info_get_tensor_info(getValue(), tensor);
        return new OutputTensorInfo(tensor);
    }
}
