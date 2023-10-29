package org.openvino.java.preprocess;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.core.Model;

/**
 * Main class for adding pre- and post- processing steps to existing ov::Model
 * This is a helper class for writing easy pre- and post- processing operations on ov::Model object assuming that
 * any preprocess operation takes one input and produces one output.
 * For advanced preprocessing scenarios, like combining several functions with multiple inputs/outputs into one,
 * client's code can use transformation passes over ov::Model
 */
public class PrePostProcessor extends OpenVINOCls {

    /**
     * Default construction through Model.
     *
     * @param model model.
     */
    public PrePostProcessor(Model model) {
        super("PrePostProcessor");
        getVino().ov_preprocess_prepostprocessor_create(model.getValue(), getVinoObj());
    }

    @Override
    protected void dispose() {
        if (!isNull()) {
            getVino().ov_preprocess_prepostprocessor_free(getValue());
            setVinoObj(null);
        }
    }

    /**
     * Gets input pre-processing data structure. Should be used only if model/function has only one input
     * Using returned structure application's code is able to set user's tensor data (e.g layout), preprocess steps, target model's data
     *
     * @return Reference to model's input information structure
     */
    public InputInfo input() {
        PointerByReference input = new PointerByReference();
        getVino().ov_preprocess_prepostprocessor_get_input_info(getValue(), input);
        return new InputInfo(input);
    }

    /**
     * Gets input pre-processing data structure for input identified by it's tensor name
     *
     * @param tensorName Tensor name of specific input. Throws if tensor name is not associated with any input in a model
     * @return Reference to model's input information structure
     */
    public InputInfo input(String tensorName) {
        PointerByReference input = new PointerByReference();
        getVino().ov_preprocess_prepostprocessor_get_input_info_by_name(getValue(), tensorName, input);
        return new InputInfo(input);
    }

    /**
     * Gets input pre-processing data structure for input identified by it's order in a model
     *
     * @param tensorIndex Input index of specific input. Throws if input index is out of range for associated function.
     * @return Reference to model's input information structure
     */
    public InputInfo input(long tensorIndex) {
        PointerByReference input = new PointerByReference();
        getVino().ov_preprocess_prepostprocessor_get_input_info_by_index(getValue(), tensorIndex, input);
        return new InputInfo(input);
    }

    /**
     * Gets output post-processing data structure. Should be used only if model/function has only one output
     * Using returned structure application's code is able to set model's output data, post-process steps, user's tensor data (e.g layout)
     *
     * @return Reference to model's output information structure
     */
    public OutputInfo output() {
        PointerByReference output = new PointerByReference();
        getVino().ov_preprocess_prepostprocessor_get_output_info(getValue(), output);
        return new OutputInfo(output);
    }

    /**
     * Gets output post-processing data structure for output identified by it's tensor name
     *
     * @param tensorName Tensor name of specific output. Throws if tensor name is not associated with any input in a model
     * @return Reference to model's output information structure
     */
    public OutputInfo output(String tensorName) {
        PointerByReference output = new PointerByReference();
        getVino().ov_preprocess_prepostprocessor_get_output_info_by_name(getValue(), tensorName, output);
        return new OutputInfo(output);
    }

    /**
     * Gets output post-processing data structure for output identified by it's order in a model
     *
     * @param tensorIndex output index of specific output. Throws if output index is out of range for associated function
     * @return Reference to model's output information structure
     */
    public OutputInfo output(long tensorIndex) {
        PointerByReference output = new PointerByReference();
        getVino().ov_preprocess_prepostprocessor_get_output_info_by_index(getValue(), tensorIndex, output);
        return new OutputInfo(output);
    }

    /**
     * Adds pre/post-processing operations to function passed in constructor
     *
     * @return Function with added pre/post-processing operations
     */
    public Model build() {
        PointerByReference model = new PointerByReference();
        getVino().ov_preprocess_prepostprocessor_build(getValue(), model);
        return new Model(model);
    }
}
