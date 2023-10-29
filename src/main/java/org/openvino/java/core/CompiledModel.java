package org.openvino.java.core;

import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.enums.NodeType;

import java.util.ArrayList;
import java.util.List;

/**
 * This class represents a compiled model.
 * A model is compiled by a specific device by applying multiple optimization
 * transformations, then mapping to compute kernels.
 */
public class CompiledModel extends OpenVINOCls {

    /**
     * Constructs CompiledModel from the initialized ptr.
     *
     * @param model
     */
    public CompiledModel(PointerByReference model) {
        super("CompiledModel", model);
    }


    /**
     * Get a const single input port of compiled_model, which only support single input compiled_model.
     *
     * @return The input port of compiled_model.
     */
    public Node getInput() {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_input(getValue(), node));
        return new Node(node, NodeType.e_const);
    }

    /**
     * Get a const input port of compiled_model by name.
     *
     * @param tensorName input tensor name
     * @return The input port of compiled_model.
     */
    public Node getInput(String tensorName) {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_input_by_name(getValue(), tensorName, node));
        return new Node(node, NodeType.e_const);
    }

    /**
     * Get a const input port of compiled_model by port index.
     *
     * @param index input tensor index.
     * @return The input port of compiled_model.
     */
    public Node getInput(long index) {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_input_by_index(getValue(), index, node));
        return new Node(node, NodeType.e_const);
    }

    /**
     * Get a const single output port of compiled_model, which only support single output model.
     *
     * @return The output port of compiled_model.
     */
    public Node getOutput() {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_output(getValue(), node));
        return new Node(node, NodeType.e_const);
    }

    /**
     * Get a const output port of compiled_model by name.
     *
     * @param tensorName output tensor name
     * @return The output port of compiled_model.
     */
    public Node getOutput(String tensorName) {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_output_by_name(getValue(), tensorName, node));
        return new Node(node, NodeType.e_const);
    }

    /**
     * Get a const output port of compiled_model by port index.
     *
     * @param index output tensor index.
     * @return The output port of compiled_model.
     */
    public Node getOutput(long index) {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_output_by_index(getValue(), index, node));
        return new Node(node, NodeType.e_const);
    }

    /**
     * Creates an inference request object used to infer the compiled model.
     * The created request has allocated input and output tensors (which can be changed later).
     *
     * @return InferRequest object
     */
    public InferRequest createInferRequest() {
        PointerByReference reference = new PointerByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_create_infer_request(getValue(), reference));
        return new InferRequest(reference);
    }

    /**
     * Get the input size of compiled_model.
     *
     * @return The input size of compiled_model.
     */
    public long getInputsSize() {
        LongByReference size = new LongByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_inputs_size(getValue(), size));
        return size.getValue();
    }

    /**
     * Get the output size of compiled_model.
     *
     * @return The output size.
     */
    public long getOutputsSize() {
        LongByReference size = new LongByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_outputs_size(getValue(), size));
        return size.getValue();
    }

    /**
     * Gets a single input of a compiled model.
     * The input is represented as an output of the ov::op::v0::Parameter operation.
     * The input contains information about input tensor such as tensor shape, names, and element type.
     *
     * @return Compiled model input.
     */
    public Input input() {
        Node node = getInput();
        return new Input(node, 0);
    }

    /**
     * Gets input of a compiled model identified by input index.
     * The input contains information about input tensor such as tensor shape, names, and element type.
     *
     * @param index Index of input.
     * @return Compiled model input.
     */
    public Input input(long index) {
        Node node = getInput(index);
        return new Input(node, index);
    }

    /**
     * Gets input of a compiled model identified by tensor_name.
     * The input contains information about input tensor such as tensor shape, names, and element type.
     *
     * @param tensorName tensor name.
     * @return Compiled model input.
     */
    public Input input(String tensorName) {
        Node node = getInput(tensorName);
        return new Input(node, 0);
    }

    /**
     * Gets a single output of a compiled model.
     * The output is represented as an output from the ov::op::v0::Result operation.
     * The output contains information about output tensor such as tensor shape, names, and element type.
     *
     * @return Compiled model output.
     */
    public Output output() {
        Node node = getOutput();
        return new Output(node, 0);
    }

    /**
     * Gets output of a compiled model identified by @p index.
     * The output contains information about output tensor such as tensor shape, names, and element type.
     *
     * @param index Index of output.
     * @return Compiled model output.
     */
    public Output output(long index) {
        Node node = getOutput(index);
        return new Output(node, 0);
    }

    /**
     * Gets output of a compiled model identified by @p tensor_name.
     * The output contains information about output tensor such as tensor shape, names, and element type.
     *
     * @param tensorName Output tensor name.
     * @return Compiled model output.
     */
    public Output output(String tensorName) {
        Node node = getOutput(tensorName);
        return new Output(node, 0);
    }

    /**
     * Gets all inputs of a compiled model.
     * Inputs are represented as a vector of outputs of the ov::op::v0::Parameter operations.
     * They contain information about input tensors such as tensor shape, names, and element type.
     *
     * @return List of model inputs.
     */
    public List<Input> inputs() {
        long inputSize = getInputsSize();
        List<Input> inputs = new ArrayList<>();
        for (long index = 0; index < inputSize; index++) {
            inputs.add(input(index));
        }
        return inputs;
    }

    /**
     * Get all outputs of a compiled model.
     * Outputs are represented as a vector of output from the ov::op::v0::Result operations.
     * Outputs contain information about output tensors such as tensor shape, names, and element type.
     *
     * @return List of model outputs.
     */
    public List<Output> outputs() {
        long outputsSize = getOutputsSize();
        List<Output> outputs = new ArrayList<>();
        for (long index = 0; index < outputsSize; index++) {
            outputs.add(output(index));
        }
        return outputs;
    }

    /**
     * Gets runtime model information from a device.
     * This object represents an internal device-specific model that is optimized for a particular
     * accelerator. It contains device-specific nodes, runtime information and can be used only
     * to understand how the source model is optimized and which kernels, element types, and layouts
     * are selected for optimal inference.
     *
     * @return
     */
    public Model getRuntimeModel() {
        PointerByReference model = new PointerByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_get_runtime_model(getValue(), model));
        return new Model(model);
    }

    /**
     * Exports the current compiled model to an output model_path.
     * The exported model can also be imported via the ov::Core::import_model method.
     *
     * @param modelPath Output path to store the model to.
     */
    public void exportModel(String modelPath) {
        verifyExceptionStatus(getVino().ov_compiled_model_export_model(getValue(), modelPath));
    }

    /**
     * Sets properties for the current compiled model.
     *
     * @param key  property name
     * @param name property value
     */
    public void setProperty(String key, String name) {
        verifyExceptionStatus(getVino().ov_compiled_model_set_property(getValue(), key, name));
    }

    /**
     * Gets properties for current compiled model
     * The method is responsible for extracting information that affects compiled model inference.
     * The list of supported configuration values can be extracted via CompiledModel::get_property
     * with the ov::supported_properties key, but some of these keys cannot be changed dynamically,
     * for example, ov::device::id cannot be changed if a compiled model has already been compiled
     * for a particular device.
     *
     * @param key Property key, can be found in openvino/runtime/properties.hpp.
     * @return Property value.
     */
    public String getProperty(String key) {
        PointerByReference value = new PointerByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_get_property(getValue(), key, value));
        return value.getValue().getString(0);
    }

    /**
     * Returns pointer to device-specific shared context on a remote accelerator device that was used to create this CompiledModel.
     *
     * @return A context.
     */
    public RemoteContext getContext() {
        PointerByReference pointer = new PointerByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_get_context(getValue(), pointer));
        return new RemoteContext(pointer);
    }

    /**
     * Release unmanaged resources
     */
    @Override
    protected void dispose() {
        if (!isNull()) {
            getVino().ov_core_free(getValue());
            setVinoObj(null);
        }
    }
}
