package org.openvino.java.core;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.domain.OvPartialShape;
import org.openvino.java.enums.NodeType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A user-defined model
 */
public class Model extends OpenVINOCls {

    /**
     * @param model Model pointer.
     */
    public Model(PointerByReference model) {
        super("model");
        setVinoObj(model);
    }

    /**
     * Gets the friendly name for a model.
     *
     * @return The friendly name for a model.
     */
    public String getFriendlyName() {
        PointerByReference name = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_get_friendly_name(getValue(), name));
        return name.getValue().getString(0);
    }

    /**
     * Get single input port of model, which only support single input model.
     *
     * @return The input port of model.
     */
    public Node getInput() {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_input(getValue(), node));
        return new Node(node, NodeType.e_nomal);
    }

    /**
     * Get an input port of model by name.
     *
     * @param tensorName input tensor name (string).
     * @return The input port of model.
     */
    public Node getInput(String tensorName) {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_input_by_name(getValue(), tensorName, node));
        return new Node(node, NodeType.e_nomal);
    }

    /**
     * Get an input port of model by port index.
     *
     * @param index input tensor index.
     * @return The input port of model.
     */
    public Node getInput(long index) {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_input_by_index(getValue(), index, node));
        return new Node(node, NodeType.e_nomal);
    }

    /**
     * Get an single output port of model, which only support single output model.
     *
     * @return The output port of model.
     */
    public Node getOutput() {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_output(getValue(), node));
        return new Node(node, NodeType.e_nomal);
    }

    /**
     * Get an output port of model by name.
     *
     * @param tensorName output tensor name (string).
     * @return The output port of model.
     */
    public Node getOutput(String tensorName) {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_output_by_name(getValue(), tensorName, node));
        return new Node(node, NodeType.e_nomal);
    }

    /**
     * Get an output port of model by port index.
     *
     * @param index input tensor index.
     * @return The output port of model.
     */
    public Node getOutput(long index) {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_output_by_index(getValue(), index, node));
        return new Node(node, NodeType.e_nomal);
    }

    /**
     * Get a const single input port of model, which only support single input model.
     *
     * @return The const input port of model.
     */
    public Node getConstInput() {
        PointerByReference input = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_const_input(getValue(), input));
        return new Node(input, NodeType.e_const);
    }

    /**
     * Get a const input port of model by name.
     *
     * @param tensorName input tensor name (string).
     * @return The const input port of model.
     */
    public Node getConstInput(String tensorName) {
        PointerByReference input = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_const_input_by_name(getValue(), tensorName, input));
        return new Node(input, NodeType.e_const);
    }

    /**
     * Get a const input port of model by port index.
     *
     * @param index input tensor index.
     * @return The const input port of model.
     */
    public Node getConstInput(long index) {
        PointerByReference input = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_const_input_by_index(getValue(), index, input));
        return new Node(input, NodeType.e_const);
    }

    /**
     * Get a single const output port of model, which only support single output model..
     *
     * @return The const output port of model.
     */
    public Node getConstOutput() {
        PointerByReference output = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_const_output(getValue(), output));
        return new Node(output, NodeType.e_const);
    }

    /**
     * Get a const output port of model by name.
     *
     * @param tensorName output tensor name (string).
     * @return The const output port of model.
     */
    public Node getConstOutput(String tensorName) {
        PointerByReference output = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_const_output_by_name(getValue(), tensorName, output));
        return new Node(output, NodeType.e_const);
    }

    /**
     * @param index
     * @return
     */
    public Node getConstOutput(long index) {
        PointerByReference output = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_const_output_by_index(getValue(), index, output));
        return new Node(output, NodeType.e_const);
    }

    /**
     * Get single input of model, which only support single input model.
     *
     * @return The input of model.
     */
    public Input input() {
        Node node = getInput();
        return new Input(node, 0);
    }

    /**
     * Get an input of model by port index.
     *
     * @param index input tensor index.
     * @return The input of model.
     */
    public Input input(long index) {
        Node node = getInput(index);
        return new Input(node, index);
    }

    /**
     * Get an input of model by name.
     *
     * @param tensorName input tensor name (string).
     * @return The input of model.
     */
    public Input input(String tensorName) {
        Node node = getInput(tensorName);
        return new Input(node, 0);
    }

    /**
     * Get single const input of model, which only support single input model.
     *
     * @return The const input of model.
     */
    public Input constInput() {
        Node node = getConstInput();
        return new Input(node, 0);
    }

    /**
     * Get an const input of model by port index.
     *
     * @param index input tensor index.
     * @return The const input of model.
     */
    public Input constInput(long index) {
        Node node = getConstInput(index);
        return new Input(node, index);
    }

    /**
     * Get an const input of model by name.
     *
     * @param tensorName input tensor name (string).
     * @return The const input of model.
     */
    public Input constInput(String tensorName) {
        Node node = getConstInput(tensorName);
        return new Input(node, 0);
    }

    /**
     * Get single output of model, which only support single output model.
     *
     * @return The output of model.
     */
    public Output output() {
        Node node = getOutput();
        return new Output(node, 0);
    }

    /**
     * Get an output of model by port index.
     *
     * @param index output tensor index.
     * @return The output of model.
     */
    public Output output(long index) {
        Node node = getOutput(index);
        return new Output(node, index);
    }

    /**
     * Get an output of model by name.
     *
     * @param tensorName output tensor name (string).
     * @return The output of model.
     */
    public Output output(String tensorName) {
        Node node = getOutput(tensorName);
        return new Output(node, 0);
    }

    /**
     * Get single const output of model, which only support single output model.
     *
     * @return The const output of model.
     */
    public Output constOutput() {
        Node node = getConstOutput();
        return new Output(node, 0);
    }

    /**
     * Get an const output of model by port index.
     *
     * @param index output tensor index.
     * @return The const output of model.
     */
    public Output constOutput(long index) {
        Node node = getConstOutput(index);
        return new Output(node, index);
    }

    /**
     * Get an const output of model by name.
     *
     * @param tensorName output tensor name (string).
     * @return The const output of model.
     */
    public Output constOutput(String tensorName) {
        Node node = getConstOutput(tensorName);
        return new Output(node, 0);
    }

    /**
     * Get the input size of model.
     *
     * @return The input size.
     */
    public long getInputsSize() {
        LongByReference size = new LongByReference();
        verifyExceptionStatus(getVino().ov_model_inputs_size(getValue(), size));
        return size.getValue();
    }

    /**
     * Get the output size of model.
     *
     * @return The output size.
     */
    public long getOutputsSize() {
        LongByReference size = new LongByReference();
        verifyExceptionStatus(getVino().ov_model_outputs_size(getValue(), size));
        return size.getValue();
    }

    /**
     * Get all input of model.
     *
     * @return All input of model.
     */
    public List<Input> inputs() {
        long size = getInputsSize();
        List<Input> inputs = new ArrayList<>();
        for (long index = 0; index < size; ++index) {
            inputs.add(input(index));
        }
        return inputs;
    }

    /**
     * Get all output of model
     *
     * @return All output of model
     */
    public List<Output> outputs() {
        long outputsSize = getOutputsSize();
        List<Output> outputs = new ArrayList<>();
        for (long index = 0; index < outputsSize; ++index) {
            outputs.add(output(index));
        }
        return outputs;
    }

    /**
     * Get all const input of model.
     *
     * @return All input of model.
     */
    public List<Input> constInputs() {
        long inputsSize = getInputsSize();
        List<Input> inputs = new ArrayList<>();
        for (long index = 0; index < inputsSize; ++index) {
            inputs.add(constInput(index));
        }
        return inputs;
    }

    /**
     * Get all const output of model
     *
     * @return All output of model
     */
    public List<Output> constOutputs() {
        long output_size = getOutputsSize();
        List<Output> outputs = new ArrayList<>();
        for (long index = 0; index < output_size; ++index) {
            outputs.add(constOutput(index));
        }
        return outputs;
    }

    /**
     * The ops defined in the model is dynamic shape.
     *
     * @return true if any of the ops defined in the model is dynamic shape.
     */
    public boolean isDynamic() {
        return getVino().ov_model_is_dynamic(getValue());
    }

    /**
     * Do reshape in model with partial shape for a specified name.
     *
     * @param partialShapeMap The list of input tensor names and PartialShape.
     */
    public void reshapeByStringMap(Map<String, PartialShape> partialShapeMap) {
        if (1 != partialShapeMap.size()) {
            String[] tensor_names_ptr = new String[partialShapeMap.size()];
            OvPartialShape[] shapes = new OvPartialShape[partialShapeMap.size()];
            int i = 0;
            for (String key : partialShapeMap.keySet()) {
                tensor_names_ptr[i] = key;
                shapes[i] = partialShapeMap.get(key).getPartialShape();
                i++;
            }
            verifyExceptionStatus(getVino().ov_model_reshape(getValue(), tensor_names_ptr, shapes[0], partialShapeMap.size()));
        } else {
            for (String key : partialShapeMap.keySet()) {
                verifyExceptionStatus(getVino().ov_model_reshape_input_by_name(getValue(), key, partialShapeMap.get(key).getPartialShape()));
            }
        }
    }

    /**
     * Do reshape in model for one node(port 0).
     *
     * @param partialShape A PartialShape.
     */
    public void reshape(PartialShape partialShape) {
        verifyExceptionStatus(getVino().ov_model_reshape_single_input(getValue(), partialShape.getPartialShape()));
    }

    /**
     * Do reshape in model with a list of (port id, partial shape).
     *
     * @param partialShapes The list of input port id and PartialShape.
     */
    public void reshapeByLongMap(Map<Long, PartialShape> partialShapes) {
        long[] indexs = new long[partialShapes.size()];
        OvPartialShape[] shapes = new OvPartialShape[partialShapes.size()];
        int i = 0;
        for (long key : partialShapes.keySet()) {
            indexs[i] = key;
            shapes[i] = partialShapes.get(key).getPartialShape();
            i++;
        }
        verifyExceptionStatus(getVino().ov_model_reshape_by_port_indexes(getValue(), indexs[0], shapes[0], partialShapes.size()));
    }

    /**
     * Do reshape in model with a list of (ov_output_port_t, partial shape).
     *
     * @param partialShapeMap The list of input node and PartialShape.
     */
    public void reshapeByNodeMap(Map<Node, PartialShape> partialShapeMap) {
        OvPartialShape[] shapes = new OvPartialShape[partialShapeMap.size()];
        int i = 0;
        Pointer pointer = new Memory(partialShapeMap.size());
        for (Node node : partialShapeMap.keySet()) {
            pointer.setPointer(i, node.getPointer());
            i++;
        }
        verifyExceptionStatus(getVino().ov_model_reshape_by_ports(getValue(), new PointerByReference(pointer), shapes[0], partialShapeMap.size()));
    }

    @Override
    protected void dispose() {
        if (!isNull()) {
            getVino().ov_model_free(getValue());
            setVinoObj(null);
        }
    }
}
