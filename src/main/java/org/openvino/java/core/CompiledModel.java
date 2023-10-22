package org.openvino.java.core;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.enums.NodeType;

public class CompiledModel extends OpenVINOCls {

    public CompiledModel(PointerByReference model) {
        super("CompiledModel",model);
    }

    /**
     * Gets a single input of a compiled model.
     * The input is represented as an output of the ov::op::v0::Parameter operation.
     * The input contains information about input tensor such as tensor shape, names, and element type.
     * @return Compiled model input.
     */
    public Input input() {
        Node node = getInput();
        return new Input(node,0);
    }

    public Node getInput() {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_input(getValue(),node));
        return new Node(node, NodeType.e_const);
    }

    public Output output() {
        Node node = getOutput();
        return new Output(node,0);
    }

    public Node getOutput() {
        PointerByReference node = new PointerByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_output(getValue(),node));
        return new Node(node, NodeType.e_const);
    }

    /**
     * Creates an inference request object used to infer the compiled model.
     * The created request has allocated input and output tensors (which can be changed later).
     * @return InferRequest object
     */
    public InferRequest createInferRequest() {
        PointerByReference reference = new PointerByReference();
        verifyExceptionStatus(getVino().ov_compiled_model_create_infer_request(getValue(),reference));
        return new InferRequest(reference);
    }

    @Override
    protected void dispose() {

    }
}
