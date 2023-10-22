package org.openvino.java.core;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.enums.NodeType;

public class Model extends OpenVINOCls {

    public Model(PointerByReference model) {
        super("model");
        setVinoObj(model);
    }

    /**
     * Gets the friendly name for a model.
     * @return The friendly name for a model.
     */
    public String getFriendlyName() {
        PointerByReference name = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_get_friendly_name(getValue(),name));
        return name.getValue().getString(0);
    }

    /**
     * Get a const input port of model by port index.
     * @param index input tensor index.
     * @return The const input port of model.
     */
    public Node getConstInput(long index) {
        PointerByReference input = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_const_input_by_index(getValue(),index,input));
        return new Node(input, NodeType.e_const);
    }

    public Node getConstOutput(long index) {
        PointerByReference output = new PointerByReference();
        verifyExceptionStatus(getVino().ov_model_const_output_by_index(getValue(),index,output));
        return new Node(output, NodeType.e_const);
    }

    @Override
    protected void dispose() {

    }
}
