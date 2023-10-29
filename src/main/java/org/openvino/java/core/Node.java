package org.openvino.java.core;

import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.domain.OvPartialShape;
import org.openvino.java.domain.OvShape;
import org.openvino.java.enums.NodeType;

public class Node extends OpenVINOCls {

    private NodeType nodeType;

    public Node(PointerByReference node, NodeType type) {
        super("Node", node);
        this.nodeType = type;
    }

    public PartialShape getPartialShape() {
        OvPartialShape shape = new OvPartialShape();
        verifyExceptionStatus(getVino().ov_port_get_partial_shape(getValue(), shape));
        return new PartialShape(shape);
    }

    /**
     * Get the unique name of the node.
     *
     * @return A const reference to the node's unique name.
     */
    public String getName() {
        PointerByReference name = new PointerByReference();
        verifyExceptionStatus(getVino().ov_port_get_any_name(getValue(), name));
        return name.getValue().getString(0);
    }

    /**
     * Checks that there is exactly one output and returns its element type.
     *
     * @return Data type.
     */
    public int getElementType() {
        IntByReference type = new IntByReference();
        verifyExceptionStatus(getVino().ov_port_get_element_type(getValue(), type));
        return type.getValue();
    }

    /**
     * Get the shape.
     *
     * @return Returns the shape.
     */
    public Shape getShape() {
        OvShape shape = new OvShape();
        if (nodeType == NodeType.e_const) {
            verifyExceptionStatus(getVino().ov_const_port_get_shape(getValue(), shape));
        } else {
            verifyExceptionStatus(getVino().ov_port_get_shape(getValue(), shape));
        }
        return new Shape(shape);
    }

    @Override
    public void dispose() {
        if (getPointer() != null && getValue() != null) {
            if (nodeType == NodeType.e_const) {
                getVino().ov_output_const_port_free(getValue());
            } else {
                getVino().ov_output_port_free(getValue());
            }
            setVinoObj(null);
        }
    }

    public NodeType getNodeType() {
        return nodeType;
    }
}
