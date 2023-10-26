package org.openvino.java.core;

import org.openvino.java.base.OpenVINOCls;

/**
 * Class holding preprocessing information for one input
 * From preprocessing pipeline perspective, each input can be represented as:
 *  - User's input parameter info (InputInfo::tensor)
 *  - Preprocessing steps applied to user's input (InputInfo::preprocess)
 *  - Model's input info, which is a final input's info after preprocessing (InputInfo::model)
 */
public class Input extends OpenVINOCls {

    /**
     * InputInfo class pointer.
     */
    private Node mNode;

    private long index;

    public Input(Node node,long index) {
        super("Input-" + index,null);
        this.mNode = node;
        this.index = index;
    }

    public String getAnyName() {
        return mNode.getName();
    }

    public int getElementType() {
        return mNode.getElementType();
    }

    public Shape getShape() {
        return mNode.getShape();
    }

    @Override
    protected void dispose() {

    }
}
