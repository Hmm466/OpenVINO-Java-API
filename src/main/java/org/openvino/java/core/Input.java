package org.openvino.java.core;

import org.openvino.java.base.OpenVINOCls;

public class Input extends OpenVINOCls {

    private Node mNode;

    private long index;

    public Input(Node node,int index) {
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
