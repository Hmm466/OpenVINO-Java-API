package org.openvino.java.core;

import org.openvino.java.base.OpenVINOCls;

public class Output extends OpenVINOCls {

    private Node mNode;

    private long index;

    public Output(Node node, long index) {
        super("Output-" + index,null);
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

    public Node getNode() {
        return mNode;
    }
}
