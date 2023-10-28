package org.openvino.java.core;

import org.openvino.java.base.OpenVINOCls;

/**
 *  A handle for one of a node's outputs.
 */
public class Output extends OpenVINOCls {

    /**
     *
     */
    private Node mNode;

    /**
     *
     */
    private long index;

    /**
     * Constructs a Output.
     * @param node The node for the output handle.
     * @param index The index of the output.
     */
    public Output(Node node, long index) {
        super("Output-" + index,null);
        this.mNode = node;
        this.index = index;
    }

    /**
     * Any tensor names associated with this input
     * @return tensor names<
     */
    public String getAnyName() {
        return mNode.getName();
    }

    /**
     * The element type of the input referred to by this input handle.
     * @return The element type of the input.
     */
    public int getElementType() {
        return mNode.getElementType();
    }

    /**
     * The shape of the input referred to by this input handle.
     * @return The shape of the input .
     */
    public Shape getShape() {
        return mNode.getShape();
    }

    /**
     * Get the node referred to by this input handle.
     * @return The ouput node
     */
    public Node getNode() {
        return mNode;
    }

    /**
     * The index of the input referred to by this input handle.
     * @return The index of the input.
     */
    public long getIndex() {
        return index;
    }

    /**
     * The partial shape of the input referred to by this input handle.
     * @return The partial shape of the input
     */
    public PartialShape getPartialShape() {
        return mNode.getPartialShape();
    }

    @Override
    protected void dispose() {
        mNode.dispose();
    }
}
