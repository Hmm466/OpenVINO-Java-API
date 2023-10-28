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

    /**
     *
     */
    private long index;

    /**
     * Constructs a input.
     * @param node The node for the input handle.
     * @param index The index of the input.
     */
    public Input(Node node,long index) {
        super("Input-" + index,null);
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
