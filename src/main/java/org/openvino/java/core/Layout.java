package org.openvino.java.core;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;

/**
 * ov::Layout represents the text information of tensor's dimensions/axes. E.g. layout `NCHW` means that 4D
 * tensor `{-1, 3, 480, 640}` will have:
 *  - 0: `N = -1`: batch dimension is dynamic
 *  - 1: `C = 3`: number of channels is '3'
 *  - 2: `H = 480`: image height is 480
 *  - 3: `W = 640`: image width is 640
 * `ov::Layout` can be specified for:
 *      - Preprocessing purposes. E.g.
 *      - To apply normalization (means/scales) it is usually required to set 'C' dimension in a layout.
 *      - To resize the image to specified width/height it is needed to set 'H' and 'W' dimensions in a layout
 *      - To transpose image - source and target layout can be set (see
 *      `ov::preprocess::PreProcessSteps::convert_layout`)
 *      - To set/get model's batch (see `ov::get_batch`/`ov::set_batch') it is required in general to specify 'N' dimension
 *  in layout for appropriate inputs
 */
public class Layout extends OpenVINOCls {

    /**
     * Constructs a Layout with static or dynamic layout information based on string representation.
     * The string used to construct Layout from.
     * The string representation can be in the following form:
     *      - can define order and meaning for dimensions "NCHW"
     *      - partial layout specialization:
     *      - "NC?" defines 3 dimensional layout, first two NC, 3rd one is not defined
     *      - "N...C" defines layout with dynamic rank where 1st dimension is N, last one is C
     *      - "NC..." defines layout with dynamic rank where first two are NC, others are not
     *    defined
     *      - only order of dimensions "adbc" (0312)
     *      - Advanced syntax can be used for multi-character names like "[N,C,H,W,...,CustomName]"
     * @param layoutDesc
     */
    public Layout(String layoutDesc) {
        super("Layout");
        verifyExceptionStatus(getVino().ov_layout_create(layoutDesc,getVinoObj()));
    }

    @Override
    protected void dispose() {
        if (!isNull()) {
            getVino().ov_layout_free(getValue());
            setVinoObj(null);
        }
    }

    /**
     * String representation of Layout.
     * @return String representation of Layout.
     */
    @Override
    public String toString() {
        return getVino().ov_layout_to_string(getValue());
    }
}
