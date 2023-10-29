package org.openvino.java.preprocess;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.core.Layout;
import org.openvino.java.enums.ColorFormat;
import org.openvino.java.enums.ElementType;
import org.openvino.java.enums.ExceptionStatus;
import org.openvino.java.enums.ResizeAlgorithm;
import org.openvino.java.utils.ListUtils;

import java.util.List;

/**
 * Preprocessing steps. Each step typically intends adding of some operation to input parameter
 * User application can specify sequence of preprocessing steps in a builder-like manner
 */
public class PreProcessSteps extends OpenVINOCls {

    /**
     * Default construction through PreProcessSteps pointer.
     *
     * @param ptr PreProcessSteps pointer.
     */
    public PreProcessSteps(PointerByReference ptr) {
        super("PreProcessSteps", ptr);
    }

    @Override
    protected void dispose() {
        if (!isNull()) {
            getVino().ov_preprocess_preprocess_steps_free(getValue());
            setVinoObj(null);
        }
    }

    /**
     * Add resize operation to model's dimensions.
     *
     * @param resize resize algorithm.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps resize(ResizeAlgorithm resize) {
        int status = getVino().ov_preprocess_preprocess_steps_resize(getValue(), resize.ordinal());
        if (status != ExceptionStatus.OK.getCode()) {

        }
        return this;
    }

    /**
     * Add scale preprocess operation. Divide each element of input by specified value.
     *
     * @param value Scaling value.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps scale(float value) {
        int status = getVino().ov_preprocess_preprocess_steps_scale(getValue(), value);
        if (status != 0) {

        }
        return this;
    }

    /**
     * Add mean preprocess operation. Subtract specified value from each element of input.
     *
     * @param value Value to subtract from each element.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps mean(float value) {
        int status = getVino().ov_preprocess_preprocess_steps_mean(getValue(), value);
        if (status != 0) {

        }
        return this;
    }

    /**
     * Crop input tensor between begin and end coordinates. Under the hood, inserts `opset8::Slice` operation to
     * execution graph. It is recommended to use to together with `ov::preprocess::InputTensorInfo::set_shape` to set
     * original input shape before cropping
     *
     * @param begin Begin indexes for input tensor cropping. Negative values represent counting elements from the end of input tensor
     * @param end   End indexes for input tensor cropping. End indexes are exclusive, which means values including end edge are not included in the output slice. Negative values represent counting elements from the end of input tensor
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps crop(int[] begin, int[] end) {
        Pointer beginPoint = new Memory(begin.length);
        beginPoint.write(0, begin, 0, begin.length);
        Pointer endPoint = new Memory(end.length);
        endPoint.write(0, end, 0, end.length);
        int status = getVino().ov_preprocess_preprocess_steps_crop(getValue(), beginPoint, begin.length, endPoint, end.length);
        if (status != 0) {

        }
        return this;
    }

    /**
     * Crop input tensor between begin and end coordinates. Under the hood, inserts `opset8::Slice` operation to
     * execution graph. It is recommended to use to together with `ov::preprocess::InputTensorInfo::set_shape` to set
     * original input shape before cropping
     *
     * @param begin Begin indexes for input tensor cropping. Negative values represent counting elements from the end of input tensor
     * @param end   End indexes for input tensor cropping. End indexes are exclusive, which means values including end edge are not included in the output slice. Negative values represent counting elements from the end of input tensor
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps crop(List<Integer> begin, List<Integer> end) {
        return crop((int[]) ListUtils.listToArray(int[].class, begin), ListUtils.listToArray(int[].class, end));
    }

    /**
     * Add 'convert layout' operation to specified layout.
     * Adds appropriate 'transpose' operation between user layout and target layout.Current implementation requires source and destination layout to have same number of dimensions
     * when user data has 'NHWC' layout (example is RGB image, [1, 224, 224, 3]) but model expects
     * planar input image ('NCHW', [1, 3, 224, 224]). Preprocessing may look like this:
     * PrePostProcessor proc = new PrePostProcessor(model);
     * proc.input().tensor().setLayout("NHWC"); // User data is NHWC
     * proc.input().preprocess().convertLayout("NCHW")) // model expects input as NCHW
     *
     * @param layout New layout after conversion. If not specified - destination layout is obtained from appropriate model input properties.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps convertLayout(Layout layout) {
        int status = getVino().ov_preprocess_preprocess_steps_convert_layout(getValue(), layout.getValue());
        if (status != 0) {

        }
        return this;
    }

    /**
     * Reverse channels operation.
     * Adds appropriate operation which reverses channels layout. Operation requires layout having 'C'
     * dimension Operation convert_color (RGB-BGR) does reversing of channels also, but only for NHWC layout
     * when user data has 'NCHW' layout (example is [1, 3, 224, 224] RGB order) but model expects
     * BGR planes order. Preprocessing may look like this:
     * PrePostProcessor proc = new PrePostProcessor(function);
     * proc.input().preprocess().convertLayout({0, 3, 1, 2});
     *
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps reverseChannels() {
        int status = getVino().ov_preprocess_preprocess_steps_reverse_channels(getValue());
        if (status != 0) {

        }
        return this;
    }

    /**
     * Converts color format for user's input tensor. Requires source color format to be specified by
     * inputTensorInfo::set_color_format.
     *
     * @param format Destination color format of input image.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps convertColor(ColorFormat format) {
        int status = getVino().ov_preprocess_preprocess_steps_convert_color(getValue(), format.ordinal());
        if (status != 0) {

        }
        return this;
    }

    /**
     * Add convert element type preprocess operation.
     *
     * @param type Desired type of input.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps convertElementType(ElementType type) {
        int status = getVino().ov_preprocess_preprocess_steps_convert_element_type(getValue(), type.ordinal());
        if (status != 0) {

        }
        return this;
    }
}
