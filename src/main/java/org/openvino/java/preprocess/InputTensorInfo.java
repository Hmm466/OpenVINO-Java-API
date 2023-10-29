package org.openvino.java.preprocess;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.core.Layout;
import org.openvino.java.core.Tensor;
import org.openvino.java.enums.ColorFormat;
import org.openvino.java.enums.ElementType;

/**
 * Information about user's input tensor. By default, it will be initialized to same data (type/shape/etc) as
 * model's input parameter. User application can override particular parameters (like 'element_type') according to
 * application's data and specify appropriate conversions in pre-processing steps
 */
public class InputTensorInfo extends OpenVINOCls {

    /**
     * Default construction through InputTensorInfo pointer.
     *
     * @param ptr InputTensorInfo pointer.
     */
    public InputTensorInfo(PointerByReference ptr) {
        super("InputTensorInfo", ptr);
    }

    @Override
    protected void dispose() {
        if (!isNull()) {
            getVino().ov_preprocess_input_tensor_info_free(getValue());
            setVinoObj(null);
        }
    }

    /**
     * Set color format for user's input tensor.
     * In general way, some formats support multi-plane input, e.g. NV12 image can be represented as 2 separate tensors
     * (planes): Y plane and UV plane. set_color_format API also allows to set sub_names for such parameters for
     * convenient usage of plane parameters. During build stage, new parameters for each plane will be inserted to the
     * place of original parameter. This means that all parameters located after will shift their positions accordingly
     * (e.g. {param1, param2} will become {param1/Y, param1/UV, param2})
     *
     * @param format Color format of input image.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public InputTensorInfo setColorFormat(ColorFormat format) {
        getVino().ov_preprocess_input_tensor_info_set_color_format(getValue(), format.ordinal());
        return this;
    }

    /**
     * @param format
     * @param subNamesSize
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public InputTensorInfo setColorFormat(ColorFormat format, long subNamesSize) {
        getVino().ov_preprocess_input_tensor_info_set_color_format_with_subname(getValue(), format.ordinal(), subNamesSize);
        return this;
    }

    /**
     * Set element type for user's input tensor
     *
     * @param type Element type for user's input tensor.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public InputTensorInfo setElementType(ElementType type) {
        getVino().ov_preprocess_input_tensor_info_set_element_type(getValue(), type.ordinal());
        return this;
    }

    /**
     * By default, input image shape is inherited from model input shape. Use this method to specify different
     * width and height of user's input image. In case if input image size is not known, use
     * `set_spatial_dynamic_shape` method.
     *
     * @param inputHeight Set fixed user's input image height.
     * @param inputWidth  Set fixed user's input image width.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public InputTensorInfo setSpatialStaticShape(long inputHeight, long inputWidth) {
        getVino().ov_preprocess_input_tensor_info_set_spatial_static_shape(getValue(), inputHeight, inputWidth);
        return this;
    }

    /**
     * Set memory type runtime information for user's input tensor
     *
     * @param memoryType Memory type. Refer to specific plugin's documentation for exact string format
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public InputTensorInfo setMemoryType(String memoryType) {
        getVino().ov_preprocess_input_tensor_info_set_memory_type(getValue(), memoryType);
        return this;
    }

    /**
     * Set layout for user's input tensor
     *
     * @param layout Layout for user's input tensor.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public InputTensorInfo setLayout(Layout layout) {
        getVino().ov_preprocess_input_tensor_info_set_layout(getValue(), layout.getValue());
        return this;
    }

    /**
     * Helper function to reuse element type and shape from user's created tensor. Use this only in case if
     * input tensor is already known and available before. Overwrites previously set element type & shape via
     * `set_element_type` and `set_shape`. Tensor's memory type is not reused, so if `runtime_tensor` represents remote
     * tensor with particular memory type - you should still specify appropriate memory type manually using
     * `set_memory_type`
     * As for `InputTensorInfo::set_shape`, this method shall not be used together with methods
     * set_spatial_dynamic_shape' and 'set_spatial_static_shape', otherwise ov::AssertFailure exception will be thrown
     *
     * @param runtimeTensor User's created tensor.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner
     */
    public InputTensorInfo setFrom(Tensor runtimeTensor) {
        getVino().ov_preprocess_input_tensor_info_set_from(getValue(), runtimeTensor.getValue());
        return this;
    }
}
