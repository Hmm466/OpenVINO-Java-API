package org.openvino.java.core;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.domain.*;
import org.openvino.java.utils.StringUtils;
import org.openvino.java.utils.SystemUtils;

public interface VINO extends Library {
    int ov_get_openvino_version(OvVersion version);

    /**
     * Constructs OpenVINO Core instance using XML configuration file with devices description.
     * See RegisterPlugins for more details.
     * @param xmlFile A path to .xml file with devices to load from. If XML configuration file is not specified, then default plugin.xml file will be used.
     * @param core A pointer to the newly created ov_core_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_core_create_with_config(String xmlFile,int core);

    /**
     * Constructs OpenVINO Core instance using XML configuration file with devices description.
     * See RegisterPlugins for more details.
     * @param xmlConfigFile A path to .xml file with devices to load from. If XML configuration file is not specified, then default plugin.xml file will be used.
     * @param core A pointer to the newly created ov_core_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_core_create_with_config(String xmlConfigFile, PointerByReference core);

    /**
     * Constructs OpenVINO Core instance by default.
     * See RegisterPlugins for more details.
     * @param core A pointer to the newly created ov_core_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_core_create(PointerByReference core);

    /**
     * Reads models from IR / ONNX / PDPD / TF / TFLite formats.
     * For IR format (*.bin):
     * if `bin_path` is empty, will try to read a bin file with the same name as xml and
     * if the bin file with the same name is not found, will load IR without weights.
     * For the following file formats the `bin_path` parameter is not used:
     *  ONNX format (*.onnx)
     *  PDPD(*.pdmodel)
     *  TF(*.pb)</para>
     *  TFLite(*.tflite)
     * @param core A pointer to the ie_core_t instance.
     * @param modelPath Path to a model.
     * @param binPath Path to a data file.
     * @param model A pointer to the newly created model.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_core_read_model(Pointer core,String modelPath,String binPath,PointerByReference model);

    /**
     * Reads a model and creates a compiled model from the IR/ONNX/PDPD file.
     * This can be more efficient than using the ov_core_read_model_from_XXX + ov_core_compile_model flow,
     * especially for cases when caching is enabled and a cached model is available.
     * @param core A pointer to the ie_core_t instance.
     * @param model Path to a model.
     * @param deviceName Name of a device to load a model to.
     * @param propertyArgsSize How many properties args will be passed, each property contains 2 args: key and value.
     * @param compiledModel A pointer to the newly created compiled_model.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_core_compile_model(Pointer core,Pointer model,String deviceName,long propertyArgsSize,PointerByReference compiledModel);

    /**
     * Gets the friendly name for a model.
     * @param model A pointer to the ov_model_t.
     * @param friendlyName the model's friendly name.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_model_get_friendly_name(Pointer model,PointerByReference friendlyName);

    /**
     * Get the single const input port of ov_compiled_model_t, which only support single input model.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param inputPort A pointer to the ov_output_const_port_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_input(Pointer compiledModel,PointerByReference inputPort);

    /**
     * Get the tensor name of port.
     * @param port A pointer to the ov_output_const_port_t.
     * @param tensorName A pointer to the tensor name.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_port_get_any_name(Pointer port,PointerByReference tensorName);

    /**
     * Get the tensor type of port.
     * @param port A pointer to the ov_output_const_port_t.
     * @param tensorType tensor type.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_port_get_element_type(Pointer port, IntByReference tensorType);

    /**
     * Get the shape of port object.
     * @param port A pointer to ov_output_const_port_t.
     * @param shape tensor shape.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_const_port_get_shape(Pointer port, OvShape shape);

    /**
     * Get the output size of ov_compiled_model_t.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param outputPort A pointer to the ov_output_const_port_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_output(Pointer compiledModel,PointerByReference outputPort);

    /**
     * Creates an inference request object used to infer the compiled model.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_create_infer_request(Pointer compiledModel,PointerByReference inferRequest);

    /**
     * Get an input tensor from the model with only one input tensor.
     * @param infer_request A pointer to the ov_infer_request_t.
     * @param tensor Reference to the tensor.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_get_input_tensor(Pointer infer_request,PointerByReference tensor);

    /**
     *
     * @param tensor
     * @param shape
     * @return
     */
    int ov_tensor_get_shape(Pointer tensor, OvShape shape);

    /**
     * the total number of elements (a product of all the dims or 1 for scalar).
     * @param tensor A point to ov_tensor_t.
     * @param elementsSize number of elements.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_tensor_get_size(Pointer tensor, LongByReference elementsSize);

    /**
     * Get a const input port of ov_model_t by port index.
     * @param model A pointer to the ov_model_t.
     * @param index input tensor index.
     * @param inputPort A pointer to the ov_output_port_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_model_const_input_by_index(Pointer model,long index,PointerByReference inputPort);

    /**
     * Get a const output port of ov_model_t by port index.
     * @param model A pointer to the ov_model_t.
     * @param index input tensor index.
     * @param outputPort A pointer to the ov_output_const_port_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_model_const_output_by_index(Pointer model,long index,PointerByReference outputPort);

    int ov_output_const_port_free(Pointer port);

    int ov_output_port_free(Pointer port);

    int ov_infer_request_infer(Pointer inferRequest);

    int ov_infer_request_get_output_tensor(Pointer inferRequest,PointerByReference tensor);

    int ov_tensor_data(Pointer tensor,PointerByReference data);

    int ov_infer_request_get_tensor(Pointer model,String name,PointerByReference tensor);

    /**
     * Release the memory allocated by ov_core_t.
     * @param core A pointer to the ov_core_t to free memory.
     */
    void ov_core_free(Pointer core);

    /**
     * Returns devices available for inference.
     * @param core A pointer to the ie_core_t instance.
     * @param devices A pointer to the ov_available_devices_t instance.Core objects go over all registered plugins and ask about available devices.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_core_get_available_devices(Pointer core, OvAvailableDevices devices);

    /**
     * Releases memory occpuied by ov_available_devices_t
     * @param devices A pointer to the ov_available_devices_t instance.
     */
    void ov_available_devices_free(Pointer devices);

    /**
     * Returns device plugins version information.
     * Device name can be complex and identify multiple devices at once like `HETERO:CPU,GPU`;
     * in this case, std::map contains multiple entries, each per device.
     * @param core A pointer to the ov_core_t instance.
     * @param deviceName Device name to identify a plugin.
     * @param version A pointer to versions corresponding to device_name.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_core_get_versions_by_device_name(Pointer core, String deviceName, OvCoreVersionList version);

    /**
     * Reads models from IR / ONNX / PDPD / TF / TFLite formats.
     * Reading ONNX / PDPD / TF / TFLite models does not support loading weights from the @p weights tensors.<
     * Created model object shares the weights with the @p weights object.
     * Thus, do not create @p weights on temporary data that can be freed later,
     * since the model constant data will point to an invalid memory.
     * @param core A pointer to the ie_core_t instance.
     * @param modelPath Path to a model.
     * @param weights Shared pointer to a constant tensor with weights.
     * @param model A pointer to the newly created model.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_core_read_model_from_memory(Pointer core,String modelPath,Pointer weights,PointerByReference model);

    /**
     * Reads a model and creates a compiled model from the IR/ONNX/PDPD file.
     * This can be more efficient than using the ov_core_read_model_from_XXX + ov_core_compile_model flow,
     * especially for cases when caching is enabled and a cached model is available.
     * @param core A pointer to the ie_core_t instance.
     * @param modelPath Path to a model.
     * @param deviceName Name of a device to load a model to.
     * @param propertyArgsSize How many properties args will be passed, each property contains 2 args: key and value.
     * @param compiledModel A pointer to the newly created compiled_model.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_core_compile_model_from_file(Pointer core,String modelPath,String deviceName,long propertyArgsSize,PointerByReference compiledModel);

    /**
     * Releases memory occupied by ov_core_version_list_t.
     * @param devices A pointer to the ie_core_versions to free memory.
     */
    void ov_core_versions_free(OvAvailableDevices devices);

    /**
     * Get a const input port of ov_compiled_model_t by name.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param name input tensor name (char *).
     * @param inputPort A pointer to the ov_output_const_port_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_input_by_name(Pointer compiledModel,String name,PointerByReference inputPort);

    /**
     * Get a const input port of ov_compiled_model_t by index.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param index input index.
     * @param inputPort A pointer to the ov_output_const_port_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_input_by_index(Pointer compiledModel,long index,PointerByReference inputPort);

    /**
     * Get a const output port of ov_compiled_model_t by name.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param name input tensor name (char *).
     * @param outputPort A pointer to the ov_output_const_port_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_output_by_name(Pointer compiledModel,String name,PointerByReference outputPort);

    /**
     * Get a const output port of ov_compiled_model_t by index.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param index input index.
     * @param outputPort A pointer to the ov_output_const_port_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_output_by_index(Pointer compiledModel,long index,PointerByReference outputPort);

    /**
     * Get the input size of ov_compiled_model_t.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param size the compiled_model's input size.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_inputs_size(Pointer compiledModel, LongByReference size);

    /**
     * Get the output size of ov_compiled_model_t.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param size the compiled_model's output size.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_outputs_size(Pointer compiledModel, LongByReference size);

    /**
     * Gets runtime model information from a device.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param model A pointer to the ov_model_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_get_runtime_model(Pointer compiledModel, PointerByReference model);

    /**
     * Exports the current compiled model to an output stream `std::ostream`.
     * The exported model can also be imported via the ov::Core::import_model method.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param modelPath Path to the file.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_export_model(Pointer compiledModel,String modelPath);

    /**
     * Sets properties for a device, acceptable keys can be found in ov_property_key_xxx.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param key The property key string.
     * @param value The property value string.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_set_property(Pointer compiledModel,String key,String value);

    /**
     * Gets properties for current compiled model.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param key Property key.
     * @param value A pointer to property value.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_get_property(Pointer compiledModel,String key,PointerByReference value);

    /**
     * Returns pointer to device-specific shared context on a remote accelerator
     * device that was used to create this CompiledModel.
     * @param compiledModel A pointer to the ov_compiled_model_t.
     * @param context Return context.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_compiled_model_get_context(Pointer compiledModel,PointerByReference context);

    /**
     * Release the memory allocated by ov_infer_request_t.
     * @param inferRequest A pointer to the ov_infer_request_t to free memory.
     * @return
     */
    void ov_infer_request_free(Pointer inferRequest);

    /**
     * Set an input/output tensor to infer on by the name of tensor.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @param tensorName Name of the input or output tensor.
     * @param tensor Reference to the tensor.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_set_tensor(Pointer inferRequest,String tensorName,Pointer tensor);

    /**
     * Set an input/output tensor to infer request for the port.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @param port Port of the input or output tensor, which can be got by calling ov_model_t/ov_compiled_model_t interface.
     * @param tensor Reference to the tensor.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_set_tensor_by_port(Pointer inferRequest,Pointer port,Pointer tensor);

    /**
     * Set an input/output tensor to infer request for the port.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @param port Const port of the input or output tensor, which can be got by call interface from ov_model_t/ov_compiled_model_t.
     * @param tensor Reference to the tensor.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_set_tensor_by_const_port(Pointer inferRequest,Pointer port,Pointer tensor);

    /**
     * Set an input tensor to infer on by the index of tensor.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @param index Index of the input port. If @p idx is greater than the number of model inputs, an error will return.
     * @param tensor Reference to the tensor.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_set_input_tensor_by_index(Pointer inferRequest,long index,Pointer tensor);

    /**
     * Set an input tensor for the model with single input to infer on.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @param tensor Reference to the tensor.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_set_input_tensor(Pointer inferRequest,Pointer tensor);

    /**
     * Set an output tensor to infer by the index of output tensor.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @param index Index of the output tensor.
     * @param tensor Reference to the tensor.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_set_output_tensor_by_index(Pointer inferRequest,long index,Pointer tensor);

    /**
     * Set an output tensor to infer models with single output.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @param tensor Reference to the tensor.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_set_output_tensor(Pointer inferRequest,Pointer tensor);

    /**
     * Get an input/output tensor by const port.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @param port Port of the tensor to get. @p port is not found, an error will return.
     * @param tensor Reference to the tensor.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_get_tensor_by_const_port(Pointer inferRequest,Pointer port,PointerByReference tensor);

    /**
     * Get an input/output tensor by port.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @param port Port of the tensor to get. @p port is not found, an error will return.
     * @param tensor Reference to the tensor.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_get_tensor_by_port(Pointer inferRequest,Pointer port,PointerByReference tensor);

    /**
     * Get an input tensor by the index of input tensor.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @param index index of the tensor to get. @p idx. If the tensor with the specified @p idx is not found, an error will return.
     * @param tensor Reference to the tensor.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_get_input_tensor_by_index(Pointer inferRequest,long index,PointerByReference tensor);

    /**
     * Get an output tensor by the index of output tensor.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @param index index of the tensor to get. @p idx. If the tensor with the specified @p idx is not found, an error will return.
     * @param tensor Reference to the tensor.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_get_output_tensor_by_index(Pointer inferRequest,long index,PointerByReference tensor);

    /**
     * Cancel inference request.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_cancel(Pointer inferRequest);

    /**
     * Start inference of specified input(s) in asynchronous mode.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_start_async(Pointer inferRequest);

    /**
     * Wait for the result to become available.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_wait(Pointer inferRequest);

    /**
     * Waits for the result to become available. Blocks until the specified timeout has elapsed or the result becomes available, whichever comes first.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @param timeout Maximum duration, in milliseconds, to block for.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_wait_for(Pointer inferRequest,long timeout);

    /**
     * Query performance measures per layer to identify the most time consuming operation.
     * @param inferRequest A pointer to the ov_infer_request_t.
     * @param list Vector of profiling information for operations in a model.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_infer_request_get_profiling_info(Pointer inferRequest, OvProfilingInfoList list);

    /**
     * Release the memory allocated by ov_profiling_info_list_t.
     * @param list A pointer to the ov_profiling_info_list_t to free memory.
     * @return Status code of the operation: OK(0) for success.
     */
    int ov_profiling_info_list_free(OvProfilingInfoList list);

    static VINO load(String path) {
        int osType = SystemUtils.getSystemType();
        if (StringUtils.isNullOrEmpty(path)) {
            switch (osType) {
                case SystemUtils.SYSTEM_WINDOWS:
                    path = "libopenvino_c.dll";
                    break;
                case SystemUtils.SYSTEM_LINUX:
                    path = "libopenvino_c.so";
                    break;
                case SystemUtils.SYSTEM_MAC:
                    path = "libopenvino_c.dylib";
                    break;
                default:
                    throw new UnsupportedOperationException("The current API does not support your operating system");
            }
        }
        VINO vino = Native.loadLibrary(path, VINO.class);
        return vino;
    }

    static VINO loadHttp(String url) {
        return null;
    }
}
