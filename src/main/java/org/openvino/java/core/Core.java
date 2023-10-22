package org.openvino.java.core;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.core.structures.OvAvailableDevices;
import org.openvino.java.utils.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class represents an OpenVINO runtime Core entity.
 * User applications can create several Core class instances, but in this case the underlying plugins
 * are created multiple times and not shared between several Core instances.The recommended way is to have
 * a single Core instance per application.
 * @author ming
 */
public class Core extends OpenVINOCls {

    /**
     * Constructs an OpenVINO Core instance with devices and their plugins description.
     */
    public Core() {
        this(null);
    }

    /**
     * Constructs an OpenVINO Core instance with devices and their plugins description.
     *
     * @param xmlConfigFile Path to the .xml file with plugins to load from. If the XML configuration file is not.specified, default OpenVINO Runtime plugins are loaded from:
     */
    public Core(String xmlConfigFile) {
        super("Core");
        if (StringUtils.isNullOrEmpty(xmlConfigFile)) {
            verifyExceptionStatus(getVino().ov_core_create(this.getVinoObj()));
        } else {
            verifyExceptionStatus(getVino().ov_core_create_with_config(xmlConfigFile, this.getVinoObj()));
        }
    }

    /**
     * Reads models from IR / ONNX / PDPD / TF / TFLite formats.
     * @param modelPath Path to a model.
     * @return A model.
     */
    public Model readModel(String modelPath) {
        return readModel(modelPath,(String) null);
    }

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
     * @param modelPath Path to a model.
     * @param binPath Path to a data file.
     * @return A model.
     */
    public Model readModel(String modelPath,String binPath) {
        if (StringUtils.isNullOrEmpty(modelPath)) {
            throw new NullPointerException("model path is null");
        }
        PointerByReference model = new PointerByReference();
        verifyExceptionStatus(getVino().ov_core_read_model(getValue(),modelPath,binPath,model));
        Model _model = new Model(model);
        return _model;
    }

    /**
     * Creates and loads a compiled model from a source model to the default OpenVINO device selected by the AUTO
     * Users can create as many compiled models as they need and use
     * them simultaneously (up to the limitation of the hardware resources).
     * @param model Model object acquired from Core::read_model.
     * @param deviceName Name of a device to load a model to.
     * @return
     */
    public CompiledModel compileModel(Model model,String deviceName) {
        if (model == null) {
            throw new RuntimeException("model is null");
        }
        if (StringUtils.isNullOrEmpty(deviceName)) {
            throw new RuntimeException("deviceName is null");
        }
        PointerByReference compileModel = new PointerByReference();
        verifyExceptionStatus(getVino().ov_core_compile_model(getValue(),model.getValue(),deviceName,0,compileModel));
        return new CompiledModel(compileModel);
    }

    /**
     * Returns device plugins version information.
     * Device name can be complex and identify multiple devices at once like `HETERO:CPU,GPU`;
     * in this case, std::map contains multiple entries, each per device.
     * @param deviceName Device name to identify a plugin.
     * @return A vector of versions.
     */
    public void getVersions(String deviceName) {
        OvAvailableDevices devices = new OvAvailableDevices();
        verifyExceptionStatus(getVino().ov_core_get_available_devices(getValue(),devices));
        getVino().ov_core_versions_free(devices);
    }

    /**
     * Reads models from IR / ONNX / PDPD / TF / TFLite formats.
     * Created model object shares the weights with the @p weights object.
     * Thus, do not create @p weights on temporary data that can be freed later, since the model constant data will point to an invalid memory.
     * @param modelPath String with a model in IR / ONNX / PDPD / TF / TFLite format.
     * @param weights Shared pointer to a constant tensor with weights.
     * @return A model.
     */
    public Model readModel(String modelPath,Tensor weights) {
        if (modelPath == null) {
            throw new NullPointerException("model path is null");
        }
        if (weights == null) {
            throw new NullPointerException("weights is null");
        }
        PointerByReference model = new PointerByReference();
        verifyExceptionStatus(getVino().ov_core_read_model_from_memory(getValue(),modelPath,weights.getValue(),model));
        return new Model(model);
    }

    /**
     * Creates a compiled model from a source model object.
     * Users can create as many compiled models as they need and use
     * them simultaneously (up to the limitation of the hardware resources).
     * @param model Model object acquired from Core::read_model.
     * @return A compiled model.
     */
    public CompiledModel compileModel(Model model) {
        if (model == null) {
            throw new NullPointerException("model is null");
        }
        PointerByReference compiledModel = new PointerByReference();
        verifyExceptionStatus(getVino().ov_core_compile_model(getValue(),model.getValue(),"AUTO",0,compiledModel));
        return new CompiledModel(compiledModel);
    }

    /**
     * Reads a model and creates a compiled model from the IR/ONNX/PDPD file.
     * This can be more efficient than using the Core::read_model + Core::compile_model(model_in_memory_object) flow,
     * especially for cases when caching is enabled and a cached model is availab
     * @param modelPath Path to a model.
     * @param deviceName Name of a device to load a model to.
     * @return A compiled model.
     */
    public CompiledModel compiledModel(String modelPath,String deviceName) {
        if (StringUtils.isNullOrEmpty(modelPath) || StringUtils.isNullOrEmpty(deviceName)) {
            throw new NullPointerException("model path or device name is null");
        }
        PointerByReference compiledModel = new PointerByReference();
        verifyExceptionStatus(getVino().ov_core_compile_model_from_file(getValue(),modelPath,deviceName,0,compiledModel));
        return new CompiledModel(compiledModel);
    }

    /**
     * Reads and loads a compiled model from the IR/ONNX/PDPD file to the default OpenVINO device selected by the AUTO plugin.
     * This can be more efficient than using the Core::read_model + Core::compile_model(model_in_memory_object) flow,
     * especially for cases when caching is enabled and a cached model is availab
     * @param modelPath Path to a model.
     * @return A compiled model.
     */
    public CompiledModel compiledModel(String modelPath) {
        return compiledModel(modelPath,"AUTO");
    }

    /**
     * Returns devices available for inference.
     * Core objects go over all registered plugins and ask about available devices.
     * @return A vector of devices. The devices are returned as { CPU, GPU.0, GPU.1, GNA }.
     * If there is more than one device of a specific type, they are enumerated with the .# suffix.
     * Such enumerated device can later be used as a device name in all Core methods like Core::compile_model,
     * Core::query_model, Core::set_property and so on.
     */
    public List<String> getAvailableDevices() {
        OvAvailableDevices devices = new OvAvailableDevices();
        verifyExceptionStatus(getVino().ov_core_get_available_devices(getValue(),devices));
        String[] ds = devices.devices.getValue().getStringArray(0);
        getVino().ov_available_devices_free(devices.getPointer());
        return Arrays.asList(ds);
    }

    @Override
    protected void dispose() {
        if (getPointer() == null) {
            return;
        }
        getVino().ov_core_free(getValue());
        setVinoObj(null);
    }


}
