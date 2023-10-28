package org.openvino.java.core;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.domain.OvProfilingInfo;
import org.openvino.java.domain.OvProfilingInfoList;
import org.openvino.java.enums.ExceptionStatus;
import org.openvino.java.enums.NodeType;
import org.openvino.java.utils.Console;

import java.util.ArrayList;
import java.util.List;

public class InferRequest extends OpenVINOCls {
    public InferRequest(PointerByReference request) {
        super("InferRequest",request);

    }

    /**
     * Gets an input tensor for inference.
     * @return The input tensor for the model. If model has several inputs, an exception is thrown.
     */
    public Tensor getInputTensor() {
        PointerByReference tensor = new PointerByReference();
        verifyExceptionStatus(getVino().ov_infer_request_get_input_tensor(getValue(),tensor));
        return new Tensor(tensor);
    }

    /**
     * Gets an input tensor for inference.
     * @param index Index of the tensor to get.
     * @return Tensor with the input index @p idx. If the tensor with the specified @p idx is not found,an exception is thrown.
     */
    public Tensor getInputTensor(long index) {
        PointerByReference tensor = new PointerByReference();
        verifyExceptionStatus(getVino().ov_infer_request_get_input_tensor_by_index(getValue(),index,tensor));
        return new Tensor(tensor);
    }

    /**
     * Gets an output tensor for inference.
     * @return Output tensor for the model. If model has several outputs, an exception is thrown.
     */
    public Tensor getOutputTensor() {
        PointerByReference tensor = new PointerByReference();
        verifyExceptionStatus(getVino().ov_infer_request_get_output_tensor(getValue(),tensor));
        return new Tensor(tensor);
    }

    /**
     * Gets an output tensor for inference.
     * @param index Index of the tensor to get.
     * @return Tensor with the output index @p idx. If the tensor with the specified @p idx is not found, an exception is thrown
     */
    public Tensor getOutputTensor(long index) {
        PointerByReference tensor = new PointerByReference();
        verifyExceptionStatus(getVino().ov_infer_request_get_output_tensor_by_index(getValue(),index,tensor));
        return new Tensor(tensor);
    }

    @Override
    protected void dispose() {
        if (!isNull()) {
            getVino().ov_infer_request_free(getValue());
            setVinoObj(null);
        }
    }

    public void infer() {
        verifyExceptionStatus(getVino().ov_infer_request_infer(getValue()));
    }

    /**
     * Gets an input/output tensor for inference by tensor name.
     * @param tensorName Name of a tensor to get.
     * @return The tensor with name @p tensor_name. If the tensor is not found, an exception is thrown.
     */
    public Tensor getTensor(String tensorName) {
        PointerByReference tensor = new PointerByReference();
        verifyExceptionStatus(getVino().ov_infer_request_get_tensor(getValue(),tensorName,tensor));
        return new Tensor(tensor);
    }

    /**
     * Gets an input/output tensor for inference by node.
     * @param node Node of the tensor to get.
     * @return Tensor for the node @n node.
     */
    public Tensor getTensor(Node node) {
        PointerByReference tensor = new PointerByReference();
        int status = 0;
        if (node.getNodeType() == NodeType.e_const) {
            status = getVino().ov_infer_request_get_tensor_by_const_port(getValue(),node.getValue(),tensor);
        } else {
            status = getVino().ov_infer_request_get_tensor_by_port(getValue(),node.getValue(),tensor);
        }
        if (status != ExceptionStatus.OK.getCode()) {
            return null;
        }
        return new Tensor(tensor);
    }

    /**
     * Gets an input/output tensor for inference.
     * @param port Port of the tensor to get.
     * @return Tensor for the port @p port.
     */
    public Tensor getTensor(Output port) {
        PointerByReference tensor = new PointerByReference();
        int status = 0;
        if (port.getNode().getNodeType() == NodeType.e_const) {
            status = getVino().ov_infer_request_get_tensor_by_const_port(getValue(),port.getNode().getValue(),tensor);
        } else {
            status = getVino().ov_infer_request_get_tensor_by_port(getValue(),port.getNode().getValue(),tensor);
        }
        if (status != ExceptionStatus.OK.getCode()) {
            return null;
        }
        return new Tensor(tensor);
    }

    /**
     * Sets an input/output tensor to infer on.
     * @param tensorName Name of the input or output tensor.
     * @param tensor Reference to the tensor. The element_type and shape of the tensor must match the model's input/output element_type and size.
     */
    public void setTensor(String tensorName,Tensor tensor) {
        verifyExceptionStatus(getVino().ov_infer_request_set_tensor(getValue(),tensorName,tensor.getValue()));
    }

    /**
     * Sets an input/output tensor to infer.
     * @param node Node of the input or output tensor.
     * @param tensor Reference to a tensor. The element_type and shape of a tensor must match the model's input/output element_type and size.
     */
    public void setTensor(Node node,Tensor tensor) {
        if (node.getNodeType() == NodeType.e_const) {
            verifyExceptionStatus(getVino().ov_infer_request_set_tensor_by_const_port(getValue(),node.getValue(),tensor.getValue()));
        } else {
            verifyExceptionStatus(getVino().ov_infer_request_set_tensor_by_port(getValue(),node.getValue(),tensor.getValue()));
        }
    }

    /**
     * Sets an input/output tensor to infer.
     * @param port Port of the input or output tensor. Use the following methods to get the ports:
     * - Model.input()
     * - Model.const_input()
     * - Model.inputs()
     * - Model.const_inputs()
     * - Model.output()
     * - Model.const_output()
     * - Model.outputs()
     * - Model.const_outputs()
     * - CompiledModel.input()
     * - CompiledModel.const_input()
     * - CompiledModel.inputs()
     * - CompiledModel.const_inputs()
     * - CompiledModel.output()
     * - CompiledModel.const_output()
     * - CompiledModel.outputs()
     * - CompiledModel.const_outputs()
     * @param tensor Reference to a tensor. The element_type and shape of a tensor must match the model's input/output element_type and size.
     */
    public void setTensor(Output port,Tensor tensor) {
        if (port.getNode().getNodeType() == NodeType.e_const) {
            verifyExceptionStatus(getVino().ov_infer_request_set_tensor_by_const_port(getValue(),port.getNode().getValue(),tensor.getValue()));
        } else {
            verifyExceptionStatus(getVino().ov_infer_request_set_tensor_by_port(getValue(),port.getNode().getValue(),tensor.getValue()));
        }
    }

    /**
     * Sets an input tensor to infer.
     * @param index Index of the input tensor. If @p idx is greater than the number of model inputs, an exception is thrown.
     * @param tensor Reference to the tensor. The element_type and shape of the tensor must match the model's input/output element_type and size.
     */
    public void setInputTensor(long index,Tensor tensor) {
        verifyExceptionStatus(getVino().ov_infer_request_set_input_tensor_by_index(getValue(),index,tensor.getValue()));
    }

    /**
     * Sets an input tensor to infer models with single input.
     * @param tensor Reference to the input tensor.
     * @exception null If model has several inputs, an exception is thrown.
     */
    public void setInputTensor(Tensor tensor) {
        verifyExceptionStatus(getVino().ov_infer_request_set_input_tensor(getValue(),tensor.getValue()));
    }

    /**
     * Sets an output tensor to infer.Index of the input preserved accross Model, CompiledModel, and InferRequest.
     * @param index Index of the output tensor.
     * @param tensor Reference to the output tensor.The type of the tensor must match the model output element type and shape.
     */
    public void setOutputTensor(long index,Tensor tensor) {
        verifyExceptionStatus(getVino().ov_infer_request_set_output_tensor_by_index(getValue(),index,tensor.getValue()));
    }

    /**
     * Sets an output tensor to infer models with single output.
     * @param tensor Reference to the output tensor.
     * @exception null. If model has several outputs, an exception is thrown.
     */
    public void setOutputTensor(Tensor tensor) {
        verifyExceptionStatus(getVino().ov_infer_request_set_output_tensor(getValue(),tensor.getValue()));
    }

    /**
     * Cancels inference request.
     */
    public void cancel() {
        verifyExceptionStatus(getVino().ov_infer_request_cancel(getValue()));
    }

    /**
     * Starts inference of specified input(s) in asynchronous mode.
     * It returns immediately. Inference starts also immediately.
     * Calling any method while the request in a running state leads to throwning the ov::Busy exception.
     */
    public void startAsync() {
        verifyExceptionStatus(getVino().ov_infer_request_start_async(getValue()));
    }

    /**
     * Waits for the result to become available. Blocks until the result becomes available.
     */
    public void startWait() {
        verifyExceptionStatus(getVino().ov_infer_request_wait(getValue()));
    }

    /**
     * Waits for the result to become available. Blocks until the specified timeout has elapsed or the result becomes available, whichever comes first.
     * @param timeout Maximum duration, in milliseconds, to block for.
     */
    public void startWait(long timeout) {
        verifyExceptionStatus(getVino().ov_infer_request_wait_for(getValue(),timeout));
    }

    /**
     * Queries performance measures per layer to identify the most time consuming operation.
     * Not all plugins provide meaningful data.
     * @return List of profiling information for operations in a model.
     */
    public List<OvProfilingInfo> getProfilingInfo() {
        OvProfilingInfoList list = new OvProfilingInfoList();
        verifyExceptionStatus(getVino().ov_infer_request_get_profiling_info(getValue(),list));
        List<OvProfilingInfo> infos = new ArrayList<>();
        for (int i = 0 ; i < list.size;i++) {
            infos.add(new OvProfilingInfo(list.profiling_infos.getPointer(i)));
        }
        getVino().ov_profiling_info_list_free(list);
        return infos;
    }
}
