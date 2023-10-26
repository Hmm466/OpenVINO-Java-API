package org.openvino.java.base;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import lombok.Data;
import org.openvino.java.OpenVINO;
import org.openvino.java.core.VINO;
import org.openvino.java.enums.ExceptionStatus;

@Data
public abstract class OpenVINOCls {

    private PointerByReference vinoObj;

    private VINO vino;

    private String type;

    public OpenVINOCls(String type) {
        vino = OpenVINO.getCore();
        initVINOObj();
        this.type = type;
    }

    public OpenVINOCls(String type, PointerByReference structure) {
        vino = OpenVINO.getCore();
        setVinoObj(structure);
        this.type = type;
    }

    /**
     * verify if an exception occurred when calling the function
     * @param status the state returned by the function
     * @throws RuntimeException If the status is not OK, a RuntimeException will be thrown
     */
    protected void verifyExceptionStatus(int status) {
        if (status == ExceptionStatus.OK.getCode()) {

        } else if (status == ExceptionStatus.GENERAL_ERROR.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.GENERAL_ERROR.getMsg());
        } else if (status == ExceptionStatus.NOT_IMPLEMENTED.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.NOT_IMPLEMENTED.getMsg());
        } else if (status == ExceptionStatus.NETWORK_NOT_LOADED.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.NETWORK_NOT_LOADED.getMsg());
        } else if (status == ExceptionStatus.PARAMETER_MISMATCH.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.PARAMETER_MISMATCH.getMsg());
        } else if (status == ExceptionStatus.NOT_FOUND.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.NOT_FOUND.getMsg());
        } else if (status == ExceptionStatus.OUT_OF_BOUNDS.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.OUT_OF_BOUNDS.getMsg());
        } else if (status == ExceptionStatus.UNEXPECTED.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.UNEXPECTED.getMsg());
        } else if (status == ExceptionStatus.REQUEST_BUSY.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.REQUEST_BUSY.getMsg());
        } else if (status == ExceptionStatus.RESULT_NOT_READY.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.RESULT_NOT_READY.getMsg());
        } else if (status == ExceptionStatus.NOT_ALLOCATED.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.NOT_ALLOCATED.getMsg());
        } else if (status == ExceptionStatus.INFER_NOT_STARTED.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.INFER_NOT_STARTED.getMsg());
        } else if (status == ExceptionStatus.NETWORK_NOT_READ.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.NETWORK_NOT_READ.getMsg());
        } else if (status == ExceptionStatus.INFER_CANCELLED.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.INFER_CANCELLED.getMsg());
        } else if (status == ExceptionStatus.INVALID_C_PARAM.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.INVALID_C_PARAM.getMsg());
        } else if (status == ExceptionStatus.UNKNOWN_C_ERROR.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.UNKNOWN_C_ERROR.getMsg());
        } else if (status == ExceptionStatus.NOT_IMPLEMENT_C_METHOD.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.NOT_IMPLEMENT_C_METHOD.getMsg());
        } else if (status == ExceptionStatus.UNKNOW_EXCEPTION.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.UNKNOW_EXCEPTION.getMsg());
        } else if (status == ExceptionStatus.PTR_NULL.getCode()) {
            throw new RuntimeException(getType() + ":" + ExceptionStatus.PTR_NULL.getMsg());
        }
    }

    void initVINOObj() {
        vinoObj = new PointerByReference();
    }

    public Pointer getPointer() {
        if (this.getVinoObj() == null) {
            return null;
        }
        return this.getVinoObj().getPointer();
    }

    public Pointer getValue() {
        if (this.getVinoObj() == null) {
            return null;
        }
        return this.getVinoObj().getValue();
    }

    /**
     * Release unmanaged resources
     */
    protected abstract void dispose();

    public boolean isNull() {
        return this.getVinoObj() == null;
    }
}
