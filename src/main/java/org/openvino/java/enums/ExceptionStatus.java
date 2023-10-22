package org.openvino.java.enums;

/**
 * This enum contains codes for all possible return values of the interface functions
 */
public enum ExceptionStatus {

    OK(0,"SUCCESS")
    ,GENERAL_ERROR(-1,"map exception to C++ interface")
    ,NOT_IMPLEMENTED(-2,"not implemented!")
    ,NETWORK_NOT_LOADED(-3,"network not loaded!")
    ,PARAMETER_MISMATCH(-4,"parameter mismatch!")
    ,NOT_FOUND(-5,"not found!")
    ,OUT_OF_BOUNDS(-6,"out of bounds!")
    ,UNEXPECTED(-7,"unexpection!")
    ,REQUEST_BUSY(-8,"request busy!")
    ,RESULT_NOT_READY(-9,"result not ready!")
    ,NOT_ALLOCATED(-10,"not allocated!")
    ,INFER_NOT_STARTED(-11,"infer not started!")
    ,NETWORK_NOT_READ(-12,"netword not read!")
    ,INFER_CANCELLED(-13,"infer cancelled!")
    ,INVALID_C_PARAM(-14,"invalid c param!")
    ,UNKNOWN_C_ERROR(-15,"unknown c error!")
    ,NOT_IMPLEMENT_C_METHOD(-16,"not implement c method!")
    ,UNKNOW_EXCEPTION(-17,"unknown exception!")
    ,PTR_NULL(-100,"ptr is null!")
    ;

    private int code;

    private String msg;

    public int getCode() {
        return code;
    }

    public String getMsg() {
        return msg;
    }

    ExceptionStatus(int code, String msg) {
        this.code = code;
        this.msg = msg;
    }
}
