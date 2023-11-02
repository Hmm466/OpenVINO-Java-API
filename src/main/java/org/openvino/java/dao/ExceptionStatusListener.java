package org.openvino.java.dao;

/**
 * Abnormal state event callback
 */
public interface ExceptionStatusListener {

    /**
     *
     */
    void ok();

    /**
     * When an exception occurs, a callback will be performed here
     * @param type Error type
     * @param source Entity Element Occurred
     * @param error Error message content
     */
    void exception(String type,String source,String error);
}
