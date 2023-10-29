package org.openvino.java.preprocess;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;

public class PreProcessSteps extends OpenVINOCls {
    public PreProcessSteps(PointerByReference ptr) {
        super("PreProcessSteps", ptr);
    }

    @Override
    protected void dispose() {

    }
}
