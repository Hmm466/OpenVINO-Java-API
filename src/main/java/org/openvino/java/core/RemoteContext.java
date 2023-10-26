package org.openvino.java.core;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;

/**
 * This class represents an abstraction for remote (non-CPU) accelerator device-specific inference context.
 * Such context represents a scope on the device within which compiled models and remote memory tensors can exist,
 * function, and exchange data.
 */
public class RemoteContext extends OpenVINOCls {

    public RemoteContext(PointerByReference structure) {
        super("RemoteContext", structure);
    }

    @Override
    protected void dispose() {

    }
}
