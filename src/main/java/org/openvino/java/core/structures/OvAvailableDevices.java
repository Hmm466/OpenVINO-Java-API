package org.openvino.java.core.structures;

import com.sun.jna.Structure;
import com.sun.jna.ptr.PointerByReference;

@Structure.FieldOrder({"devices","size"})
public class OvAvailableDevices extends Structure {

    /**
     * devices' name
     */
    public PointerByReference devices;

    /**
     * devices' number
     */
    public long size;
}
