package org.openvino.java.domain;

import com.sun.jna.Structure;
import com.sun.jna.ptr.PointerByReference;

/**
 * [struct] Represents version information that describes all devices and ov runtime library
 */
@Structure.FieldOrder({"core_version","size"})
public class OvCoreVersionList extends Structure {

    /**
     * An array of device versions
     */
    public PointerByReference core_version;

    /**
     * A number of versions in the array
     */
    public long size;
}
