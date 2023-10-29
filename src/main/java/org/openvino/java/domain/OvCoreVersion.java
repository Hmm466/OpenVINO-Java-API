package org.openvino.java.domain;

import com.sun.jna.Structure;

/**
 * Represents version information that describes device and ov runtime library
 */
@Structure.FieldOrder({"device_name", "version"})
public class OvCoreVersion extends Structure {
    /**
     * A device name
     */
    public String device_name;
    public OvVersion version;
}
