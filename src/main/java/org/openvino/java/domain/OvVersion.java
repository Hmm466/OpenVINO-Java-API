package org.openvino.java.domain;

import com.sun.jna.Structure;

/**
 * Represents version information that describes plugins and the OpemVINO library
 */
@Structure.FieldOrder({"buildNumber","description"})
public class OvVersion extends Structure {

    /**
     * A null terminated string with build number
     */
    public String buildNumber;

    /**
     * A null terminated description string
     */
    public String description;
}
