package org.openvino.java.domain;

import com.sun.jna.Structure;

@Structure.FieldOrder({"buildNumber","description"})
public class Version extends Structure {
    public String buildNumber;

    public String description;
}
