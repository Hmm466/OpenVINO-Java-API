package org.openvino.java.domain;

import com.sun.jna.Structure;
import com.sun.jna.ptr.LongByReference;

@Structure.FieldOrder({"rank","dims"})
public class OvShape extends Structure {

    public long rank;

    public LongByReference dims;
}
