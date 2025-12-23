package org.openvino.java.domain;

import com.sun.jna.Structure;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;

@Structure.FieldOrder({"rank", "dims"})
public class OvShape extends Structure {

    public long rank;

    public PointerByReference dims;
}
