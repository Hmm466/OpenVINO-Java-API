package org.openvino.java.domain;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.ptr.PointerByReference;

/**
 * A list of profiling info data
 */
@Structure.FieldOrder({"profiling_infos","size"})
public class OvProfilingInfoList extends Structure {

    /**
     * The list of ProfilingInfo
     */
    public Pointer profiling_infos;

    /**
     * he list size
     */
    public long size;
}
