package org.openvino.java.domain;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

/**
 * Represents basic inference profiling information per operation.
 * If the operation is executed using tiling, the sum time per each tile is indicated as the total execution time.
 * Due to parallel execution, the total execution time for all nodes might be greater than the total inference time.
 */
@Structure.FieldOrder({"real_time", "cpu_time", "node_name", "exec_type", "node_type"})
public class OvProfilingInfo extends Structure {

    /**
     * The absolute time, in microseconds, that the node ran (in total).
     */
    public long real_time;

    /**
     * The net host CPU time that the node ran.
     */
    public long cpu_time;

    /**
     * Name of a node.
     */
    public String node_name;

    /**
     * Execution type of a unit.
     */
    public String exec_type;

    /**
     * Node type.
     */
    public String node_type;

    public OvProfilingInfo() {

    }

    public OvProfilingInfo(Pointer pointer) {
        super(pointer);
    }
}
