package org.openvino.java.core;

import lombok.Data;
import org.openvino.java.domain.OvShape;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Data
public class Shape {

    private OvShape ovShapeT;

    private long rank;

    private List<Long> dims;

    public Shape(OvShape shapeT) {
        dims = new ArrayList<>();
        rank = shapeT.rank;
        long[] data = shapeT.dims.getPointer().getLongArray(0,(int)shapeT.rank);
        for (int i = 0 ; i < data.length;i++) {
            dims.add(data[i]);
        }
    }

    @Override
    public String toString() {
        return "Shape{" +
                ", rank=" + rank +
                ", dims=" + dims.stream().map(String::valueOf).collect(Collectors.joining(",")) +
                '}';
    }
}
