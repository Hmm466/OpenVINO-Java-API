package org.openvino.java.utils;

public class Console {
    public static void WriteLine(String format, Object... objects) {
        System.out.println(String.format(format, objects));
    }
}
