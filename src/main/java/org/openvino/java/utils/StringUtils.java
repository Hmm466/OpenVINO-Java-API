package org.openvino.java.utils;

public class StringUtils {
    public static boolean isNullOrEmpty(String str) {
        return str == null || str.trim().length() < 1;
    }
}
