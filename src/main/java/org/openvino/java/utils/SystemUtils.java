package org.openvino.java.utils;

import java.util.Locale;

public class SystemUtils {

    public static final int SYSTEM_WINDOWS = 0;

    public static final int SYSTEM_MAC = 1;

    public static final int SYSTEM_LINUX = 2;

    public static final int SYSTEM_SUNOS = 3;

    public static final int SYSTEM_UNKNOWN = -1;

    public static int getSystemType() {
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("windows")) {
            return SYSTEM_WINDOWS;
        } else if (os.contains("mac")) {
            return SYSTEM_MAC;
        } else if (os.contains("linux")) {
            return SYSTEM_LINUX;
        } else if (os.contains("sunos")) {
            return SYSTEM_SUNOS;
        } else {
            return SYSTEM_UNKNOWN;
        }
    }
}
