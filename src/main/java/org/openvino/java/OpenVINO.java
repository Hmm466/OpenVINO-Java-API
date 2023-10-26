package org.openvino.java;

import com.sun.jna.ptr.PointerByReference;
import org.openvino.java.base.OpenVINOCls;
import org.openvino.java.core.VINO;
import org.openvino.java.domain.OvVersion;
import org.openvino.java.utils.StringUtils;
import org.openvino.java.utils.SystemUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

//@Data
public class OpenVINO extends OpenVINOCls {
    private static VINO core = null;

    private static boolean initialized;

    /**
     * Initialize the open vino API
     * @return
     */
    public static OpenVINO load() {
        return load(null);
    }

    /**
     *
     * @param path
     * @return
     */
    public static OpenVINO load(String path) {
        OpenVINO openVINO = new OpenVINO();
        openVINO.core = VINO.load(path);
        openVINO.initialized = true;
        return openVINO;
    }

    public static OpenVINO loadPath(String path) {
        throw new UnsupportedOperationException("The current version does not support this operation. Please wait for the new version.");
    }

    public static OpenVINO loadHttp(String url) {
        throw new UnsupportedOperationException("The current version does not support this operation. Please wait for the new version.");
    }

    public OvVersion getVersion() {
        OvVersion ovVersion = new OvVersion();
        verifyExceptionStatus(core.ov_get_openvino_version(ovVersion));
        return ovVersion;
    }

    public boolean initByXml(String xmlFile,int coreCode) {
        verifyExceptionStatus(core.ov_core_create_with_config(xmlFile,coreCode));
        return true;
    }

    private OpenVINO() {
        super("OpenVINO",null);
    }

    /**
     * Has it been initialized
     * @return Initialized will return true, otherwise false
     */
    public static boolean initialized() {
        return initialized;
    }

    public static synchronized VINO getCore() {
        if (core == null && initialized()) {
            throw new NullPointerException("The core is empty, please check if it has been initialized.");
        }
        return core;
    }

    @Override
    protected void dispose() {

    }

    public void free(PointerByReference obj) {

    }

    public static void loadCvDll(String path) {
        String fileSeparator = System.getProperty("file.separator");
        int osType = SystemUtils.getSystemType();
        if (osType != SystemUtils.SYSTEM_MAC && osType != SystemUtils.SYSTEM_WINDOWS && osType != SystemUtils.SYSTEM_LINUX) {
            throw new UnsupportedOperationException("The current API does not support your operating system");
        }
        if (path == null) {
            path = new File("").getAbsolutePath() + fileSeparator + "libs" + fileSeparator + "cv";
        }
        if (!path.endsWith(fileSeparator)) {
            path += fileSeparator;
        }
        String fileType = osType == SystemUtils.SYSTEM_WINDOWS ? "dll" : osType == SystemUtils.SYSTEM_MAC ? "dylib" : "so";
        List<Integer> targetFiles = new ArrayList<>();
        File[] files = new File(path).listFiles();
        if (files != null && files.length > 0) {
            for (File file : files) {
                if (file.getName().matches("libopencv_java\\d+\\." + fileType)) {
                    int version = getCvVersion(file.getName().trim());
                    if (version != -1) {
                        targetFiles.add(version);
                    }
                }
            }
            targetFiles.sort((a,b)->b.compareTo(a));
            if (targetFiles.size() > 0) {
                System.load(path + "libopencv_java" + targetFiles.get(0) + "." + fileType);
                return;
            }
        }
        throw new NullPointerException("Could not find opencv dll for the specified platform");
    }

    private static int getCvVersion (String name) {
        if (StringUtils.isNullOrEmpty(name)) {
            return -1;
        }
        Matcher matcher = Pattern.compile("libopencv_java(\\d+)\\.\\D+").matcher(name);
        if (matcher.find()) {
            return Integer.parseInt(matcher.group(1));
        }
        return -1;
    }

    public static void loadCvDll() {
        loadCvDll(null);
    }
}
