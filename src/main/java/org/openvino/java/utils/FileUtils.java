package org.openvino.java.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

public class FileUtils {

    public boolean existsFile(String path) {
        File file = new File(path);
        return file.isFile() && file.exists();
    }

    /**
     * 获取文件中的内容
     *
     * @param filePath 文件路径
     * @return
     */
    public static String read(String filePath) {
        try {
            File file = new File(filePath);
            BufferedReader input = new BufferedReader(new FileReader(file));
            StringBuffer buffer = new StringBuffer();
            String text;
            while ((text = input.readLine()) != null)
                buffer.append(text + "\n");
            text = buffer.toString();
            input.close();
            return text;
        } catch (Exception e) {
            return "";
        }
    }
}
