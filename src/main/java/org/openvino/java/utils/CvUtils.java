package org.openvino.java.utils;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;

public class CvUtils {
    public static Mat multiply(Mat mat1, Mat mat2) {
        if (mat1.type() != mat2.type()) {
            return null;
        }
        Mat result = Mat.zeros(new Size(mat1.cols(), mat2.rows()), mat1.type()); // 创建一个大小为mat1的列数和mat2的行数的零矩阵
        if (mat1.cols() == mat2.rows()) { // 如果mat1的列数等于mat2的行数
            Core.gemm(mat1, mat2, 1, new Mat(), 0, result); // 使用Opencv的gemm函数进行矩阵乘法计算
        } else if (mat1.cols() > mat2.rows()) { // 如果mat1的列数大于mat2的行数
            Mat temp = Mat.zeros(new Size(mat1.cols(), mat2.rows()), mat1.type()); // 创建一个大小为mat1的列数和mat2的行数的零矩阵
            for (int i = 0; i < mat2.rows(); i++) {
                mat1.row(i).copyTo(temp.row(i)); // 将mat1中的第i行复制到temp中的第i行
            }
            Core.gemm(temp, mat2, 1, new Mat(), 0, result); // 使用Opencv的gemm函数进行矩阵乘法计算
        } else { // 如果mat1的列数小于mat2的行数
            Mat temp = Mat.zeros(new Size(mat1.cols(), mat2.rows()), mat2.type()); // 创建一个大小为mat1的列数和mat2的行数的零矩阵
            for (int i = 0; i < mat1.cols(); i++) {
                mat2.col(i).copyTo(temp.col(i)); // 将mat2中的第i列复制到temp中的第i列
            }
            Core.gemm(mat1, temp, 1, new Mat(), 0, result); // 使用Opencv的gemm函数进行矩阵乘法计算
        }
        return result;
    }
}
