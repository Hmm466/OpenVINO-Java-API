package org.openvino.java.model.yolo;

import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.List;

/**
 * Yolov8 Human Key Point Data
 */
public class PoseData {

    /**
     * Key point prediction score
     */
    public float[] score;

    /**
     * Key point prediction results.
     */
    public List<Point> point;

    /**
     * Default Constructor
     *
     * @param data   Key point prediction results.
     * @param scales Image scaling ratio.
     */
    public PoseData(float[] data, float[] scales) {
        score = new float[data.length];
        point = new ArrayList<>();
        for (int i = 0; i < 17; i++) {
            Point p = new Point((int) (data[3 * i] * scales[0]), (int) (data[3 * i + 1] * scales[1]));
            this.point.add(p);
            this.score[i] = data[3 * i + 2];
        }
    }

    /**
     * Convert PoseData to string.
     *
     * @return PoseData string.
     */
    @Override
    public String toString() {
        String[] point_str = new String[]{"Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
                "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist",
                "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"};
        String ss = "";
        for (int i = 0; i < point.size(); i++) {
            ss += point_str[i] + ": (" + point.get(i).x + " ," + point.get(i).y + " ," + score[i] + ") ";
        }
        return ss;
    }
}
