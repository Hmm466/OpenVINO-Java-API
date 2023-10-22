package org.openvino.java.model.yolo;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.List;

/**
 * Yolov8 Key point data
 */
public class Result {

    /**
     * result type
     */
    private int type;

    /**
     * Identification result class
     */
    public List<Integer> classes = new ArrayList<Integer>();

    /**
     * Confidence value
     */
    public List<Float> scores = new ArrayList<Float>();

    /**
     * Prediction box
     */
    public List<Rect> rects = new ArrayList<Rect>();

    /**
     * Split Region
     */
    public List<Mat> masks = new ArrayList<Mat>();

    /**
     * Key points of the human body
     */
    public List<PoseData> poses = new ArrayList<>();

    /**
     * object detection
     * @param score Predictiveness scores
     * @param rect Identification box
     * @param cla Identification class
     */
    public void add(float score, Rect rect, int cla) {
        type = TYPE_DETECTION;
        scores.add(score);
        rects.add(rect);
        classes.add(cla);
    }

    /**
     * object segmentation
     * @param score Predictiveness scores
     * @param rect Identification box
     * @param cla Identification class
     * @param mask Semantic segmentation results
     */
    public void add(float score, Rect rect, int cla, Mat mask) {
        type = TYPE_SEGMENTATION;
        scores.add(score);
        rects.add(rect);
        classes.add(cla);
        masks.add(mask);
    }

    /**
     * Key point prediction
     * @param score Predictiveness scores
     * @param rect Identification box
     * @param pose Key point data
     */
    public void add(float score, Rect rect, PoseData pose) {
        type = KEY_KEY_POINT_PREDICTION;
        scores.add(score);
        rects.add(rect);
        poses.add(pose);
    }

    public int getType() {
        return type;
    }

    /**
     * Get Result Length
     * @return
     */
    public int getLength(){
        return scores.size();
    }

    public static final int TYPE_DETECTION = 0;

    public static final int TYPE_SEGMENTATION = 1;

    public static final int KEY_KEY_POINT_PREDICTION = 2;
}
