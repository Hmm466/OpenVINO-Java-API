package org.openvino.java.model.yolo;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;
import org.openvino.java.utils.Console;
import org.openvino.java.utils.CvUtils;
import org.openvino.java.utils.FileUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class YoloV8 {

    public static class ResultProcess {
        /// <summary>
        /// Identify Result Types
        /// </summary>
        public String[] classNames;
        /// <summary>
        /// Image information scaling ratio h, scaling ratio h, height, width
        /// </summary>
        public float[] scales;
        /// <summary>
        /// Confidence threshold
        /// </summary>
        public float scoreThreshold;
        /// <summary>
        /// Non maximum suppression threshold
        /// </summary>
        public float nmsThreshold;
        /// <summary>
        /// Number of categories
        /// </summary>
        public int categNums = 0;


        /// <summary>
        /// SegmentationResult processing class construction
        /// </summary>
        /// <param name="scales">scaling ratio h, scaling ratio h, height, width</param>
        /// <param name="score_threshold">score threshold</param>
        /// <param name="nms_threshold">nms threshold</param>
        public ResultProcess(float[] scales, int categNums) {
            this(scales,categNums,0.3f,0.5f);
        }

        public ResultProcess(float[] scales, int categNums, float scoreThreshold, float nmsThreshold) {
            this.scales = scales;
            this.scoreThreshold = scoreThreshold;
            this.nmsThreshold = nmsThreshold;
            this.categNums = categNums;
        }

        public Result processDetResult(float[] result) {
            Mat resultData = new Mat(4 + categNums, 8400, CvType.CV_32F);
            resultData.put(0,0,result);
            resultData = resultData.t();
//            result_data = result_data.T();
            // Storage results list
            List<Rect2d> positionBoxes = new ArrayList<>();
            List<Integer> classIds = new ArrayList<Integer>();
            List<Float> confidences = new ArrayList<Float>();
            // Preprocessing output results
            for (int i = 0; i < resultData.rows(); i++) {
                Mat classes_scores = resultData.row(i).colRange(4, 4 + categNums);//GetArray(i, 5, classes_scores);
                Point maxClassIdPoint, minClassIdPoint;
                double maxScore, minScore;
                // Obtain the maximum value and its position in a set of data
                Core.MinMaxLocResult result1 = Core.minMaxLoc(classes_scores);
                maxScore = result1.maxVal;
                minScore = result1.minVal;
                maxClassIdPoint = result1.maxLoc;
                minClassIdPoint = result1.minLoc;
                // Confidence level between 0 ~ 1
                // Obtain identification box information
                if (maxScore > 0.25) {
                    float cx = resultData.at(float.class,i,0).getV();
                    float cy = resultData.at(float.class,i,1).getV();
                    float ow = resultData.at(float.class,i,2).getV();
                    float oh = resultData.at(float.class,i,3).getV();
                    int x = (int)((cx - 0.5 * ow) * this.scales[0]);
                    int y = (int)((cy - 0.5 * oh) * this.scales[1]);
                    int width = (int)(ow * this.scales[0]);
                    int height = (int)(oh * this.scales[1]);
                    Rect2d box = new Rect2d();
                    box.x = x;
                    box.y = y;
                    box.width = width;
                    box.height = height;

                    positionBoxes.add(box);
                    classIds.add((int)maxClassIdPoint.x);
                    confidences.add((float)maxScore);
                }
            }
            MatOfRect2d bboxes = new MatOfRect2d();
            MatOfInt matOfInt = new MatOfInt();
            matOfInt.fromList(classIds);
            bboxes.fromList(positionBoxes);
            MatOfFloat scores = new MatOfFloat();
            scores.fromList(confidences);
            Dnn.NMSBoxes(bboxes,scores, scoreThreshold, nmsThreshold,matOfInt);
            Result reResult = new Result();
            if (!matOfInt.empty()) {
                int[] resultArray = matOfInt.toArray();
                for (int i = 0; i < resultArray.length; i++) {
                    int index = resultArray[i];
                    Rect2d rect2d = positionBoxes.get(index);
                    Rect rect = new Rect((int) rect2d.x, (int) rect2d.y, (int) rect2d.width, (int) rect2d.height);
                    reResult.add(confidences.get(index), rect, classIds.get(index).intValue());
                }
            }
            return reResult;
        }

        public void readClassNames(String path) {
            Console.WriteLine("read className:%s",path);
            classNames = FileUtils.read(path).trim().split("\n");
        }

        public void printResult(Result result){
            if (result.poses.size() != 0) {
                Console.WriteLine("\n Classification  result : \n");
                for (int i = 0; i < result.getLength(); ++i) {
                    String ss = (i + 1) + ": 1   " + result.scores.get(i) + "   " + result.rects.get(i) +"  " + result.poses.get(i);
                    Console.WriteLine(ss);
                }
                return;
            }

            if (result.masks.size() != 0)
            {
                Console.WriteLine("\n  Segmentation  result : \n");
                for (int i = 0; i < result.getLength(); ++i)
                {
                    String ss = (i + 1) + ": " + result.classes.get(i)+ "\t" + result.scores.get(i) + "   " + result.rects.get(i);
                    Console.WriteLine(ss);
                }
                return;
            }
            Console.WriteLine("\n  Detection  result : \n");
            for (int i = 0; i < result.getLength(); ++i)
            {
                String ss = (i + 1) + ": " + result.classes.get(i) + "\t" + result.scores.get(i) + "   " + result.rects.get(i);
                Console.WriteLine(ss);
            }

        }

        public Mat drawDetResult(Result result, Mat image) {
            // Draw recognition results on the image
            for (int i = 0; i < result.getLength(); i++) {
                Imgproc.rectangle(image, result.rects.get(i), new Scalar(0, 0, 255), 2, Imgproc.LINE_8);
                Imgproc.rectangle(image, new Point(result.rects.get(i).x, result.rects.get(i).y),
                        new Point(result.rects.get(i).x + result.rects.get(i).width, result.rects.get(i).y + 30), new Scalar(0, 255, 255), -1);
                Imgproc.putText(image, classNames[result.classes.get(i)] + "-" + result.scores.get(i),
                        new Point(result.rects.get(i).x, result.rects.get(i).y + 25),Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(0, 0, 0), 2);
            }
            return image;
        }

        public Result processSegResult(float[] detect, float[] proto) {
            Mat detectData = new Mat(36 + categNums, 8400, CvType.CV_32F);
            detectData.put(0,0,detect);
            Mat protoData = new Mat(32, 25600, CvType.CV_32F);
            protoData.put(0,0,proto);
            detectData = detectData.t();
            List<Rect2d> positionBoxes = new ArrayList<>();
            List<Integer> classIds = new ArrayList<>();
            List<Float> confidences = new ArrayList<>();
            List<Mat> masks = new ArrayList<>();
            for (int i = 0; i < detectData.rows(); i++) {
                Mat classesScores = detectData.row(i).colRange(4, 4 + categNums);//GetArray(i, 5, classes_scores);
                Point maxClassIdPoint, minClassIdPoint;
                double maxScore, minScore;
                Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(classesScores);
                maxClassIdPoint = minMaxLocResult.maxLoc;
                minClassIdPoint = minMaxLocResult.minLoc;
                maxScore = minMaxLocResult.maxVal;
                minScore = minMaxLocResult.minVal;

                if (maxScore > 0.25) {
                    //Console.WriteLine(max_score);
                    Mat mask = detectData.row(i).colRange(4 + categNums, categNums + 36);
                    float cx = detectData.at(float.class,i, 0).getV();
                    float cy = detectData.at(float.class,i, 1).getV();
                    float ow = detectData.at(float.class,i, 2).getV();
                    float oh = detectData.at(float.class,i, 3).getV();
                    int x = (int)((cx - 0.5 * ow) * this.scales[0]);
                    int y = (int)((cy - 0.5 * oh) * this.scales[1]);
                    int width = (int)(ow * this.scales[0]);
                    int height = (int)(oh * this.scales[1]);
                    Rect2d box = new Rect2d();
                    box.x = x;
                    box.y = y;
                    box.width = width;
                    box.height = height;

                    positionBoxes.add(box);
                    classIds.add((int)maxClassIdPoint.x);
                    confidences.add((float)maxScore);
                    masks.add(mask);
                }
            }

            MatOfRect2d bboxes = new MatOfRect2d();
            MatOfInt matOfInt = new MatOfInt();
            matOfInt.fromList(classIds);
            bboxes.fromList(positionBoxes);
            MatOfFloat scores = new MatOfFloat();
            scores.fromList(confidences);
            Dnn.NMSBoxes(bboxes,scores, scoreThreshold, nmsThreshold,matOfInt);

            Result reResult = new Result(); // Output Result Class
            // RGB images with colors
            Mat rgbMask = Mat.zeros(new Size((int)scales[3], (int)scales[2]), CvType.CV_8UC3);
            Random rd = new Random(); // Generate Random Numbers
            if (!matOfInt.empty()) {
                int[] resultArray = matOfInt.toArray();
                for (int i = 0; i < resultArray.length; i++) {
                    int index = resultArray[i];
                    // Division scope
                    Rect2d box = positionBoxes.get(index);
                    int boxX1 = (int) Math.max(0, box.x);
                    int boxY1 = (int) Math.max(0, box.y);
                    int boxX2 = (int) Math.max(0, box.x + box.width);
                    int boxY2 = (int) Math.max(0, box.y + box.height);

                    // Segmentation results
                    Mat originalMask = CvUtils.multiply(masks.get(index),protoData);
                    for (int col = 0; col < originalMask.cols(); col++) {
                        originalMask.at(float.class,0, col).setV(sigmoid(originalMask.at(float.class,0, col).getV()));
                    }
                    // 1x25600 -> 160x160 Convert to original size
                    Mat reshapeMask = originalMask.reshape(1, 160);

                    // Split size after scaling
                    int mx1 = Math.max(0, (int) ((boxX1 / scales[0]) * 0.25));
                    int mx2 = Math.max(0, (int) ((boxX2 / scales[0]) * 0.25));
                    int my1 = Math.max(0, (int) ((boxY1 / scales[1]) * 0.25));
                    int my2 = Math.max(0, (int) ((boxY2 / scales[1]) * 0.25));
                    // Crop Split Region

                    Mat mask_roi = new Mat(reshapeMask, new Range(my1, my2), new Range(mx1, mx2));
                    // Convert the segmented area to the actual size of the image
                    Mat actualMaskm = new Mat();
                    Imgproc.resize(mask_roi, actualMaskm, new Size(boxX2 - boxX1, boxY2 - boxY1));
                    // Binary segmentation region
                    for (int r = 0; r < actualMaskm.rows(); r++) {
                        for (int c = 0; c < actualMaskm.cols(); c++) {
                            float pv = actualMaskm.at(float.class,r, c).getV();
                            if (pv > 0.5) {
                                actualMaskm.at(float.class,r, c).setV(1.0f);
                            } else {
                                actualMaskm.at(float.class,r, c).setV(0.0f);
                            }
                        }
                    }
                    Mat binMask = new Mat();
                    Core.multiply(actualMaskm,new Scalar(200),actualMaskm);
                    actualMaskm.convertTo(binMask, CvType.CV_8UC1);
                    if ((boxY1 + binMask.rows()) >= scales[2]) {
                        boxY2 = (int) scales[2] - 1;
                    }
                    if ((boxX1 + binMask.cols()) >= scales[3]) {
                        boxX2 = (int) scales[3] - 1;
                    }
                    // Obtain segmentation area
                    Mat mask = Mat.zeros(new Size((int) scales[3], (int) scales[2]), CvType.CV_8UC1);
                    binMask = new Mat(binMask, new Range(0, boxY2 - boxY1), new Range(0, boxX2 - boxX1));
                    Rect roi = new Rect(boxX1, boxY1, boxX2 - boxX1, boxY2 - boxY1);
                    binMask.copyTo(new Mat(mask, roi));
                    // Color segmentation area
                    Core.add(rgbMask, new Scalar(rd.nextInt(255), rd.nextInt( 255), rd.nextInt(255)), rgbMask, mask);
                    Rect2d rect2d = positionBoxes.get(index);
                    reResult.add(confidences.get(index), new Rect((int) rect2d.x, (int) rect2d.y, (int) rect2d.width, (int) rect2d.height) , classIds.get(index), rgbMask.clone());
                }
            }
            return reResult;
        }
        /// <summary>
        /// sigmoid
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        private float sigmoid(float a) {
            float b = 1.0f / (1.0f + (float)Math.exp(-a));
            return b;
        }

        public Mat draw_seg_result(Result result, Mat image) {
            Mat masked_img = new Mat();
            // Draw recognition results on the image
            for (int i = 0; i < result.getLength(); i++) {
                Imgproc.rectangle(image, result.rects.get(i), new Scalar(0, 0, 255), 2, Imgproc.LINE_8);
                Imgproc.rectangle(image, new Point(result.rects.get(i).x, result.rects.get(i).y + 30),
                        new Point(result.rects.get(i).x, result.rects.get(i).y), new Scalar(0, 255, 255), -1);
                Imgproc.putText(image, classNames[result.classes.get(i)] + "-" + result.scores.get(i),
                        new Point(result.rects.get(i).x, result.rects.get(i).y + 25),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(0, 0, 0), 2);
                Core.addWeighted(image, 0.5, result.masks.get(i), 0.5, 0, masked_img);
            }
            return masked_img;
        }

        public Result processPoseResult(float[] result)
        {
            Mat resultData = new Mat(56, 8400, CvType.CV_32F);
            resultData.put(0,0,result);
            resultData = resultData.t();
            List<Rect2d> positionBoxes = new ArrayList<>();
            List<Float> confidences = new ArrayList<>();
            List<PoseData> pose_datas = new ArrayList<>();
            for (int i = 0; i < resultData.rows(); i++)
            {
                if (resultData.at(float.class,i, 4).getV() > 0.25)
                {
                    //Console.WriteLine(max_score);
                    float cx = resultData.at(float.class,i, 0).getV();
                    float cy = resultData.at(float.class,i, 1).getV();
                    float ow = resultData.at(float.class,i, 2).getV();
                    float oh = resultData.at(float.class,i, 3).getV();
                    int x = (int)((cx - 0.5 * ow) * this.scales[0]);
                    int y = (int)((cy - 0.5 * oh) * this.scales[1]);
                    int width = (int)(ow * this.scales[0]);
                    int height = (int)(oh * this.scales[1]);
                    Rect2d box = new Rect2d();
                    box.x = x;
                    box.y = y;
                    box.width = width;
                    box.height = height;
                    Mat pose_mat = resultData.row(i).colRange(5, 56);
                    float[] pose_data = new float[51];
                    pose_mat.get(0,0,pose_data);
                    PoseData pose = new PoseData(pose_data, this.scales);

                    positionBoxes.add(box);

                    confidences.add(resultData.at(float.class,i, 4).getV());
                    pose_datas.add(pose);
                }
            }
            int[] indexes = new int[positionBoxes.size()];
            MatOfRect2d bboxes = new MatOfRect2d();
            MatOfInt matOfInt = new MatOfInt();
            List<Integer> d = new ArrayList<>();
            for (int i = 0 ; i < indexes.length;i++) {
                d.add(0);
            }
//            matOfInt.fromList(new (indexes));
            bboxes.fromList(positionBoxes);
            MatOfFloat scores = new MatOfFloat();
            scores.fromList(confidences);
            Dnn.NMSBoxes(bboxes,scores, scoreThreshold, nmsThreshold,matOfInt);
//            Dnn.NMSBoxes(bboxes,);
//            CvDnn.NMSBoxes(position_boxes, confidences, this.score_threshold, this.nms_threshold, out indexes);

            Result reResult = new Result();
            int[] datas= matOfInt.toArray();
            for (int i = 0; i < datas.length; i++)
            {
                int index = datas[i];
                Rect2d rect2d = positionBoxes.get(index);
                reResult.add(confidences.get(index),new Rect((int) rect2d.x, (int) rect2d.y, (int) rect2d.width, (int) rect2d.height), pose_datas.get(index));
                //Console.WriteLine("rect: {0}, score: {1}", position_boxes[index], confidences[index]);
            }
            return reResult;

        }
        /// <summary>
        /// Result drawing
        /// </summary>
        /// <param name="result">recognition result</param>
        /// <param name="image">image</param>
        /// <returns></returns>
        public Mat draw_pose_result(Result result, Mat image, double visualThresh)
        {

            // 将识别结果绘制到图片上
            for (int i = 0; i < result.getLength(); i++)
            {
                Imgproc.rectangle(image, result.rects.get(i), new Scalar(0, 0, 255), 2, Imgproc.LINE_8);

                drawPoses(result.poses.get(i), image, visualThresh);
            }
            return image;
        }

        /// <summary>
        /// Key point result drawing
        /// </summary>
        /// <param name="pose">Key point data</param>
        /// <param name="image">image</param>
        public void drawPoses(PoseData pose, Mat image, double visualThresh)
        {
            // Connection point relationship
            int[][] edgs = new int[][] { { 0, 1 }, { 0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}, {6, 8},
            {7, 9}, {8, 10}, {5, 11}, {6, 12}, {11, 13}, {12, 14},{13, 15 }, {14, 16 }, {11, 12 } };
            // Color Library
            Scalar[] colors = new Scalar[] { new Scalar(255, 0, 0), new Scalar(255, 85, 0), new Scalar(255, 170, 0),
                new Scalar(255, 255, 0), new Scalar(170, 255, 0), new Scalar(85, 255, 0), new Scalar(0, 255, 0),
                new Scalar(0, 255, 85), new Scalar(0, 255, 170), new Scalar(0, 255, 255), new Scalar(0, 170, 255),
                new Scalar(0, 85, 255), new Scalar(0, 0, 255), new Scalar(85, 0, 255), new Scalar(170, 0, 255),
                new Scalar(255, 0, 255), new Scalar(255, 0, 170), new Scalar(255, 0, 85) };
            // Draw Keys
            for (int p = 0; p < 17; p++)
            {
                if (pose.score[p] < visualThresh)
                {
                    continue;
                }

                Imgproc.circle(image, pose.point.get(p), 2, colors[p], -1);
                //Console.WriteLine(pose.point[p]);
            }
            // draw
            for (int p = 0; p < 17; p++)
            {
                if (pose.score[edgs[p][0]] < visualThresh || pose.score[edgs[p][ 1]] < visualThresh)
                {
                    continue;
                }


                float[] pointX = new float[] {(float) pose.point.get(edgs[p][ 0]).x, (float) pose.point.get(edgs[p][ 1]).x};
                float[] pointY = new float[] {(float) pose.point.get(edgs[p][ 0]).y, (float) pose.point.get(edgs[p][1]).y};

                Point center_point = new Point((int)((pointX[0] + pointX[1]) / 2), (int)((pointY[0] + pointY[1]) / 2));
                double length = Math.sqrt(Math.pow((double)(pointX[0] - pointX[1]), 2.0) + Math.pow((double)(pointY[0] - pointY[1]), 2.0));
                int stick_width = 2;
                Size axis = new Size(length / 2, stick_width);
                double angle = (Math.atan2((double)(pointY[0] - pointY[1]), (double)(pointX[0] - pointX[1]))) * 180 / Math.PI;
                MatOfPoint point = new MatOfPoint();
                Imgproc.ellipse2Poly(center_point, axis, (int)angle, 0, 360, 1,point);
                Imgproc.fillConvexPoly(image, point, colors[p]);
            }
        }

        public List<IntFloatKeyValuePair> processClsResult(float[] result)
        {
            List<Float[]> newList = new ArrayList<>();
            for (int i = 0; i < result.length; i++)
            {
                newList.add(new Float[] { result[i], Float.valueOf(i)});
            }
            newList.sort((a, b) -> b[0].compareTo(a[0]));
//            new_list.sort((a,b) -> b.compareTo);
            List<IntFloatKeyValuePair> cls = new ArrayList<>();
//            KeyValuePair<int, float>[] cls = new KeyValuePair<int, float>[10];
            for (int i = 0; i < 10; ++i) {
                cls.add(new IntFloatKeyValuePair(newList.get(i)[1].intValue(), newList.get(i)[0]));
//                cls[i] = new KeyValuePair<int, float>((int)new_list[i][1], new_list[i][0]);
            }
            return cls;
        }

        // <summary>
        /// Print and output image classification results
        /// </summary>
        /// <param name="result">classification results</param>
        public void printResult(List<IntFloatKeyValuePair> result)
        {
            Console.WriteLine("\n Classification Top 10 result : \n");
            Console.WriteLine("classid probability");
            Console.WriteLine("------- -----------");
            for (int i = 0; i < 10; ++i)
            {
                Console.WriteLine("{%d}     {%f}", result.get(i).getKey(),  result.get(i).getValue());
            }
        }

    }

    public static class IntFloatKeyValuePair {
        private int key;

        private float value;

        public IntFloatKeyValuePair(int key,float value) {
            this.key = key;
            this.value = value;
        }

        public int getKey() {
            return key;
        }

        public float getValue() {
            return value;
        }
    }
}
