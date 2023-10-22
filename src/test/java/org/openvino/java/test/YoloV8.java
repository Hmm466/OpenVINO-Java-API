package org.openvino.java.test;

import com.sun.jna.Pointer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.openvino.java.OpenVINO;
import org.openvino.java.core.*;
import org.openvino.java.core.Core;
import org.openvino.java.domain.Version;
import org.openvino.java.model.yolo.Result;
import org.openvino.java.utils.Console;
import org.openvino.java.utils.StringUtils;

import java.io.File;
import java.util.List;
import java.util.Map;


public class YoloV8 {

    private OpenVINO vino;
    private String classer_path = "dataset/lable/COCO_lable.txt";
    private String imgPath = "dataset/image/demo_3.jpg";
    private String modelPath = "model/yolov8/yolov8s.xml";

    @Before
    public void setUp() {
        vino = OpenVINO.load();
    }

    @After
    public void tearDown() {
    }

    @Test
    public void yoloV8Test() {
        Console.WriteLine("path:" + new File("").getAbsolutePath());
//        System.load("libs/cv/libopencv_java480.dylib");
        vino.loadCvDll();
        System.setProperty("jna.encoding","utf-8");
        Version version = vino.getVersion();
        Console.WriteLine("---- OpenVINO INFO----");
        Console.WriteLine("Description : %s", version.description);
        Console.WriteLine("Build number: %s", version.buildNumber);
        cls();
    }

    private void det() {
        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        Core core = new Core();
        // -------- Step 2. Read a model --------
        Console.WriteLine("[INFO] Loading model files: %s", modelPath);
        Model model = core.readModel(modelPath);
        printModelInfo(model);

        // -------- Step 3. Loading a model to the device --------
        CompiledModel compiled_model = core.compileModel(model, "AUTO");

        // -------- Step 4. Create an infer request --------
        InferRequest infer_request = compiled_model.createInferRequest();

        // -------- Step 5. Process input images --------
        Console.WriteLine("[INFO] Read image  files: %s", imgPath);

        Mat image = Imgcodecs.imread(imgPath);

        int max_image_length = image.cols() > image.rows() ? image.cols() : image.rows();
        Mat max_image = Mat.zeros(new Size(max_image_length, max_image_length), CvType.CV_8UC3);
        Rect roi = new Rect(0, 0, image.cols(), image.rows());
        image.copyTo(new Mat(max_image, roi));
        float[] factors = new float[4];
        factors[0] = factors[1] = (float)(max_image_length / 640.0);
        factors[2] = image.rows();
        factors[3] = image.cols();

        // -------- Step 6. Set up input --------
        Tensor input_tensor = infer_request.getInputTensor();
        Shape input_shape = input_tensor.getShape();
        Mat input_mat = Dnn.blobFromImage(max_image, 1.0 / 255.0, new Size(input_shape.getDims().get(2), input_shape.getDims().get(3)), new Scalar(0,0, 0), true, false);
        float[] input_data = new float[(int)(input_shape.getDims().get(1) * input_shape.getDims().get(2) * input_shape.getDims().get(3))];
        input_data = new Pointer(input_mat.dataAddr()).getFloatArray(0,input_data.length);
        input_tensor.setData(input_data);
        // -------- Step 7. Do inference synchronously --------
        infer_request.infer();

        Tensor outputTensor = infer_request.getOutputTensor();
        int outputLength = (int)outputTensor.getSize();
        float[] outputData = outputTensor.getData(float.class,outputLength);
        org.openvino.java.model.yolo.YoloV8.ResultProcess process = new org.openvino.java.model.yolo.YoloV8.ResultProcess(factors, 80);
        Result result = process.processDetResult(outputData);
        process.print_result(result);
        if (!StringUtils.isNullOrEmpty(classer_path)) {
            process.readClassNames(classer_path);
            Mat resultImage = process.draw_det_result(result, image);
            HighGui.imshow("result",resultImage);
            HighGui.waitKey(0);
        }
    }

    public void seq(){
//        String modelPath = "/Users/ming/Downloads/OpenVINO-CSharp-API-csharp3.0/model/yolov8/yolov8s.xml";
        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        Core core = new Core();
        // -------- Step 2. Read a model --------
        Console.WriteLine("[INFO] Loading model files: %s", modelPath);
        Model model = core.readModel(modelPath);
        printModelInfo(model);

        // -------- Step 3. Loading a model to the device --------
        CompiledModel compiled_model = core.compileModel(model, "AUTO");

        // -------- Step 4. Create an infer request --------
        InferRequest infer_request = compiled_model.createInferRequest();

        // -------- Step 5. Process input images --------
        Console.WriteLine("[INFO] Read image  files: %s", imgPath);

        Mat image = Imgcodecs.imread(imgPath);

        int max_image_length = image.cols() > image.rows() ? image.cols() : image.rows();
        Mat max_image = Mat.zeros(new Size(max_image_length, max_image_length), CvType.CV_8UC3);
        Rect roi = new Rect(0, 0, image.cols(), image.rows());
        image.copyTo(new Mat(max_image, roi));
        float[] factors = new float[4];
        factors[0] = factors[1] = (float)(max_image_length / 640.0);
        factors[2] = image.rows();
        factors[3] = image.cols();

        // -------- Step 6. Set up input --------
        Tensor input_tensor = infer_request.getInputTensor();
        Shape input_shape = input_tensor.getShape();
        Mat input_mat = Dnn.blobFromImage(max_image, 1.0 / 255.0, new Size(input_shape.getDims().get(2), input_shape.getDims().get(3)), new Scalar(0,0, 0), true, false);
        float[] input_data = new float[(int)(input_shape.getDims().get(1) * input_shape.getDims().get(2) * input_shape.getDims().get(3))];
        input_data = new Pointer(input_mat.dataAddr()).getFloatArray(0,input_data.length);
        input_tensor.setData(input_data);
        // -------- Step 7. Do inference synchronously --------
        infer_request.infer();

        Tensor output_tensor_det = infer_request.getTensor("output0");
        int output_length_det = (int)output_tensor_det.getSize();
        float[] output_data_det = output_tensor_det.getData(float[].class,output_length_det);

        Tensor output_tensor_pro = infer_request.getTensor("output1");
        int output_length_pro = (int)output_tensor_pro.getSize();
        float[] output_data_pro = output_tensor_pro.getData(float[].class,output_length_pro);

        org.openvino.java.model.yolo.YoloV8.ResultProcess process = new org.openvino.java.model.yolo.YoloV8.ResultProcess(factors, 80);
        Result result = process.process_seg_result(output_data_det, output_data_pro);

        process.print_result(result);

        if (classer_path != null)
        {
            process.readClassNames(classer_path);
            Mat result_image = process.draw_seg_result(result, image);
            HighGui.imshow("result",result_image);
            HighGui.waitKey(0);
        }
    }

    private void pose() {
        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        Core core = new Core();
        // -------- Step 2. Read a model --------
        Console.WriteLine("[INFO] Loading model files: %s", modelPath);
        Model model = core.readModel(modelPath);
        printModelInfo(model);

        // -------- Step 3. Loading a model to the device --------
        CompiledModel compiled_model = core.compileModel(model, "AUTO");

        // -------- Step 4. Create an infer request --------
        InferRequest infer_request = compiled_model.createInferRequest();

        // -------- Step 5. Process input images --------
        Console.WriteLine("[INFO] Read image  files: %s", imgPath);

        Mat image = Imgcodecs.imread(imgPath);

        int max_image_length = image.cols() > image.rows() ? image.cols() : image.rows();
        Mat max_image = Mat.zeros(new Size(max_image_length, max_image_length), CvType.CV_8UC3);
        Rect roi = new Rect(0, 0, image.cols(), image.rows());
        image.copyTo(new Mat(max_image, roi));
        float[] factors = new float[4];
        factors[0] = factors[1] = (float)(max_image_length / 640.0);
        factors[2] = image.rows();
        factors[3] = image.cols();

        // -------- Step 6. Set up input --------
        Tensor input_tensor = infer_request.getInputTensor();
        Shape input_shape = input_tensor.getShape();
        Mat input_mat = Dnn.blobFromImage(max_image, 1.0 / 255.0, new Size(input_shape.getDims().get(2), input_shape.getDims().get(3)), new Scalar(0,0, 0), true, false);
        float[] input_data = new float[(int)(input_shape.getDims().get(1) * input_shape.getDims().get(2) * input_shape.getDims().get(3))];
        input_data = new Pointer(input_mat.dataAddr()).getFloatArray(0,input_data.length);
        input_tensor.setData(input_data);
        // -------- Step 7. Do inference synchronously --------
        infer_request.infer();

        // -------- Step 9. Process output --------
        Tensor output_tensor = infer_request.getOutputTensor();
        int output_length = (int)output_tensor.getSize();
        float[] output_data = output_tensor.getData(float[].class,output_length);

        org.openvino.java.model.yolo.YoloV8.ResultProcess process = new org.openvino.java.model.yolo.YoloV8.ResultProcess(factors, 80);
        Result result = process.process_pose_result(output_data);


        Mat result_image = process.draw_pose_result(result, image, 0.2);
        process.print_result(result);
        HighGui.imshow("result",result_image);
        HighGui.waitKey(0);
    }

    public void cls() {
        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        Core core = new Core();
        // -------- Step 2. Read a model --------
        Console.WriteLine("[INFO] Loading model files: %s", modelPath);
        Model model = core.readModel(modelPath);
        printModelInfo(model);

        // -------- Step 3. Loading a model to the device --------
        CompiledModel compiled_model = core.compileModel(model, "AUTO");

        // -------- Step 4. Create an infer request --------
        InferRequest infer_request = compiled_model.createInferRequest();

        // -------- Step 5. Process input images --------
        Console.WriteLine("[INFO] Read image  files: %s", imgPath);

        Mat image = Imgcodecs.imread(imgPath);

        int max_image_length = image.cols() > image.rows() ? image.cols() : image.rows();
        Mat max_image = Mat.zeros(new Size(max_image_length, max_image_length), CvType.CV_8UC3);
        Rect roi = new Rect(0, 0, image.cols(), image.rows());
        image.copyTo(new Mat(max_image, roi));
        float[] factors = new float[4];
        factors[0] = factors[1] = (float)(max_image_length / 640.0);
        factors[2] = image.rows();
        factors[3] = image.cols();

        // -------- Step 6. Set up input --------
        Tensor input_tensor = infer_request.getInputTensor();
        Shape input_shape = input_tensor.getShape();
        Mat input_mat = Dnn.blobFromImage(max_image, 1.0 / 255.0, new Size(input_shape.getDims().get(2), input_shape.getDims().get(3)), new Scalar(0,0, 0), true, false);
        float[] input_data = new float[(int)(input_shape.getDims().get(1) * input_shape.getDims().get(2) * input_shape.getDims().get(3))];
        input_data = new Pointer(input_mat.dataAddr()).getFloatArray(0,input_data.length);
        input_tensor.setData(input_data);
        // -------- Step 7. Do inference synchronously --------
        infer_request.infer();

        Tensor output_tensor = infer_request.getOutputTensor();
        int output_length = (int)output_tensor.getSize();
        float[] output_data = output_tensor.getData(float[].class,output_length);

        org.openvino.java.model.yolo.YoloV8.ResultProcess process = new org.openvino.java.model.yolo.YoloV8.ResultProcess(factors, 80);
        List<org.openvino.java.model.yolo.YoloV8.IntFloatKeyValuePair> result = process.process_cls_result(output_data);

        process.print_result(result);
    }

    private void printModelInfo(Model model) {
        Console.WriteLine("[INFO] model name: %s", model.getFriendlyName());
        Node inputNode = model.getConstInput(0);
        Console.WriteLine("[INFO]    inputs:");
        Console.WriteLine("[INFO]      input name: %s", inputNode.getName());
        Console.WriteLine("[INFO]      input type: %s", inputNode.getType());
        Console.WriteLine("[INFO]      input shape: %s", inputNode.getShape().toString());
        inputNode.dispose();
        Node outputNode = model.getConstOutput(0);
        Console.WriteLine("[INFO]    outputs:");
        Console.WriteLine("[INFO]      output name: %s", outputNode.getName());
        Console.WriteLine("[INFO]      output type: %s", outputNode.getType());
        Console.WriteLine("[INFO]      output shape: %s", outputNode.getShape().toString());
        outputNode.dispose();
    }

    private static void println(String format,Object ...objects) {
        System.out.println(String.format(format,objects));
    }
}