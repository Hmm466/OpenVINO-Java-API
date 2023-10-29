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
import org.openvino.java.domain.OvVersion;
import org.openvino.java.model.yolo.Result;
import org.openvino.java.model.yolo.YoloV8;
import org.openvino.java.utils.Console;
import org.openvino.java.utils.StringUtils;

import java.io.File;
import java.util.List;


public class YoloV8Test {

    private OpenVINO vino;
    private String classer_path = "dataset/lable/COCO_lable.txt";
    private String imgPath = "dataset/image/demo_2.jpg";
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
        vino.loadCvDll();
        System.setProperty("jna.encoding", "utf-8");
        OvVersion version = vino.getVersion();
        Console.WriteLine("---- OpenVINO INFO----");
        Console.WriteLine("Description : %s", version.description);
        Console.WriteLine("Build number: %s", version.buildNumber);
        seg();
    }

    private void det() {
        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        Core core = new Core();
        // -------- Step 2. Read a model --------
        Console.WriteLine("[INFO] Loading model files: %s", modelPath);
        Model model = core.readModel(modelPath);
        printModelInfo(model);

        // -------- Step 3. Loading a model to the device --------
        CompiledModel compiledModel = core.compileModel(model, "AUTO");

        // -------- Step 4. Create an infer request --------
        InferRequest inferRequest = compiledModel.createInferRequest();

        // -------- Step 5. Process input images --------
        Console.WriteLine("[INFO] Read image  files: %s", imgPath);

        Mat image = Imgcodecs.imread(imgPath);

        int maxImageLength = image.cols() > image.rows() ? image.cols() : image.rows();
        Mat maxImage = Mat.zeros(new Size(maxImageLength, maxImageLength), CvType.CV_8UC3);
        Rect roi = new Rect(0, 0, image.cols(), image.rows());
        image.copyTo(new Mat(maxImage, roi));
        float[] factors = new float[4];
        factors[0] = factors[1] = (float) (maxImageLength / 640.0);
        factors[2] = image.rows();
        factors[3] = image.cols();

        // -------- Step 6. Set up input --------
        Tensor inputTensor = inferRequest.getInputTensor();
        Shape inputShape = inputTensor.getShape();
        Mat inputMat = Dnn.blobFromImage(maxImage, 1.0 / 255.0, new Size(inputShape.getDims().get(2), inputShape.getDims().get(3)), new Scalar(0, 0, 0), true, false);
        float[] inputData = new float[(int) (inputShape.getDims().get(1) * inputShape.getDims().get(2) * inputShape.getDims().get(3))];
        inputData = new Pointer(inputMat.dataAddr()).getFloatArray(0, inputData.length);
        inputTensor.setData(inputData);
        // -------- Step 7. Do inference synchronously --------
        inferRequest.infer();

        Tensor outputTensor = inferRequest.getOutputTensor();
        int outputLength = (int) outputTensor.getSize();
        float[] outputData = outputTensor.getData(float.class, outputLength);
        YoloV8 process = new YoloV8(factors, 80);
        Result result = process.processDetResult(outputData);
        process.printResult(result);
        if (!StringUtils.isNullOrEmpty(classer_path)) {
            process.readClassNames(classer_path);
            Mat resultImage = process.drawDetResult(result, image);
            HighGui.imshow("result", resultImage);
            HighGui.waitKey(0);
        }
    }

    public void seg() {
        String modelPath = new File("").getAbsolutePath() + "/model/yolov8/yolov8s-seg.xml";
        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        Core core = new Core();
        // -------- Step 2. Read a model --------
        Console.WriteLine("[INFO] Loading model files: %s", modelPath);
        Model model = core.readModel(modelPath);
        printModelInfo(model);

        // -------- Step 3. Loading a model to the device --------
        CompiledModel compileModel = core.compileModel(model, "AUTO");

        // -------- Step 4. Create an infer request --------
        InferRequest inferRequest = compileModel.createInferRequest();

        // -------- Step 5. Process input images --------
        Console.WriteLine("[INFO] Read image  files: %s", imgPath);

        Mat image = Imgcodecs.imread(imgPath);

        int maxImageLength = image.cols() > image.rows() ? image.cols() : image.rows();
        Mat maxImage = Mat.zeros(new Size(maxImageLength, maxImageLength), CvType.CV_8UC3);
        Rect roi = new Rect(0, 0, image.cols(), image.rows());
        image.copyTo(new Mat(maxImage, roi));
        float[] factors = new float[4];
        factors[0] = factors[1] = (float) (maxImageLength / 640.0);
        factors[2] = image.rows();
        factors[3] = image.cols();

        // -------- Step 6. Set up input --------
        Tensor inputTensor = inferRequest.getInputTensor();
        Shape input_shape = inputTensor.getShape();
        Mat input_mat = Dnn.blobFromImage(maxImage, 1.0 / 255.0, new Size(input_shape.getDims().get(2), input_shape.getDims().get(3)), new Scalar(0, 0, 0), true, false);
        float[] inputData = new float[(int) (input_shape.getDims().get(1) * input_shape.getDims().get(2) * input_shape.getDims().get(3))];
        inputData = new Pointer(input_mat.dataAddr()).getFloatArray(0, inputData.length);
        inputTensor.setData(inputData);
        // -------- Step 7. Do inference synchronously --------
        inferRequest.infer();

        Tensor outputTensorDet = inferRequest.getTensor("output0");
        int outputLengthDet = (int) outputTensorDet.getSize();
        float[] outputDataDet = outputTensorDet.getData(float[].class, outputLengthDet);

        Tensor outputTensorPro = inferRequest.getTensor("output1");
        int outputLengthPro = (int) outputTensorPro.getSize();
        float[] outputDataPro = outputTensorPro.getData(float[].class, outputLengthPro);

        YoloV8 process = new YoloV8(factors, 80);
        Result result = process.processSegResult(outputDataDet, outputDataPro);

        process.printResult(result);

        if (classer_path != null) {
            process.readClassNames(classer_path);
            Mat resultImage = process.drawSegResult(result, image);
            HighGui.imshow("result", resultImage);
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
        InferRequest inferRequest = compiled_model.createInferRequest();

        // -------- Step 5. Process input images --------
        Console.WriteLine("[INFO] Read image  files: %s", imgPath);

        Mat image = Imgcodecs.imread(imgPath);

        int maxImageLength = image.cols() > image.rows() ? image.cols() : image.rows();
        Mat maxImage = Mat.zeros(new Size(maxImageLength, maxImageLength), CvType.CV_8UC3);
        Rect roi = new Rect(0, 0, image.cols(), image.rows());
        image.copyTo(new Mat(maxImage, roi));
        float[] factors = new float[4];
        factors[0] = factors[1] = (float) (maxImageLength / 640.0);
        factors[2] = image.rows();
        factors[3] = image.cols();

        // -------- Step 6. Set up input --------
        Tensor inputTensor = inferRequest.getInputTensor();
        Shape inputShape = inputTensor.getShape();
        Mat inputMat = Dnn.blobFromImage(maxImage, 1.0 / 255.0, new Size(inputShape.getDims().get(2), inputShape.getDims().get(3)), new Scalar(0, 0, 0), true, false);
        float[] inputData = new float[(int) (inputShape.getDims().get(1) * inputShape.getDims().get(2) * inputShape.getDims().get(3))];
        inputData = new Pointer(inputMat.dataAddr()).getFloatArray(0, inputData.length);
        inputTensor.setData(inputData);
        // -------- Step 7. Do inference synchronously --------
        inferRequest.infer();

        // -------- Step 9. Process output --------
        Tensor outputTensor = inferRequest.getOutputTensor();
        int outputLength = (int) outputTensor.getSize();
        float[] outputData = outputTensor.getData(float[].class, outputLength);

        YoloV8 process = new YoloV8(factors, 80);
        Result result = process.processPoseResult(outputData);


        Mat resultImage = process.drawPoseResult(result, image, 0.2);
        process.printResult(result);
        HighGui.imshow("result", resultImage);
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
        CompiledModel compiledModel = core.compileModel(model, "AUTO");

        // -------- Step 4. Create an infer request --------
        InferRequest inferRequest = compiledModel.createInferRequest();

        // -------- Step 5. Process input images --------
        Console.WriteLine("[INFO] Read image  files: %s", imgPath);

        Mat image = Imgcodecs.imread(imgPath);

        int maxImageLength = image.cols() > image.rows() ? image.cols() : image.rows();
        Mat maxImage = Mat.zeros(new Size(maxImageLength, maxImageLength), CvType.CV_8UC3);
        Rect roi = new Rect(0, 0, image.cols(), image.rows());
        image.copyTo(new Mat(maxImage, roi));
        float[] factors = new float[4];
        factors[0] = factors[1] = (float) (maxImageLength / 640.0);
        factors[2] = image.rows();
        factors[3] = image.cols();

        // -------- Step 6. Set up input --------
        Tensor inputTensor = inferRequest.getInputTensor();
        Shape inputShape = inputTensor.getShape();
        Mat inputMat = Dnn.blobFromImage(maxImage, 1.0 / 255.0, new Size(inputShape.getDims().get(2), inputShape.getDims().get(3)), new Scalar(0, 0, 0), true, false);
        float[] inputData = new float[(int) (inputShape.getDims().get(1) * inputShape.getDims().get(2) * inputShape.getDims().get(3))];
        inputData = new Pointer(inputMat.dataAddr()).getFloatArray(0, inputData.length);
        inputTensor.setData(inputData);
        // -------- Step 7. Do inference synchronously --------
        inferRequest.infer();

        Tensor outputTensor = inferRequest.getOutputTensor();
        int outputLength = (int) outputTensor.getSize();
        float[] outputData = outputTensor.getData(float[].class, outputLength);

        YoloV8 process = new YoloV8(factors, 80);
        List<org.openvino.java.model.yolo.YoloV8.IntFloatKeyValuePair> result = process.processClsResult(outputData);

        process.printResult(result);
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

    private static void println(String format, Object... objects) {
        System.out.println(String.format(format, objects));
    }
}