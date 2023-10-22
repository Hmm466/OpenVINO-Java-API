package org.openvino.java.test;

import org.openvino.java.OpenVINO;
import org.openvino.java.core.*;
import org.openvino.java.domain.Version;

public class OpenVINOTest {
    public static void main(String[] args) {
        System.setProperty("jna.encoding","utf-8");
        String modelPath = "/Users/ming/Downloads/OpenVINO-CSharp-API-csharp3.0/model/yolov8/yolov8s-cls.xml";
        OpenVINO vino = OpenVINO.load("libopenvino_c.dylib");
        Version version = vino.getVersion();

        System.out.println(version.buildNumber + " -- " + version.description);
        Core core = new Core();
        Model model = core.readModel(modelPath);
        CompiledModel compiledModel = core.compileModel(model,"AUTO");
        println("Model name: %s",model.getFriendlyName());
        Input input = compiledModel.input();
        println("/------- [In] -------/");
        println("Input name: %s", input.getAnyName());
        println("Input type: %d", input.getElementType());
        println("Input shape: %s", input.getShape().toString());
        Output output = compiledModel.output();
        println("/------- [Out] -------/");
        println("Output name: %s", output.getAnyName());
        println("Output type: %d", output.getElementType());
        println("Output shape: %s", output.getShape().toString());
        // 创建推理请求
        InferRequest infer_request = compiledModel.createInferRequest();
        // 获取输入张量
        Tensor input_tensor = infer_request.getInputTensor();
        println("/------- [Input tensor] -------/");
        println("Input tensor type: %d", input_tensor.getElementType());
        println("Input tensor shape: %s", input_tensor.getShape().toString());
        println("Input tensor size: %d", input_tensor.getSize());
        // 读取并处理输入数据

        // 加载推理数据
        Shape input_shape = input_tensor.getShape();
        long channels = input_shape.getDims().get(1);
        long height = input_shape.getDims().get(2);
        long width = input_shape.getDims().get(3);
        float[] input_data = new float[(int)(channels * height * width)];
//        Marshal.Copy(input_mat.Ptr(0), input_data, 0, input_data.Length);
//        input_tensor.set_data(input_data);
    }

    private static void println(String format,Object ...objects) {
        System.out.println(String.format(format,objects));
    }
}
