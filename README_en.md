![OpenVINO™ Java API](https://socialify.git.ci/Hmm466/OpenVINO-Java-API/image?description=1&descriptionEditable=%F0%9F%92%9EOpenVINO%20Wrapper%20for%20Java%20%F0%9F%92%9E&font=Inter&forks=1&issues=1&language=1&name=1&owner=1&pattern=Circuit%20Board&pulls=1&stargazers=1&theme=Light)
<p align="center">    
    <a href="./LICENSE.txt">
    </a>    

[简体中文](README.md)| English

## 📚 Introduction

[OpenVINO™ ](www.openvino.ai)is an open source toolkit for optimizing and deploying AI inference.

- Boost deep learning performance in computer vision, automatic speech recognition, natural language processing, and other common tasks
- Use models trained with popular frameworks like TensorFlow, PyTorch, and more
- Reduce resource requirements and deploy efficiently on a range of Intel® platforms, from edge to cloud

Currently, some developers have attempted to use OpenVINO™ in Ubuntu, but this requires C++ compilation, which can be confusing and prevents out-of-the-box functionality. Therefore, this project uses JNA to implement the OpenVINO™ Java API, based on the OpenVINO™ toolkit, with the goal of promoting the use of OpenVINO™ in Java. Since the OpenVINO™ Java API is developed based on OpenVINO™, it supports the same platforms as OpenVINO™. For more information, please refer to OpenVINO™.

### Version Plan：

- 1.0: Implemented basic functions and provided a Yolov8 example.
- 1.1: Implemented Maven online installation.
- 2.0: Implemented local library loading, eliminating complex installation.
- 3.0: Implemented online loading.

(Gen AI API is under development)

### Java library disclosure：
- JNA:
- OpenCV:
- OpenVINO

## ⚙ How to install

The following article provides installation instructions for the OpenVINO™ Java API on different platforms. You can install it based on your platform.

A simplified installation requires:
- Download the runtime library for your platform from the OpenVINO official website
- Add the runtime library to your environment variables
- Windows: Place it in
- Linux/Mac OS: Place the library file in /usr/lib/

**Detailed usage documentation**

- [Mac OS](docs/en/mac_install.md)

- [Windows](docs/en/windows_install.md)

- [Linux](docs/en/linux_install.md)

## 🏷Get Started

- **Quick Experience**


- **How to use**

If you don’t know how to use it, you can learn how to use it through the following code.

```java
public class OpenVINOTest {
    
    public static void main(String[] args) {
        //Implement OpenVINO library loading.
        OpenVINO vino = OpenVINO.load("libopenvino_c.dylib");
        //If you place the library in the path directory (/usr/lib), you can shorten it like this
        //OpenVINO vino = OpenVINO.load();
        Core core = new Core();  // Initialize the Core
        Model model = core.readModel("./model.xml");  // Reading model files
        CompiledModel compiledModel = core.compiledModel(model, "AUTO");  // Load the model to the device
        InferRequest inferRequest = compiledModel.createInferRequest();  // Creating an inference channel
        Tensor inputTensor = inferRequest.getTensor("images");  // Get the input node Tensor
        inferRequest.infer();  // Model Inference
        Tensor output_tensor = inferRequest.getTensor("output0");  // Get the output node Tensor
        //Cleaning up Core unmanaged memory
        core.free();  
    }
}
```

The classes and objects encapsulated in the project, such as Core, Model, Tensor, etc., are implemented by calling the C API interface. They have unmanaged resources and need to be processed by calling the **dispose()** method, otherwise memory leaks will occur.

## 💻 Application Cases
- Deploy the Yolov8 model on the Aix development board using the OpenVINO™ Java API
- Online AI service based on Spring Boot
- Run in client mode

## Examples
The [OpenVINO-Java-API-Examples](https://github.com/Hmm466/OpenVINO-Java-API-Examples) repository contains examples using the OpenVINO-Java-API library. You can run them here.

| # | Model Name | Description | Link |
|---|---------------|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | HelloOpenVINO | Print OpenVINO version information to verify that OpenVINO can be loaded successfully. |[OpenVINOTest.java](https://github.com/Hmm466/OpenVINO-Java-API-Examples/blob/master/src/main/java/org/openvino/java/examples/OpenVINOTest.java) |
| 2 | YoloV8 | Perform seg/pose/cls inference using the YoloV8 model. | [YoloV8Test.java](https://github.com/Hmm466/OpenVINO-Java-API-Examples/blob/master/src/main/java/org/openvino/java/examples/yolo/YoloV8Test.java)                   |

## Test system
- Mac OS:Sonoma 
- Ubuntu:23.04(64)
- Windows

## 🗂 API documentation


## 🔃 Changelog


## 🎖 Contribute

&emsp;    If you are interested in using OpenVINO™ in Java and want to contribute to the open source community, please join us and develop the OpenVINO™ Java API.

&emsp;    If you have some ideas or suggestions for improvement on this project, please feel free to contact us and we will provide you with guidance on our work.

## <img title="" src="https://user-images.githubusercontent.com/48054808/157835345-f5d24128-abaf-4813-b793-d2e5bdc70e5a.png" alt="" width="40"> License

This project is released under the [Apache 2.0 license](LICENSE) license.

