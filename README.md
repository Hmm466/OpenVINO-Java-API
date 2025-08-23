![OpenVINO™ Java API](https://socialify.git.ci/Hmm466/OpenVINO-Java-API/image?description=1&descriptionEditable=%F0%9F%92%9EOpenVINO%20Wrapper%20for%20Java%20%F0%9F%92%9E&font=Inter&forks=1&issues=1&language=1&name=1&owner=1&pattern=Circuit%20Board&pulls=1&stargazers=1&theme=Light)
<p align="center">    
    <a href="./LICENSE.txt">
    </a>    

简体中文| [English](README_en.md)

## 📚 简介

[OpenVINO™ ](www.openvino.ai)是一个用于优化和部署 AI 推理的开源工具包。

- 提升深度学习在计算机视觉、自动语音识别、自然语言处理和其他常见任务中的性能
- 使用流行框架（如TensorFlow，PyTorch等）训练的模型
- 减少资源需求，并在从边缘到云的一系列英特尔®平台上高效部署

目前有开发者以实现Ubuntu下使用OpenVINO™方法，但是却需要通过C++编译等，带来一些使用困惑，不能达到开箱即用的效果，所以该项目才用JNA实现基于OpenVINO™工具套件推出的OpenVINO™ Java API，旨在推动 OpenVINO™在Java领域的应用。OpenVINO™ Java API 由于是基于 OpenVINO™ 开发，所支持的平台与OpenVINO™ 一致，具体信息可以参考 OpenVINO™。

### 版本计划：

- 1.0: 实现基本函数，并提供Yolov8范例
- 1.1: 实现maven 在线安装
- 2.0: 实现库本地加载，告别复杂安装.
- 3.0: 实现在线加载

（Gen AI API 正在开发中）

### Java库公示：
- JNA:
- OpenCV:
- OpenVINO

## ⚙ 如何安装

以下文章提供了OpenVINO™ Java API在不同平台的安装方法，可以根据自己使用平台进行安装。
简短安装既要：
- 在OpenVINO官网下载对应平台的runtime库
- 将Runtime库加入环境变量
- Windows: 放在
- Linux/Mac OS: 将库文件放入/usr/lib/

**详细使用文档**

- [Mac OS](docs/cn/mac_install.md)

- [Windows](docs/cn/windows_install.md)

- [Linux](docs/cn/linux_install.md)

## 🏷开始使用

- **快速体验**


- **使用方法**

如果你不知道如何使用，通过下面代码简单了解使用方法。

```java
public class OpenVINOTest {
    
    public static void main(String[] args) {
        //实现OpenVINO 库加载.
        OpenVINO vino = OpenVINO.load("libopenvino_c.dylib");
        //如果将库放置path目录(/usr/lib)可以这样简写
        //OpenVINO vino = OpenVINO.load();
        Core core = new Core();  // 初始化 Core 核心
        Model model = core.readModel("./model.xml");  // 读取模型文件
        CompiledModel compiledModel = core.compiledModel(model, "AUTO");  // 将模型加载到设备
        InferRequest inferRequest = compiledModel.createInferRequest();  // 创建推理通道
        Tensor inputTensor = inferRequest.getTensor("images");  // 获取输入节点Tensor
        inferRequest.infer();  // 模型推理
        Tensor output_tensor = inferRequest.getTensor("output0");  // 获取输出节点Tensor
        //清理 Core 非托管内存
        core.free();
    }
}
```

项目中所封装的类、对象例如Core、Model、Tensor等，通过调用 C api 接口实现，具有非托管资源，需要调用**dispose()**方法处理，否则就会出现内存泄漏。

## 💻 应用案例
- 爱克斯开发板使用OpenVINO™ Java API部署Yolov8模型
- 基于Spring Boot 在线AI服务
- 基于客服端模式运行

## 示例
[OpenVINO-Java-API-Examples](https://github.com/Hmm466/OpenVINO-Java-API-Examples) 仓库包含使用 OpenVINO-Java-API 库的示例。您可以点击此处进行运行。

| # | 模型名称          | 描述                                  | Link                  |
|---|---------------|-------------------------------------|-----------------------|
| 1 | HelloOpenVINO | 打印OpenVINO的版本信息，验证OpenVINO 是否能够正常加载 | [OpenVINOTest.java](https://github.com/Hmm466/OpenVINO-Java-API-Examples/blob/master/src/main/java/org/openvino/java/examples/OpenVINOTest.java) |
| 2 | YoloV8        | 使用YoloV8模型进行seg/pose/cls 推理         |    [YoloV8Test.java](https://github.com/Hmm466/OpenVINO-Java-API-Examples/blob/master/src/main/java/org/openvino/java/examples/yolo/YoloV8Test.java)                   |


## 测试系统
- Mac OS:Sonoma 
- Ubuntu:23.04(64)
- Windows

## 🗂 API 文档


## 🔃 更新日志


## 🎖 贡献

&emsp;    如果您对OpenVINO™ 在Java使用感兴趣，有兴趣对开源社区做出自己的贡献，欢迎加入我们，一起开发OpenVINO™ Java API。

&emsp;    如果你对该项目有一些想法或改进思路，欢迎联系我们，指导下我们的工作。

## <img title="" src="https://user-images.githubusercontent.com/48054808/157835345-f5d24128-abaf-4813-b793-d2e5bdc70e5a.png" alt="" width="40"> 许可证书

本项目的发布受[Apache 2.0 license](LICENSE)许可认证。

