
<p align="center">    
    <a href="./LICENSE.txt">
    </a>    

简体中文| [English](README.md)

## 📚 简介

[OpenVINO™ ](www.openvino.ai)是一个用于优化和部署 AI 推理的开源工具包。

- 提升深度学习在计算机视觉、自动语音识别、自然语言处理和其他常见任务中的性能
- 使用流行框架（如TensorFlow，PyTorch等）训练的模型
- 减少资源需求，并在从边缘到云的一系列英特尔®平台上高效部署

目前有开发者以实现Ubuntu下使用OpenVINO™方法，但是却需要通过C++编译等，带来一些使用困惑，不能达到开箱即用的效果，所以该项目才用JNA实现基于OpenVINO™工具套件推出的OpenVINO™ Java API，旨在推动 OpenVINO™在Java领域的应用。OpenVINO™ Java API 由于是基于 OpenVINO™ 开发，所支持的平台与OpenVINO™ 一致，具体信息可以参考 OpenVINO™。

发版计划：

- 1.0: 实现基本函数，并提供Yolov8范例
- 1.1: 实现maven 在线安装
- 2.0: 实现库本地加载，告别复杂安装.
- 3.0: 实现在线加载

Java库公示：
- JNA:
- opencv:
- 
## ⚙ 如何安装

以下文章提供了OpenVINO™ Java API在不同平台的安装方法，可以根据自己使用平台进行安装。

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
        //实现OpenVINO 库加载
        OpenVINO vino = OpenVINO.load("libopenvino_c.dylib");
        Core core = new Core();  // 初始化 Core 核心
        Model model = core.read_model("./model.xml");  // 读取模型文件
        CompiledModel compiled_model = core.compiled_model(model, "AUTO");  // 将模型加载到设备
        InferRequest infer_request = compiled_model.create_infer_request();  // 创建推理通道
        Tensor input_tensor = infer_request.get_tensor("images");  // 获取输入节点Tensor
        infer_request.infer();  // 模型推理
        Tensor output_tensor = infer_request.get_tensor("output0");  // 获取输出节点Tensor
        //清理 Core 非托管内存
        core.free();  
    }
}
```

项目中所封装的类、对象例如Core、Model、Tensor等，通过调用 C api 接口实现，具有非托管资源，需要调用**dispose()**方法处理，否则就会出现内存泄漏。



## 💻 应用案例


## 🗂 API 文档


## 🔃 更新日志

## 🎖 贡献

&emsp;    如果您对OpenVINO™ 在Java使用感兴趣，有兴趣对开源社区做出自己的贡献，欢迎加入我们，一起开发OpenVINO™ Java API。

&emsp;    如果你对该项目有一些想法或改进思路，欢迎联系我们，指导下我们的工作。

## <img title="" src="https://user-images.githubusercontent.com/48054808/157835345-f5d24128-abaf-4813-b793-d2e5bdc70e5a.png" alt="" width="40"> 许可证书

本项目的发布受[Apache 2.0 license](LICENSE)许可认证。

