# Mac 安装 OpenVINO™ Java API

&emsp;    OpenVINO™ Java API 主要基于 OpenVINO™ 和 Java 开发，支持 Mac、Windows、Linux版本

当前验证/测试环境：
- Windows:7
- Jdk:11.0.10
- OpenVINO Runtime:2023.2.0-12538-e7c1344d3c3
- OpenCV:4.8.0

## OpenVINO™ 环境配置
[OpenVINO™ ](www.openvino.ai)是一个用于优化和部署 AI 推理的开源工具包。

- 提升深度学习在计算机视觉、自动语音识别、自然语言处理和其他常见任务中的性能
- 使用流行框架（如TensorFlow，PyTorch等）训练的模型
- 减少资源需求，并在从边缘到云的一系列英特尔®平台上高效部署

下载所属电脑对应平台：
将runtime/3rdparty/tbb/lib/*.dll 拷贝至 runtime/lib/intel64/release/下
### 默认库路径安装
将runtime/lib/intel64/release/*.dll 拷贝至以下任意路径之一即可:
- 自己喜欢的路径
- C:\\Windows

## OpenCV 环境配置
在官网下载Windows 平台的OpenCV，解压缩，并加入环境变量

将opencv_java*.dll拷贝到自己喜欢的目录

### 使用

```java
//或者将build/lib/的libopencv_java*.dll 移动到自己喜欢的目录
OpenVINO.loadCvDll({opencv_java所在的目录})
```

## OpenVINO™ Java API 安装

### 源码构建
- git clone
- maven install

maven 引用
```xml
<dependencies>
    <dependency>
        <groupId>org.openvino</groupId>
        <artifactId>java-api</artifactId>
        <version>[按照最新下载的版本填入]</version>
    </dependency>
</dependencies>
```

### 在线maven 安装
** 等待1.1 版本 **  
