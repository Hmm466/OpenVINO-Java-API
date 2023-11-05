# Linux 安装 OpenVINO™ Java API

&emsp;    OpenVINO™ Java API 主要基于 OpenVINO™ 和 Java 开发，支持 Mac、Windows、Linux版本

当前验证/测试环境：
- Ubuntu:23.04(64)
- Jdk:1.8.0_391
- OpenVINO Runtime:2023.2.0-12538-e7c1344d3c3
- OpenCV:4.8.0

## OpenVINO™ 环境配置
[OpenVINO™ ](www.openvino.ai)是一个用于优化和部署 AI 推理的开源工具包。

- 提升深度学习在计算机视觉、自动语音识别、自然语言处理和其他常见任务中的性能
- 使用流行框架（如TensorFlow，PyTorch等）训练的模型
- 减少资源需求，并在从边缘到云的一系列英特尔®平台上高效部署

下载所属电脑对应平台： 
将runtime/3rdparty/tbb/lib/*.dylib 拷贝至 runtime/lib/intel64/release/下
### 默认库路径安装
将runtime/lib/intel64/release/*.so 拷贝至以下任意路径之一即可:
- /usr/lib/
- /usr/local/lib/

### 指定库路径安装
将runtime/lib/intel64/release/*.so 拷贝到任意自己觉得合适的目录,如/User/OpenVINO/libs/

在初始化代码中加入
```java
System.setProperty("java.library.path", "/User/OpenVINO/libs");
//实现OpenVINO 库加载
OpenVINO vino = OpenVINO.load("libopenvino_c.so");
//也可以直接省略
OpenVINO vino = OpenVINO.load();
```
或者在启动vm参数中加入
```java
-Djava.library.path=/User/OpenVINO/libs
```

## OpenCV 环境配置
64位系统非必要,除非显示opencv的函数没有找到
### 源码编译
[opencv官网](https://opencv.org) 下载源码

解压缩之后进入文件夹
```shell
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=OFF -DWITH_IPP=OFF -DBUILD_ZLIB=OFF -DCMAKE_INSTALL_PREFIX=你的opencv目录 
  -DJAVA_INCLUDE_PATH={jdk 所在位置}/include -DJAVA_AWT_INCLUDE_PATH={jdk 所在位置}/include 
  -DJAVA_INCLUDE_PATH2={jdk 所在位置}/include -DBUILD_JAVA=ON ../
```
注意看输出有没有
```
--   Java:                          
--     ant:                         
--     JNI:                         
--     Java wrappers:               
--     Java tests:   
```
需要不为NO或者有目录

然后编译安装
```shell
make -j 8
make install
```


### 使用

```java
//或者将build/lib/的libopencv_java*.so移动到自己喜欢的目录
OpenVINO.loadCvDll(CP目录/build/lib)
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
