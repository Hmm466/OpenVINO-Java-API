# Windows Installation OpenVINO™ Java API

&emsp;    The OpenVINO™ Java API is mainly developed based on OpenVINO™ and Java, and supports Mac, Windows, and Linux versions.

Current verification/testing environment:
- Windows:7
- Jdk:11.0.10
- OpenVINO Runtime:2023.2.0-12538-e7c1344d3c3
- OpenCV:4.8.0

## OpenVINO™ Environment Configuration
[OpenVINO™ ](www.openvino.ai)An open source toolkit for optimizing and deploying AI inference.

- Boost deep learning performance in computer vision, automatic speech recognition, natural language processing, and other common tasks
- Use models trained with popular frameworks like TensorFlow, PyTorch, and more
- Reduce resource requirements and deploy efficiently on a range of Intel® platforms, from edge to cloud

Download the corresponding computer platform:
Copy runtime/3rdparty/tbb/lib/*.dll to runtime/lib/intel64/release/
### Default Library Path Installation
Copy runtime/lib/intel64/release/*.dll to any of the following paths:
- Your preferred path
- C:\\Windows

## OpenCV Environment Configuration
Download OpenCV for Windows from the official website, unzip it, and add it to your environment variables.

Copy the opencv_java*.dll files to a directory of your choice.

(The project will be carried by default.)

### How to use

```java
//Or move the libopencv_java*.dll in build/lib/ to your favorite directory
OpenVINO.loadCvDll({The directory where opencv_java is located})
```

## OpenVINO™ Java API Installation

### Source code build
- git clone
- maven install

maven References
```xml
<dependencies>
    <dependency>
        <groupId>org.openvino</groupId>
        <artifactId>java-api</artifactId>
        <version>[Fill in according to the latest downloaded version]</version>
    </dependency>
</dependencies>
```

### Online Maven installation
** Waiting for version 1.1 **  
