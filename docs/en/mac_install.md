# Installing OpenVINO™ Java API on Mac

&emsp;    The OpenVINO™ Java API is mainly developed based on OpenVINO™ and Java, and supports Mac, Windows, and Linux versions.

Current verification/testing environment:
- macOS Sonoma:14.0
- Jdk:11.0.10
- OpenVINO Runtime:2023.2.0-12538-e7c1344d3c3
- OpenCV:4.8.0

## OpenVINO™ Environment Configuration
[OpenVINO™ ](www.openvino.ai)An open source toolkit for optimizing and deploying AI inference.

- Boost deep learning performance in computer vision, automatic speech recognition, natural language processing, and other common tasks
- Use models trained with popular frameworks like TensorFlow, PyTorch, and more
- Reduce resource requirements and deploy efficiently on a range of Intel® platforms, from edge to cloud

Download the appropriate platform for your computer:
Copy runtime/3rdparty/tbb/lib/*.dylib to runtime/lib/intel64/release/

### Default library path installation
Copy runtime/lib/intel64/release/*.dylib to one of the following paths:
- /System/Volumes/Preboot/Cryptexes/
- JDK directory/Contents/Home/lib/jli/
- /usr/lib/
- /usr/local/lib/

### Specify the library path for installation
Copy runtime/lib/intel64/release/*.dylib to any directory you find appropriate, such as /Users/OpenVINO/libs/

Add to the initialization code
```java
System.setProperty("java.library.path", "/User/OpenVINO/libs");
//Implement OpenVINO library loading
OpenVINO vino = OpenVINO.load("libopenvino_c.dylib");
//You can also omit it directly
OpenVINO vino = OpenVINO.load();
```
Or add in the startup vm parameters
```java
-Djava.library.path=/User/OpenVINO/libs
```

## OpenCV 环境配置
Not necessary unless it shows that the opencv function is not found

(Note: If you want to experience it directly or are not familiar with this project, you can directly use Windows to experience it, which is basically out of the box.)

### Homebrew
Use the command to edit the configuration of opencv
```shell
brew edit opencv
```
Change -DBUILD_opencv_java=OFF to-DBUILD_opencv_java=ON

Then compile
```shell
brew reinstall --build-from-source opencv
```

### Source code compilation
[opencv official website](https://opencv.org) Download source code

After decompression, enter the folder
```shell
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=OFF -DWITH_IPP=OFF -DBUILD_ZLIB=OFF -DCMAKE_INSTALL_PREFIX=OpenCVLocation
  -DJAVA_INCLUDE_PATH=JDKLocation/Contents/Home/include -DJAVA_AWT_INCLUDE_PATH=JDKLocation/Contents/Home/include 
  -DJAVA_INCLUDE_PATH2=JDKLocation/Contents/Home/include/darwin -DBUILD_JAVA=ON ../
```
Pay attention to the output.
```
--   Java:                          
--     ant:                         
--     JNI:                         
--     Java wrappers:               
--     Java tests:   
```
Need not be NO or have a directory

Then compile and install
```shell
make -j 8
make install
```


### How to use

```java
//Or move the libopencv_java*.dylib in build/lib/ to your favorite directory
OpenVINO.loadCvDll()
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
