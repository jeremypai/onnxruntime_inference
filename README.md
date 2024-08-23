# ONNX Runtime Inference Test
+ This repository is a storage for model inference test with ONNX Runtime.

## Prerequisites
+ Prepare googletest and onnxruntime libraries:
    * Put them to thirdparty libraries
    * The CMake setting would automatically extract those libraries to build folder
+ Set paths of model for inference
+ Set paths of input image for model to do inference

## Profiling
+ Valgrind: a tool to detect memory related issues