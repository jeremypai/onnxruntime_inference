#include <gtest/gtest.h>

#include "image_manipulate.h"
#include "onnxruntime_executor.h"

TEST(ONNXRuntimeTest, ExampleTest) {
  const char *modelPath = "/home/onnxruntime_inferece/data/model.onnx";
  const char *inputPath =
      "/home/onnxruntime_inferece/data/test_input_image.png";
  const char *outputPath =
      "/home/onnxruntime_inferece/data/test_input_image_output.png";
  const char *inputNames[] = {"input"};
  const char *outputNames[] = {"output"};
  int width = 228;
  int height = 228;
  int channel = 3;

  // initialize model and onnxruntime engine
  ONNXRuntimeExecutor_Handle_t *executor =
      ONNXRuntimeExecutor_Create(modelPath, inputNames, outputNames, height,
                                 width, channel, height, width, channel);

  // prepare input image
  int imageWidth, imageHeight;
  unsigned char *imageHWC =
      readColorImage(inputPath, &imageWidth, &imageHeight);
  ASSERT_EQ(width, imageWidth);
  ASSERT_EQ(height, imageHeight);

  float *imageHWCNormal = (float *)malloc(width * height * 1 * sizeof(float));
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < 1; ++c) {
        imageHWCNormal[h * width * 1 + w * 1 + c] =
            (float)imageHWC[h * width * 1 + w * 1 + c] / 255.0F;
      }
    }
  }

  float *imageCHWNormal =
      convertHWCToCHW(imageHWCNormal, width, height, channel);

  float *outputDataCHW =
      (float *)malloc(width * height * channel * sizeof(float));
  int ret =
      ONNXRuntimeExecutor_Inference(executor, outputDataCHW, imageCHWNormal);
  ASSERT_EQ(ret, 0) << "Inference fail";

  float *outputDataHWC = convertCHWToHWC(outputDataCHW, width, height, channel);

  unsigned char *outputDataHWCChar =
      (unsigned char *)malloc(width * height * channel * sizeof(unsigned char));

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channel; ++c) {
        outputDataHWCChar[h * width * channel + w * channel + c] =
            (unsigned char)outputDataHWC[h * width * channel + w * channel + c];
      }
    }
  }

  ASSERT_EQ(writeColorImage(outputDataHWCChar, outputPath, width, height), 0);

  free(outputDataHWCChar);
  outputDataHWCChar = NULL;
  free(outputDataHWC);
  outputDataHWC = NULL;
  free(outputDataCHW);
  outputDataCHW = NULL;
  free(imageCHWNormal);
  imageCHWNormal = NULL;
  free(imageHWCNormal);
  imageHWCNormal = NULL;
  free(imageHWC);
  imageHWC = NULL;
  ONNXRuntimeExecutor_Delete(&executor);
}