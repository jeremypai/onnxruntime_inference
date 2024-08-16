#include <gtest/gtest.h>

#include "image_manipulate.h"
#include "onnxruntime_executor.h"

TEST(ONNXRuntimeTest, ExampleTest) {
  const char *modelPath = "/workspaces/onnxruntime_inferece/data/model.onnx";
  const char *inputPath =
      "/workspaces/onnxruntime_inferece/data/test_input_image.png";
  const char *outputPath =
      "/workspaces/onnxruntime_inferece/data/test_input_image_output.png";
  int width = 96;
  int height = 96;
  int channel = 1;

  // initialize model and onnxruntime engine
  ONNXRuntimeExecutor_Handle_t *executor =
      ONNXRuntimeExecutor_Create(modelPath, width, height, channel);

  // prepare input image
  int imageWidth, imageHeight;
  unsigned char *imageHWC = readGrayImage(inputPath, &imageWidth, &imageHeight);
  ASSERT_EQ(width, imageWidth);
  ASSERT_EQ(height, imageHeight);

  float *imageHWCNormal = (float *)malloc(width * height * 1 * sizeof(float));
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < 1; ++c) {
        imageHWCNormal[h * width * 1 + w * 1 + c] =
            (float)imageHWC[h * width * 1 + w * 1 + c] / 127.5F - 1.0F;
      }
    }
  }

  float *imageCHWNormal =
      convertHWCToCHW(imageHWCNormal, width, height, channel);

  float *outputDataHWC;
  int ret =
      ONNXRuntimeExecutor_Inference(executor, &outputDataHWC, imageCHWNormal);
  ASSERT_EQ(ret, 0) << "Inference fail";

  unsigned char *segmentHWC =
      (unsigned char *)malloc(width * height * 3 * sizeof(unsigned char));
  memset(segmentHWC, 0, width * height * 3 * sizeof(unsigned char));
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      const float pupilChannel = outputDataHWC[h * width * 3 + w * 3 + 0];
      const float eyelidChannel = outputDataHWC[h * width * 3 + w * 3 + 1];
      const float bgChannel = outputDataHWC[h * width * 3 + w * 3 + 2];

      if (pupilChannel >= eyelidChannel && pupilChannel >= bgChannel) {
        segmentHWC[h * width * 3 + w * 3 + 0] = 255;
      } else if (eyelidChannel >= pupilChannel && eyelidChannel >= bgChannel) {
        segmentHWC[h * width * 3 + w * 3 + 1] = 255;
      } else {
        segmentHWC[h * width * 3 + w * 3 + 2] = 255;
      }
    }
  }

  ASSERT_EQ(writeColorImage(segmentHWC, outputPath, width, height), 0);

  free(segmentHWC);
  free(imageCHWNormal);
  free(imageHWCNormal);
  free(imageHWC);
  ONNXRuntimeExecutor_Delete(&executor);
}