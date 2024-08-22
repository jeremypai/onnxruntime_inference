#include <gtest/gtest.h>

#include "image_manipulate.h"

TEST(ConvertTest, ConvertHWCToCHWAndBack) {
  int width = 100;
  int height = 100;
  int channel = 3;
  float *temp = (float *)malloc(width * height * channel * sizeof(float));
  float value = 0.0F;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channel; ++c) {
        temp[h * width * channel + w * channel + c] = value;
        value = (float)((int)(value + 1.0F) % 255);
      }
    }
  }

  float *tempCHW = convertHWCToCHW(temp, width, height, channel);
  float *tempHWC = convertCHWToHWC(tempCHW, width, height, channel);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channel; ++c) {
        ASSERT_EQ(temp[h * width * channel + w * channel + c],
                  tempHWC[h * width * channel + w * channel + c]);
      }
    }
  }

  free(tempHWC);
  free(tempCHW);
  free(temp);
}

TEST(ConvertTest, ConvertCHWToHWCAndBack) {
  int width = 100;
  int height = 100;
  int channel = 3;
  float *temp = (float *)malloc(width * height * channel * sizeof(float));
  float value = 0.0F;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channel; ++c) {
        temp[c * width * height + h * width + w] = value;
        value = (float)((int)(value + 1.0F) % 255);
      }
    }
  }

  float *tempHWC = convertCHWToHWC(temp, width, height, channel);
  float *tempCHW = convertHWCToCHW(tempHWC, width, height, channel);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channel; ++c) {
        ASSERT_EQ(temp[c * width * height + h * width + w],
                  tempCHW[c * width * height + h * width + w]);
      }
    }
  }

  free(tempCHW);
  free(tempHWC);
  free(temp);
}

TEST(PNGTest, ReadGrayImageAndSave) {
  const char *savePath = "/home/onnxruntime_inferece/data/output_gray.png";
  const char *imagePath =
      "/home/onnxruntime_inferece/data/test_input_image_gray.png";
  int width;
  int height;
  unsigned char *image = readGrayImage(imagePath, &width, &height);
  ASSERT_EQ(writeGrayImage(image, savePath, width, height), 0);
  free(image);
}

TEST(PNGTest, ReadColorImageAndSave) {
  const char *savePath = "/home/onnxruntime_inferece/data/output_color.png";
  const char *imagePath =
      "/home/onnxruntime_inferece/data/test_input_image_color.png";
  int width;
  int height;
  unsigned char *image = readColorImage(imagePath, &width, &height);
  ASSERT_EQ(writeColorImage(image, savePath, width, height), 0);
  free(image);
}
