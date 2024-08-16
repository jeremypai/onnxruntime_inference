#include "image_manipulate.h"

#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned char *readGrayImage(const char *filename, int *width, int *height) {
  png_image image; /* The control structure used by libpng */

  /* Initialize the 'png_image' structure. */
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (png_image_begin_read_from_file(&image, filename) == 0) {
    /* Something went wrong reading or writing the image.  libpng stores a
     * textual message in the 'png_image' structure:
     */
    printf("pngtopng: error: %s\n", image.message);
    return NULL;
  }

  image.format = PNG_FORMAT_GRAY;
  size_t input_data_length = PNG_IMAGE_SIZE(image);

  unsigned char *buffer = (unsigned char *)malloc(input_data_length);
  memset(buffer, 0, input_data_length);
  if (png_image_finish_read(&image, NULL /*background*/, buffer,
                            0 /*row_stride*/, NULL /*colormap*/) == 0) {
    return NULL;
  }

  *width = image.width;
  *height = image.height;

  png_image_free(&image);

  return buffer;
}

int writeGrayImage(const unsigned char *output, const char *filename, int width,
                   int height) {
  png_image image;
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.format = PNG_FORMAT_GRAY;
  image.height = height;
  image.width = width;
  int ret = 0;
  if (png_image_write_to_file(&image, filename, 0 /*convert_to_8bit*/, output,
                              0 /*row_stride*/, NULL /*colormap*/) == 0) {
    printf("write to '%s' failed:%s\n", filename, image.message);
    png_image_free(&image);
    ret = -1;
  }

  png_image_free(&image);
  return ret;
}

unsigned char *readColorImage(const char *filename, int *width, int *height) {
  png_image image; /* The control structure used by libpng */

  /* Initialize the 'png_image' structure. */
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (png_image_begin_read_from_file(&image, filename) == 0) {
    /* Something went wrong reading or writing the image.  libpng stores a
     * textual message in the 'png_image' structure:
     */
    printf("pngtopng: error: %s\n", image.message);
    return NULL;
  }

  image.format = PNG_FORMAT_BGR;
  size_t input_data_length = PNG_IMAGE_SIZE(image);

  unsigned char *buffer = (unsigned char *)malloc(input_data_length);
  memset(buffer, 0, input_data_length);
  if (png_image_finish_read(&image, NULL /*background*/, buffer,
                            0 /*row_stride*/, NULL /*colormap*/) == 0) {
    return NULL;
  }

  *width = image.width;
  *height = image.height;

  png_image_free(&image);

  return buffer;
}

int writeColorImage(const unsigned char *output, const char *filename,
                    int width, int height) {
  png_image image;
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.format = PNG_FORMAT_BGR;
  image.height = height;
  image.width = width;
  int ret = 0;
  if (png_image_write_to_file(&image, filename, 0 /*convert_to_8bit*/, output,
                              0 /*row_stride*/, NULL /*colormap*/) == 0) {
    printf("write to '%s' failed:%s\n", filename, image.message);
    png_image_free(&image);
    ret = -1;
  }

  png_image_free(&image);
  return ret;
}

float *convertHWCToCHW(const float *input, int width, int height, int channel) {
  float *output = (float *)malloc(width * height * channel * sizeof(float));
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channel; ++c) {
        output[c * width * height + h * width + w] =
            input[h * width * channel + w * channel + c];
      }
    }
  }
  return output;
}

float *convertCHWToHWC(const float *input, int width, int height, int channel) {
  float *output = (float *)malloc(width * height * channel * sizeof(float));
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channel; ++c) {
        output[h * width * channel + w * channel + c] =
            input[c * width * height + h * width + w];
      }
    }
  }
  return output;
}