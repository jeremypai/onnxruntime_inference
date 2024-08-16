#ifndef IMAGE_MANIPULATE_H_
#define IMAGE_MANIPULATE_H_

#ifdef __cplusplus
extern "C" {
#endif

unsigned char *readGrayImage(const char *filename, int *width, int *height);

int writeGrayImage(const unsigned char *output, const char *filename, int width,
                   int height);

unsigned char *readColorImage(const char *filename, int *width, int *height);

int writeColorImage(const unsigned char *output, const char *filename,
                    int width, int height);

float *convertHWCToCHW(const float *input, int width, int height, int channel);

float *convertCHWToHWC(const float *input, int width, int height, int channel);

#ifdef __cplusplus
}
#endif

#endif // !IMAGE_MANIPULATE_H_