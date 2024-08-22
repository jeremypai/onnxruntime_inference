#ifndef ONNXRUNTIME_EXECUTOR_H_
#define ONNXRUNTIME_EXECUTOR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "onnxruntime_c_api.h"

typedef struct ONNXRuntimeExecutor_Handle_t {
  OrtApi *ort;
  OrtEnv *env;
  OrtSessionOptions *sessionOptions;
  OrtSession *session;
  OrtStatus *status;
  char **inputNames;
  char **outputNames;
  int64_t inputShape[4];
  size_t inputShapeLen;
  size_t modelInputLen;
  size_t modelOutputLen;
} ONNXRuntimeExecutor_Handle_t;

ONNXRuntimeExecutor_Handle_t *
ONNXRuntimeExecutor_Create(const char *modelPath, const char **inputNames,
                           const char **outputNames, int outputHeight,
                           int outputWidth, int outputChannel, int inputHeight,
                           int inputWidth, int inputChannel);

void ONNXRuntimeExecutor_Delete(ONNXRuntimeExecutor_Handle_t **executor);

int ONNXRuntimeExecutor_Inference(ONNXRuntimeExecutor_Handle_t *executor,
                                  float *outputData, float *inputData);

#ifdef __cplusplus
}
#endif

#endif // !ONNXRUNTIME_EXECUTOR_H_