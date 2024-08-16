#ifndef MODEL_EXECUTOR_H_
#define MODEL_EXECUTOR_H_

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
  int64_t inputShape[4];
  size_t inputShapeLen;
  size_t modelInputLen;
} ONNXRuntimeExecutor_Handle_t;

ONNXRuntimeExecutor_Handle_t *ONNXRuntimeExecutor_Create(const char *modelPath,
                                                         int width, int height,
                                                         int channel);

void ONNXRuntimeExecutor_Delete(ONNXRuntimeExecutor_Handle_t **executor);

int ONNXRuntimeExecutor_Inference(ONNXRuntimeExecutor_Handle_t *executor,
                                  float **outputData, float *inputData);

#ifdef __cplusplus
}
#endif

#endif // !MODEL_EXECUTOR_H_