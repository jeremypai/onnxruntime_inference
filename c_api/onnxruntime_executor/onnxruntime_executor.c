#include "model_executor.h"

#include "stdio.h"

int ONNXRuntime_Status_Check(ONNXRuntimeExecutor_Handle_t *executor) {
  if (executor->status != NULL) {
    const char *msg = executor->ort->GetErrorMessage(executor->status);
    fprintf(stderr, "%s\n", msg);
    executor->ort->ReleaseStatus(executor->status);
    ONNXRuntimeExecutor_Delete(&executor);
    return 1;
  }
  return 0;
}

void enableCuda(ONNXRuntimeExecutor_Handle_t *executor) {
  // OrtCUDAProviderOptions is a C struct. C programming language doesn't have
  // constructors/destructors.
  OrtCUDAProviderOptions o;
  // Here we use memset to initialize every field of the above data struct to
  // zero.
  memset(&o, 0, sizeof(o));
  // But is zero a valid value for every variable? Not quite. It is not
  // guaranteed. In the other words: does every enum type contain zero? The
  // following line can be omitted because EXHAUSTIVE is mapped to zero in
  // onnxruntime_c_api.h.
  o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  o.gpu_mem_limit = SIZE_MAX;

  executor->status = executor->ort->SessionOptionsAppendExecutionProvider_CUDA(
      executor->sessionOptions, &o);
}

ONNXRuntimeExecutor_Handle_t *ONNXRuntimeExecutor_Create(const char *modelPath,
                                                         int width, int height,
                                                         int channel) {
  ONNXRuntimeExecutor_Handle_t *executor =
      (ONNXRuntimeExecutor_Handle_t *)malloc(
          sizeof(ONNXRuntimeExecutor_Handle_t));
  executor->ort = NULL;
  executor->env = NULL;
  executor->sessionOptions = NULL;
  executor->session = NULL;
  executor->status = NULL;
  executor->inputShape[0] = 1;
  executor->inputShape[1] = channel;
  executor->inputShape[2] = height;
  executor->inputShape[3] = width;
  executor->inputShapeLen =
      sizeof(executor->inputShape) / sizeof(executor->inputShape[0]);
  executor->modelInputLen = width * height * channel * sizeof(float);

  // init ONNX Runtime engine
  executor->ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  // init env
  executor->status = executor->ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR,
                                              "ONNXRuntime", &executor->env);
  if (ONNXRuntime_Status_Check(executor)) {
    return NULL;
  }

  // init session option
  executor->status =
      executor->ort->CreateSessionOptions(&executor->sessionOptions);
  if (ONNXRuntime_Status_Check(executor)) {
    return NULL;
  }

  // init execution provider
  enableCuda(executor);
  if (ONNXRuntime_Status_Check(executor)) {
    printf("CUDA is not available\n");
    return NULL;
  }
  printf("CUDA is enabled\n");

  // init session
  executor->status = executor->ort->CreateSession(
      executor->env, modelPath, executor->sessionOptions, &executor->session);
  if (ONNXRuntime_Status_Check(executor)) {
    return NULL;
  }

  // verify input and output count
  size_t count;
  executor->status =
      executor->ort->SessionGetInputCount(executor->session, &count);
  printf("model input counts: %ld\n", count);
  executor->status =
      executor->ort->SessionGetOutputCount(executor->session, &count);
  printf("model output counts: %ld\n", count);

  return executor;
}

void ONNXRuntimeExecutor_Delete(ONNXRuntimeExecutor_Handle_t **executor) {
  if (*executor == NULL) {
    return;
  }

  if ((*executor)->session) {
    (*executor)->ort->ReleaseSession((*executor)->session);
  }

  if ((*executor)->sessionOptions) {
    (*executor)->ort->ReleaseSessionOptions((*executor)->sessionOptions);
  }

  if ((*executor)->env) {
    (*executor)->ort->ReleaseEnv((*executor)->env);
  }

  if ((*executor)->status) {
    (*executor)->ort->ReleaseStatus((*executor)->status);
  }

  (*executor)->status = NULL;
  (*executor)->ort = NULL;

  free(*executor);
  *executor = NULL;
}

int ONNXRuntimeExecutor_Inference(ONNXRuntimeExecutor_Handle_t *executor,
                                  float **outputData, float *inputData) {
  OrtMemoryInfo *memoryInfo;
  executor->status = executor->ort->CreateCpuMemoryInfo(
      OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo);
  if (ONNXRuntime_Status_Check(executor)) {
    printf(
        "ONNXRuntimeExecutor_Inference fail: memoryInfo fail to initialize\n");
    return 1;
  }

  OrtValue *inputTensor = NULL;
  executor->status = executor->ort->CreateTensorWithDataAsOrtValue(
      memoryInfo, inputData, executor->modelInputLen, executor->inputShape,
      executor->inputShapeLen, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &inputTensor);
  if (ONNXRuntime_Status_Check(executor)) {
    printf(
        "ONNXRuntimeExecutor_Inference fail: inputTensor fail to initialize\n");
    return 1;
  }

  int isTensor;
  executor->status = executor->ort->IsTensor(inputTensor, &isTensor);
  if (ONNXRuntime_Status_Check(executor)) {
    printf("ONNXRuntimeExecutor_Inference fail: inputTensor is not a tensor "
           "type\n");
    return 1;
  }

  executor->ort->ReleaseMemoryInfo(memoryInfo);

  OrtValue *outputTensor = NULL;
  static const char *inputNames[] = {"input"};
  static const char *outputNames[] = {"output"};
  executor->status = executor->ort->Run(executor->session, NULL, inputNames,
                                        (const OrtValue *const *)&inputTensor,
                                        1, outputNames, 1, &outputTensor);
  if (ONNXRuntime_Status_Check(executor)) {
    printf("ONNXRuntimeExecutor_Inference fail: inference fail\n");
    return 1;
  }

  executor->status = executor->ort->IsTensor(outputTensor, &isTensor);
  if (ONNXRuntime_Status_Check(executor)) {
    printf("ONNXRuntimeExecutor_Inference fail: outputTensor is not a tensor "
           "type\n");
    return 1;
  }

  float *outputTensorData = NULL;
  executor->status = executor->ort->GetTensorMutableData(
      outputTensor, (void **)&outputTensorData);
  if (ONNXRuntime_Status_Check(executor)) {
    printf("ONNXRuntimeExecutor_Inference fail: outputTensorData fail to "
           "initialize\n");
    return 1;
  }

  *outputData = outputTensorData;

  executor->ort->ReleaseValue(outputTensor);
  executor->ort->ReleaseValue(inputTensor);

  return 0;
}