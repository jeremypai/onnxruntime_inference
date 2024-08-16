import onnxruntime as ort
import cv2
import numpy as np

if __name__ == "__main__":
    model_path = "data/model.onnx"
    image_path = "data/test_input_image.png"

    # load the ONNX model
    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # get input and output information of model
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(
        f"input name: {input_info.name}, shape: {input_info.shape}, and type: {input_info.type}"
    )
    print(
        f"output name: {output_info.name}, shape: {output_info.shape}, and type: {output_info.type}"
    )

    # prepare input data
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    image = image[np.newaxis, np.newaxis, ...]
    image = image / 255.0
    print(f"image shape: {image.shape}, max: {image.max()}, min: {image.min()}")

    # run inference
    results = session.run([output_info.name], {input_info.name: image})
    result = results[0]
    print(result.shape)
