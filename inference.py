from pathlib import Path
from PIL import Image
import os
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

from src.inference.face_detector_onnx import FaceDetectorONNX  # sửa lại đúng path import của bạn

def main():
    detector = FaceDetectorONNX(
        onnx_path="onnx/retinaface.onnx",
        metadata_path="onnx/retinaface.json",
    )

    image_path = "datasets/val/images/0--Parade/0_Parade_marchingband_1_20.jpg"

    # detect
    detections = detector.detect(image_path)

    # draw kết quả
    result = detector.draw(image_path, detections)

    Image.fromarray(result).save("output_result2.jpg")
    print("Saved to output_result.jpg")


if __name__ == "__main__":
    main()
