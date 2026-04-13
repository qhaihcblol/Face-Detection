import cv2
import time

from src.inference.face_detector_onnx import FaceDetectorONNX


def main() -> None:
    detector = FaceDetectorONNX(
        onnx_path="onnx/retinaface2.onnx",
        metadata_path="onnx/retinaface.json",
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (camera index 0)")

    print("Webcam started. Press 'q' or 'Esc' to exit.")

    prev_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read frame from webcam.")
                break

            detections = detector.detect(frame, assume_bgr=True)
            # OpenCV drawing ops need a writable contiguous array.
            result = detector.draw(frame, detections, assume_bgr=True).copy()

            cv2.putText(
                result,
                f"Detections: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            current_time = time.perf_counter()
            dt = current_time - prev_time
            prev_time = current_time
            if dt > 0:
                instant_fps = 1.0 / dt
                # Smooth FPS to avoid noisy instantaneous fluctuations.
                fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)

            cv2.putText(
                result,
                f"FPS: {fps:.1f}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("RetinaFace Webcam", result)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
