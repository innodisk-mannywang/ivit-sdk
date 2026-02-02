#!/usr/bin/env python3
"""
iVIT-SDK Quickstart Example

最簡單的使用範例，直接執行即可體驗 iVIT-SDK 的物件偵測功能。

Usage:
    python examples/python/01_quickstart.py
"""

import sys
import os

# Get project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'python'))

import ivit
from ivit.vision import Detector
import cv2


def main():
    # Print version and available devices
    print(f"iVIT-SDK v{ivit.__version__}")
    print("\nDevices:")
    for d in ivit.list_devices():
        print(f"  - {d.id}: {d.name} [{d.backend}]")

    # Model and image paths
    # NOTE: Update model_path to point to your own ONNX/OpenVINO/TensorRT model file.
    # You can download a YOLOv8n ONNX model from https://github.com/ultralytics/ultralytics
    # or use: python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
    model_path = os.environ.get(
        "IVIT_MODEL_PATH",
        os.path.join(PROJECT_ROOT, "models", "onnx", "yolov8n.onnx"),
    )
    image_path = os.environ.get(
        "IVIT_IMAGE_PATH",
        os.path.join(PROJECT_ROOT, "examples", "data", "images", "bus.jpg"),
    )

    # Check files exist
    if not os.path.exists(model_path):
        print(f"\nModel not found: {model_path}")
        return 1
    if not os.path.exists(image_path):
        print(f"\nImage not found: {image_path}")
        return 1

    # Create detector (auto select best device)
    print(f"\nLoading model...")
    device_order = ["cuda:0", "gpu:0", "cpu"]
    detector = None
    used_device = None

    for device in device_order:
        try:
            detector = Detector(model_path, device=device)
            used_device = device
            break
        except Exception as e:
            print(f"  {device}: not available ({type(e).__name__})")

    if detector is None:
        print("Error: No available device for inference")
        return 1

    print(f"  Using: {used_device}")
    print(f"  Input: {detector.input_size}")

    # Load and process image
    image = cv2.imread(image_path)
    print(f"\nProcessing: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # Run detection
    results = detector.predict(image, conf_threshold=0.5)

    # Print results
    print(f"\nFound {len(results.detections)} objects in {results.inference_time_ms:.1f}ms:")
    for det in results.detections:
        print(f"  - {det.label}: {det.confidence:.0%}")

    # Save visualization
    output_path = os.path.join(os.path.dirname(image_path), "quickstart_output.jpg")
    vis = results.visualize(image)
    cv2.imwrite(output_path, vis)
    print(f"\nSaved: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
