#!/usr/bin/env python3
"""
iVIT-SDK Quickstart Example

Simple example demonstrating basic usage of the iVIT-SDK Python bindings.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import ivit
import cv2


def main():
    # Print version and available devices
    print(f"iVIT-SDK v{ivit.version()}")
    print("\nDevices:")
    for d in ivit.list_devices():
        print(f"  - {d.id}: {d.name}")

    # Model and image paths
    model_path = "../../models/yolov8n.onnx"
    image_path = "../../test_images/bus.jpg"

    # Check files exist
    if not os.path.exists(model_path):
        print(f"\nModel not found: {model_path}")
        return 1
    if not os.path.exists(image_path):
        print(f"\nImage not found: {image_path}")
        return 1

    # Create detector (auto-selects best device)
    print(f"\nLoading model...")
    detector = ivit.Detector(model_path, "auto")
    print(f"  Using: {ivit.get_best_device().id}")
    print(f"  Input: {detector.input_size}")

    # Load and process image
    image = cv2.imread(image_path)
    print(f"\nProcessing: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # Run detection
    results = detector(image, conf_threshold=0.25)

    # Print results
    print(f"\nFound {results.num_detections()} objects in {results.inference_time_ms:.1f}ms:")
    for det in results.detections:
        print(f"  - {det.label}: {det.confidence:.0%}")

    # Save visualization
    output_path = "quickstart_output.jpg"
    vis = results.visualize(image)
    cv2.imwrite(output_path, vis)
    print(f"\nSaved: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
