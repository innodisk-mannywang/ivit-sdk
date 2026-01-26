#!/usr/bin/env python3
"""
iVIT-SDK Quick Start Example

This example demonstrates the basic usage of iVIT-SDK for object detection.
"""

import ivit
from ivit.vision import Detector


def main():
    # Print version
    print(f"iVIT-SDK Version: {ivit.__version__}")

    # List available devices
    print("\nAvailable Devices:")
    for device in ivit.list_devices():
        print(f"  - {device.id}: {device.name} ({device.backend})")

    # Get best device
    best = ivit.get_best_device()
    print(f"\nBest Device: {best.name}")

    # Create detector
    print("\nCreating detector...")
    detector = Detector("yolov8n", device="auto")
    print(f"  Model: {detector.model.name}")
    print(f"  Device: {detector.model.device}")
    print(f"  Classes: {detector.num_classes}")

    # Run inference (example - replace with actual image path)
    image_path = "test_image.jpg"

    try:
        print(f"\nRunning inference on: {image_path}")
        results = detector.predict(
            image_path,
            conf_threshold=0.5,
            iou_threshold=0.45
        )

        # Print results
        print(f"\nResults:")
        print(f"  Inference Time: {results.inference_time_ms:.2f} ms")
        print(f"  Detections: {len(results.detections)}")

        for i, det in enumerate(results.detections):
            print(f"    [{i}] {det.label}: {det.confidence:.2%}")
            print(f"        BBox: ({det.bbox.x1:.0f}, {det.bbox.y1:.0f}) - "
                  f"({det.bbox.x2:.0f}, {det.bbox.y2:.0f})")

        # Visualize results
        vis = results.visualize(save_path="output.jpg")
        print("\nVisualization saved to: output.jpg")

    except FileNotFoundError:
        print(f"\nNote: Create a test image at '{image_path}' to run inference")
        print("This example demonstrates the API usage.")


if __name__ == "__main__":
    main()
