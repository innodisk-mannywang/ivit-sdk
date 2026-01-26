#!/usr/bin/env python3
"""
iVIT-SDK Object Detection Example

This example demonstrates object detection using iVIT-SDK with various options.
"""

import ivit
from ivit.vision import Detector
from ivit.utils import Visualizer, VideoStream


def detect_image(image_path: str):
    """Detect objects in an image."""

    # Create detector
    detector = Detector("yolov8n", device="auto")
    print(f"Model: {detector.model.name}")
    print(f"Device: {detector.model.device}")
    print(f"Classes: {detector.num_classes}")

    # Run inference
    results = detector.predict(
        image_path,
        conf_threshold=0.5,
        iou_threshold=0.45,
    )

    # Print results
    print(f"\nDetection Results for: {image_path}")
    print(f"Inference Time: {results.inference_time_ms:.2f} ms")
    print(f"Objects Found: {len(results.detections)}")

    # Group by class
    class_counts = {}
    for det in results.detections:
        class_counts[det.label] = class_counts.get(det.label, 0) + 1

    print("\nObjects by Class:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label}: {count}")

    # Show detailed detections
    print("\nDetailed Detections:")
    for i, det in enumerate(results.detections):
        print(f"  [{i}] {det.label}")
        print(f"      Confidence: {det.confidence:.2%}")
        print(f"      BBox: ({det.bbox.x1:.1f}, {det.bbox.y1:.1f}) - "
              f"({det.bbox.x2:.1f}, {det.bbox.y2:.1f})")
        print(f"      Size: {det.bbox.width:.1f} x {det.bbox.height:.1f}")

    # Visualize
    vis = results.visualize()
    return vis, results


def detect_with_filters(image_path: str):
    """Detect with class filtering."""

    detector = Detector("yolov8n", device="auto")

    # Only detect specific classes (0=person, 2=car in COCO)
    results = detector.predict(
        image_path,
        conf_threshold=0.3,
        classes=[0, 2],  # Only person and car
    )

    print(f"Filtered Results (person, car only):")
    for det in results.detections:
        print(f"  {det.label}: {det.confidence:.2%}")

    return results


def detect_video(video_path: str, output_path: str = None):
    """Detect objects in video."""
    from ivit.utils import VideoWriter
    import cv2

    detector = Detector("yolov8n", device="auto")

    # Open video
    stream = VideoStream(video_path)
    print(f"Video: {stream.width}x{stream.height} @ {stream.fps:.1f} FPS")

    # Optional: create output video
    writer = None
    if output_path:
        writer = VideoWriter(
            output_path,
            fps=stream.fps,
            resolution=(stream.width, stream.height)
        )

    frame_count = 0
    total_detections = 0

    def process_frame(results, frame):
        nonlocal frame_count, total_detections

        frame_count += 1
        total_detections += len(results.detections)

        # Print progress
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {len(results.detections)} objects, "
                  f"{results.inference_time_ms:.1f}ms")

        # Visualize and write
        vis = results.visualize(frame)
        if writer:
            writer.write(vis)

        # Display (optional)
        cv2.imshow("Detection", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

    try:
        detector.predict_video(
            video_path,
            callback=process_frame,
            conf_threshold=0.5,
        )
    finally:
        stream.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames")
    print(f"Total detections: {total_detections}")
    print(f"Average: {total_detections/frame_count:.1f} objects/frame")


def compare_models():
    """Compare different detection models."""
    from ivit.utils import Profiler
    import numpy as np

    models = ["yolov8n", "yolov8s", "yolov5n"]
    profiler = Profiler()

    print("Model Comparison:")
    print("-" * 60)

    for model_name in models:
        try:
            detector = Detector(model_name, device="auto")

            report = profiler.benchmark(
                detector.model,
                input_shape=(1, 3, 640, 640),
                iterations=50,
                warmup=5
            )

            print(f"{model_name:15s}: {report.latency_mean:6.2f} ms, "
                  f"{report.throughput_fps:5.1f} FPS")

        except Exception as e:
            print(f"{model_name:15s}: Not available ({e})")


def main():
    # Image detection
    print("=" * 60)
    print("Image Detection")
    print("=" * 60)

    try:
        detect_image("test_image.jpg")
    except FileNotFoundError:
        print("Note: Create 'test_image.jpg' to run this example")

    # Filtered detection
    print("\n" + "=" * 60)
    print("Filtered Detection")
    print("=" * 60)

    try:
        detect_with_filters("test_image.jpg")
    except FileNotFoundError:
        print("Note: Create 'test_image.jpg' to run this example")

    # Model comparison
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    try:
        compare_models()
    except Exception as e:
        print(f"Model comparison skipped: {e}")


if __name__ == "__main__":
    main()
