#!/usr/bin/env python3
"""
iVIT-SDK Image Classification Example

This example demonstrates image classification using iVIT-SDK.
"""

import ivit
from ivit.vision import Classifier


def classify_image(image_path: str, top_k: int = 5):
    """Classify an image and print results."""

    # Create classifier
    classifier = Classifier("efficientnet-b0", device="auto")
    print(f"Model: {classifier.model.name}")
    print(f"Device: {classifier.model.device}")
    print(f"Classes: {classifier.num_classes}")
    print(f"Input Size: {classifier.input_size}")

    # Run inference
    results = classifier.predict(image_path, top_k=top_k)

    # Print results
    print(f"\nClassification Results for: {image_path}")
    print(f"Inference Time: {results.inference_time_ms:.2f} ms")
    print(f"\nTop-{top_k} Predictions:")

    for i, cls_result in enumerate(results.classifications[:top_k]):
        bar = "█" * int(cls_result.score * 20)
        print(f"  {i+1}. {cls_result.label:20s} {cls_result.score:6.2%} {bar}")

    return results


def batch_classification(image_paths: list):
    """Classify multiple images in batch."""

    classifier = Classifier("mobilenet_v3_small", device="auto")

    results_list = classifier.predict_batch(image_paths)

    for path, results in zip(image_paths, results_list):
        top1 = results.top1
        print(f"{path}: {top1.label} ({top1.score:.2%})")

    return results_list


def main():
    # Single image classification
    print("=" * 60)
    print("Single Image Classification")
    print("=" * 60)

    try:
        classify_image("test_image.jpg")
    except FileNotFoundError:
        print("Note: Create 'test_image.jpg' to run this example")

    # Batch classification example
    print("\n" + "=" * 60)
    print("Batch Classification")
    print("=" * 60)

    images = ["image1.jpg", "image2.jpg", "image3.jpg"]

    try:
        batch_classification(images)
    except FileNotFoundError:
        print("Note: Create test images to run batch classification")

    # Performance benchmark
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    from ivit.utils import Profiler
    import numpy as np

    classifier = Classifier("mobilenet_v3_small", device="auto")

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    profiler = Profiler()
    report = profiler.benchmark(
        classifier.model,
        input_shape=(1, 3, 224, 224),
        iterations=100,
        warmup=10
    )

    print(report)


if __name__ == "__main__":
    main()
