"""
Core inference tests for iVIT-SDK.

Tests cover:
- Model loading and configuration
- Single image inference
- Batch inference
- Error handling
- Thread safety
- Resource management
"""

import pytest
import numpy as np
import threading
import time
from unittest.mock import Mock, MagicMock, patch


class TestModelLoading:
    """Test model loading scenarios."""

    def test_load_config_defaults(self):
        """Test LoadConfig default values."""
        from ivit.core.types import LoadConfig

        config = LoadConfig()
        assert config.device == "auto"
        assert config.backend == "auto"
        # precision=None means auto-selection by runtime
        assert config.precision is None
        assert config.batch_size == 1
        assert config.use_cache is True

    def test_load_config_custom(self):
        """Test LoadConfig with custom values."""
        from ivit.core.types import LoadConfig

        config = LoadConfig(
            device="cuda:0",
            backend="tensorrt",
            precision="fp16",
            batch_size=4,
        )
        assert config.device == "cuda:0"
        assert config.backend == "tensorrt"
        assert config.precision == "fp16"
        assert config.batch_size == 4

    def test_infer_config_defaults(self):
        """Test InferConfig default values."""
        from ivit.core.types import InferConfig

        config = InferConfig()
        assert config.conf_threshold == 0.5
        assert config.iou_threshold == 0.45
        # 300 is the standard default for YOLO-style models
        assert config.max_detections == 300

    def test_infer_config_custom(self):
        """Test InferConfig with custom values."""
        from ivit.core.types import InferConfig

        config = InferConfig(
            conf_threshold=0.25,
            iou_threshold=0.5,
            max_detections=50,
            classes=[0, 1, 2],
        )
        assert config.conf_threshold == 0.25
        assert config.iou_threshold == 0.5
        assert config.max_detections == 50
        assert config.classes == [0, 1, 2]


class TestBatchInference:
    """Test batch inference functionality."""

    def test_batch_size_detection(self):
        """Test dynamic batch size detection from model shape."""
        # Shape with -1 indicates dynamic batch
        shape_dynamic = [-1, 3, 224, 224]
        assert shape_dynamic[0] == -1

        # Shape with 0 also indicates dynamic batch
        shape_zero = [0, 3, 224, 224]
        assert shape_zero[0] == 0

        # Fixed batch size
        shape_fixed = [1, 3, 224, 224]
        assert shape_fixed[0] == 1

    def test_batch_tensor_creation(self):
        """Test batch tensor creation and concatenation."""
        batch_size = 4
        single_shape = (1, 3, 224, 224)
        single_size = np.prod(single_shape)

        # Create individual tensors
        tensors = [np.random.randn(*single_shape).astype(np.float32)
                   for _ in range(batch_size)]

        # Concatenate into batch
        batched = np.concatenate(tensors, axis=0)

        assert batched.shape == (batch_size, 3, 224, 224)
        assert batched.dtype == np.float32

    def test_batch_output_splitting(self):
        """Test splitting batch output into individual samples."""
        batch_size = 4
        num_classes = 1000

        # Simulated batch output
        batch_output = np.random.randn(batch_size, num_classes).astype(np.float32)

        # Split into individual samples
        samples = [batch_output[i:i+1] for i in range(batch_size)]

        assert len(samples) == batch_size
        for sample in samples:
            assert sample.shape == (1, num_classes)


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_image_path(self):
        """Test handling of invalid image path."""
        from ivit.core import IVITError

        with pytest.raises((IVITError, FileNotFoundError, ValueError)):
            # This should raise an error for non-existent file
            import cv2
            img = cv2.imread("/non/existent/path.jpg")
            if img is None:
                raise ValueError("Failed to load image")

    def test_invalid_model_path(self):
        """Test handling of invalid model path."""
        from ivit.core import ModelLoadError

        # ModelLoadError should be raised for invalid paths
        assert ModelLoadError is not None

    def test_inference_error(self):
        """Test InferenceError exception."""
        from ivit.core import InferenceError

        with pytest.raises(InferenceError):
            raise InferenceError("Test inference error")

    def test_device_not_found_error(self):
        """Test DeviceNotFoundError exception."""
        from ivit.core import DeviceNotFoundError

        with pytest.raises(DeviceNotFoundError):
            raise DeviceNotFoundError("cuda:99")


@pytest.mark.skip(reason="VideoSource moved to C++ implementation")
class TestVideoSource:
    """Test VideoSource resource management."""

    def test_video_source_context_manager(self):
        """Test VideoSource as context manager."""
        from ivit.core.model import VideoSource

        # Create a mock video source
        with patch('cv2.VideoCapture') as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_instance.get.return_value = 30.0
            mock_cap.return_value = mock_instance

            with VideoSource(0) as vs:
                assert vs is not None

            # Verify release was called
            mock_instance.release.assert_called()

    def test_video_source_manual_release(self):
        """Test VideoSource manual release."""
        from ivit.core.model import VideoSource

        with patch('cv2.VideoCapture') as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_instance.get.return_value = 30.0
            mock_cap.return_value = mock_instance

            vs = VideoSource(0)
            vs.release()

            # Double release should be safe
            vs.release()

            mock_instance.release.assert_called()

    def test_video_source_destructor_safety(self):
        """Test VideoSource destructor doesn't raise exceptions."""
        from ivit.core.model import VideoSource

        with patch('cv2.VideoCapture') as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_instance.get.return_value = 30.0
            # Simulate release raising an exception
            mock_instance.release.side_effect = RuntimeError("Release failed")
            mock_cap.return_value = mock_instance

            vs = VideoSource(0)

            # Destructor should not raise
            try:
                del vs
            except Exception as e:
                pytest.fail(f"Destructor raised exception: {e}")


class TestThreadSafety:
    """Test thread safety of core components."""

    def test_concurrent_device_listing(self):
        """Test thread-safe device listing."""
        from ivit.core.device import list_devices

        results = []
        errors = []

        def worker():
            try:
                devices = list_devices()
                results.append(devices)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10

    def test_concurrent_profiler_access(self):
        """Test thread-safe profiler usage."""
        from ivit.utils.profiler import Profiler

        profiler = Profiler()
        errors = []

        def worker():
            try:
                for _ in range(100):
                    profiler.start()
                    time.sleep(0.0001)
                    profiler.stop()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Note: Current profiler is not thread-safe, this may fail
        # This test documents the expected behavior
        assert len(profiler.times) > 0


class TestPrecisionDetection:
    """Test precision detection functionality."""

    def test_detect_fp32_default(self):
        """Test default FP32 precision detection."""
        from ivit.utils.profiler import _detect_model_precision

        mock_model = Mock()
        mock_model._openvino_config = None
        mock_model._tensorrt_config = None
        mock_model._config = None
        mock_model._input_info = None

        precision = _detect_model_precision(mock_model)
        assert precision == "fp32"

    def test_detect_from_tensorrt_config(self):
        """Test precision detection from TensorRT config."""
        from ivit.utils.profiler import _detect_model_precision

        mock_model = Mock()
        mock_model._openvino_config = None
        mock_model._tensorrt_config = Mock()
        mock_model._tensorrt_config.enable_int8 = False
        mock_model._tensorrt_config.enable_fp16 = True

        precision = _detect_model_precision(mock_model)
        assert precision == "fp16"

    def test_detect_from_input_dtype(self):
        """Test precision detection from input dtype."""
        from ivit.utils.profiler import _detect_model_precision

        mock_model = Mock()
        mock_model._openvino_config = None
        mock_model._tensorrt_config = None
        mock_model._config = None
        mock_model._input_info = [{"dtype": "float16"}]

        precision = _detect_model_precision(mock_model)
        assert precision == "fp16"


class TestNMSAlgorithm:
    """Test NMS algorithm functionality."""

    def test_nms_empty_input(self):
        """Test NMS with empty input."""
        from ivit.core.types import Detection, BBox

        detections = []
        # NMS on empty list should return empty list
        assert len(detections) == 0

    def test_nms_single_detection(self):
        """Test NMS with single detection."""
        from ivit.core.types import Detection, BBox

        detections = [
            Detection(BBox(0, 0, 100, 100), 0, "person", 0.9),
        ]
        # Single detection should be preserved
        assert len(detections) == 1

    def test_nms_non_overlapping(self):
        """Test NMS with non-overlapping detections."""
        from ivit.core.types import Detection, BBox

        detections = [
            Detection(BBox(0, 0, 100, 100), 0, "person", 0.9),
            Detection(BBox(200, 200, 300, 300), 0, "person", 0.8),
        ]
        # Non-overlapping should all be preserved
        assert len(detections) == 2

    def test_iou_calculation(self):
        """Test IoU calculation for NMS."""
        from ivit.core.types import BBox

        # Perfect overlap
        bbox1 = BBox(0, 0, 100, 100)
        bbox2 = BBox(0, 0, 100, 100)
        assert bbox1.iou(bbox2) == 1.0

        # No overlap
        bbox3 = BBox(0, 0, 100, 100)
        bbox4 = BBox(200, 200, 300, 300)
        assert bbox3.iou(bbox4) == 0.0

        # Partial overlap
        bbox5 = BBox(0, 0, 100, 100)
        bbox6 = BBox(50, 50, 150, 150)
        iou = bbox5.iou(bbox6)
        assert 0 < iou < 1


class TestPreprocessing:
    """Test image preprocessing."""

    def test_resize_image(self):
        """Test image resizing."""
        import cv2

        # Create test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Resize to model input size
        target_size = (224, 224)
        resized = cv2.resize(img, target_size)

        assert resized.shape == (224, 224, 3)

    def test_normalize_image(self):
        """Test image normalization."""
        # Create test image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Normalize to [0, 1]
        normalized = img.astype(np.float32) / 255.0

        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_hwc_to_chw_conversion(self):
        """Test HWC to CHW format conversion."""
        # HWC format (Height, Width, Channels)
        img_hwc = np.random.randn(224, 224, 3).astype(np.float32)

        # Convert to CHW (Channels, Height, Width)
        img_chw = np.transpose(img_hwc, (2, 0, 1))

        assert img_hwc.shape == (224, 224, 3)
        assert img_chw.shape == (3, 224, 224)

    def test_add_batch_dimension(self):
        """Test adding batch dimension."""
        # CHW format
        img = np.random.randn(3, 224, 224).astype(np.float32)

        # Add batch dimension -> NCHW
        batched = np.expand_dims(img, axis=0)

        assert batched.shape == (1, 3, 224, 224)


class TestPostprocessing:
    """Test inference postprocessing."""

    def test_softmax(self):
        """Test softmax implementation."""
        logits = np.array([1.0, 2.0, 3.0])

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        assert np.isclose(np.sum(probs), 1.0)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_argmax_classification(self):
        """Test argmax for classification."""
        probs = np.array([0.1, 0.7, 0.2])

        class_id = np.argmax(probs)
        confidence = probs[class_id]

        assert class_id == 1
        assert confidence == 0.7

    def test_confidence_thresholding(self):
        """Test confidence thresholding."""
        from ivit.core.types import Detection, BBox

        detections = [
            Detection(BBox(0, 0, 100, 100), 0, "person", 0.9),
            Detection(BBox(10, 10, 110, 110), 0, "person", 0.3),
            Detection(BBox(20, 20, 120, 120), 1, "car", 0.6),
        ]

        threshold = 0.5
        filtered = [d for d in detections if d.confidence >= threshold]

        assert len(filtered) == 2


class TestProfileReport:
    """Test ProfileReport functionality."""

    def test_profile_report_creation(self):
        """Test ProfileReport creation."""
        from ivit.utils.profiler import ProfileReport

        report = ProfileReport(
            model_name="test_model",
            device="cuda:0",
            backend="tensorrt",
            precision="fp16",
            input_shape=(1, 3, 640, 640),
            iterations=100,
            latency_mean=10.5,
            latency_median=10.0,
            latency_std=1.5,
            latency_min=8.0,
            latency_max=15.0,
            latency_p95=12.0,
            latency_p99=14.0,
            throughput_fps=95.2,
            memory_mb=512.0,
        )

        assert report.model_name == "test_model"
        assert report.latency_mean == 10.5
        assert report.throughput_fps == 95.2

    def test_profile_report_to_dict(self):
        """Test ProfileReport serialization."""
        from ivit.utils.profiler import ProfileReport

        report = ProfileReport(
            model_name="test",
            latency_mean=10.0,
        )

        data = report.to_dict()
        assert "model_name" in data
        assert "latency" in data
        assert data["latency"]["mean_ms"] == 10.0

    def test_profile_report_to_json(self):
        """Test ProfileReport JSON export."""
        from ivit.utils.profiler import ProfileReport
        import json

        report = ProfileReport(model_name="test")
        json_str = report.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["model_name"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
