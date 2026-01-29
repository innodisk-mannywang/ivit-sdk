"""
API tests for iVIT-SDK.
"""

import pytest
import numpy as np


class TestImport:
    """Test module imports."""

    def test_import_ivit(self):
        """Test main module import."""
        import ivit
        assert hasattr(ivit, '__version__')
        assert ivit.__version__ == "1.0.0"

    def test_import_core(self):
        """Test core imports."""
        from ivit.core import (
            LoadConfig,
            InferConfig,
            DeviceInfo,
            BBox,
            Detection,
            ClassificationResult,
        )

    def test_import_vision(self):
        """Test vision imports."""
        from ivit.vision import Classifier, Detector, Segmentor

    def test_import_utils(self):
        """Test utils imports."""
        from ivit.utils import Visualizer, Profiler, VideoStream


class TestTypes:
    """Test data types."""

    def test_bbox_creation(self):
        """Test BBox creation."""
        from ivit.core.types import BBox

        bbox = BBox(10, 20, 100, 150)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 150

    def test_bbox_properties(self):
        """Test BBox properties."""
        from ivit.core.types import BBox

        bbox = BBox(0, 0, 100, 50)
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.area == 5000
        assert bbox.center == (50, 25)

    def test_bbox_iou(self):
        """Test BBox IoU calculation."""
        from ivit.core.types import BBox

        bbox1 = BBox(0, 0, 100, 100)
        bbox2 = BBox(50, 50, 150, 150)

        iou = bbox1.iou(bbox2)
        # Intersection: 50x50 = 2500
        # Union: 100x100 + 100x100 - 2500 = 17500
        # IoU = 2500 / 17500 ≈ 0.143
        assert abs(iou - 0.143) < 0.01

    def test_bbox_conversion(self):
        """Test BBox format conversions."""
        from ivit.core.types import BBox

        bbox = BBox(10, 20, 110, 120)

        # to_xywh
        xywh = bbox.to_xywh()
        assert xywh == (10, 20, 100, 100)

        # to_cxcywh
        cxcywh = bbox.to_cxcywh()
        assert cxcywh == (60, 70, 100, 100)

        # from_xywh
        bbox2 = BBox.from_xywh(10, 20, 100, 100)
        assert bbox2.x1 == 10
        assert bbox2.x2 == 110

        # from_cxcywh
        bbox3 = BBox.from_cxcywh(60, 70, 100, 100)
        assert bbox3.x1 == 10
        assert bbox3.x2 == 110

    def test_load_config(self):
        """Test LoadConfig defaults."""
        from ivit.core.types import LoadConfig

        config = LoadConfig()
        assert config.device == "auto"
        assert config.backend == "auto"
        assert config.batch_size == 1

    def test_infer_config(self):
        """Test InferConfig defaults."""
        from ivit.core.types import InferConfig

        config = InferConfig()
        assert config.conf_threshold == 0.5
        assert config.iou_threshold == 0.45
        assert config.max_detections == 300  # YOLO default


class TestResults:
    """Test Results class."""

    def test_empty_results(self):
        """Test empty results."""
        from ivit.core.result import Results

        results = Results()
        assert len(results) == 0
        assert results.empty

    def test_results_with_detections(self):
        """Test results with detections."""
        from ivit.core.result import Results
        from ivit.core.types import Detection, BBox

        results = Results()
        results.detections = [
            Detection(BBox(0, 0, 100, 100), 0, "person", 0.9),
            Detection(BBox(50, 50, 150, 150), 1, "car", 0.8),
        ]

        assert len(results) == 2
        assert not results.empty
        assert len(results.detections) == 2

    def test_results_filter(self):
        """Test results filtering."""
        from ivit.core.result import Results
        from ivit.core.types import Detection, BBox

        results = Results()
        results.detections = [
            Detection(BBox(0, 0, 100, 100), 0, "person", 0.9),
            Detection(BBox(50, 50, 150, 150), 1, "car", 0.8),
            Detection(BBox(100, 100, 200, 200), 0, "person", 0.6),
        ]

        # Filter by class
        persons = results.filter_by_class(["person"])
        assert len(persons) == 2

        # Filter by confidence
        high_conf = results.filter_by_confidence(0.85)
        assert len(high_conf) == 1
        assert high_conf[0].label == "person"

    def test_results_to_dict(self):
        """Test results serialization."""
        from ivit.core.result import Results
        from ivit.core.types import Detection, BBox

        results = Results()
        results.detections = [
            Detection(BBox(0, 0, 100, 100), 0, "person", 0.9),
        ]
        results.inference_time_ms = 10.5
        results.device_used = "cuda:0"
        results.image_size = (480, 640)

        data = results.to_dict()
        assert "detections" in data
        assert len(data["detections"]) == 1
        assert data["inference_time_ms"] == 10.5
        assert data["device_used"] == "cuda:0"

    def test_results_to_json(self):
        """Test results JSON export."""
        from ivit.core.result import Results
        from ivit.core.types import ClassificationResult
        import json

        results = Results()
        results.classifications = [
            ClassificationResult(0, "cat", 0.95),
            ClassificationResult(1, "dog", 0.05),
        ]

        json_str = results.to_json()
        data = json.loads(json_str)
        assert "classifications" in data


class TestDevice:
    """Test device management."""

    def test_list_devices(self):
        """Test device listing."""
        from ivit.core.device import list_devices

        devices = list_devices()
        assert isinstance(devices, list)
        # Should at least have CPU fallback
        assert len(devices) >= 1

    def test_get_best_device(self):
        """Test best device selection."""
        from ivit.core.device import get_best_device

        device = get_best_device()
        assert device.id is not None
        assert device.backend is not None


class TestProfiler:
    """Test profiler."""

    def test_profiler_timing(self):
        """Test profiler timing."""
        from ivit.utils.profiler import Profiler
        import time

        profiler = Profiler()

        profiler.start()
        time.sleep(0.01)  # 10ms
        profiler.stop()

        elapsed = profiler.elapsed_ms()
        assert elapsed >= 10
        assert elapsed < 50  # Allow some margin

    def test_profiler_stats(self):
        """Test profiler statistics."""
        from ivit.utils.profiler import Profiler
        import time

        profiler = Profiler()

        for _ in range(10):
            profiler.start()
            time.sleep(0.001)  # 1ms
            profiler.stop()

        mean, median, std, min_t, max_t, p95, p99 = profiler.calculate_stats()

        assert mean > 0
        assert median > 0
        assert min_t <= mean <= max_t


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
