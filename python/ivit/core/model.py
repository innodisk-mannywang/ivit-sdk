"""
Model loading and management.
"""

from typing import Union, Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import logging

from .types import LoadConfig, InferConfig, TensorInfo, TaskType
from .result import Results

logger = logging.getLogger(__name__)


class Model:
    """
    Base model class for inference.

    This class wraps the underlying runtime-specific model and provides
    a unified interface for inference.
    """

    def __init__(
        self,
        path: str,
        config: LoadConfig,
        runtime: Any = None,
    ):
        """
        Initialize model.

        Args:
            path: Model path
            config: Load configuration
            runtime: Runtime backend instance
        """
        self._path = path
        self._config = config
        self._runtime = runtime
        self._handle = None
        self._labels: List[str] = []
        self._input_info: List[TensorInfo] = []
        self._output_info: List[TensorInfo] = []
        self._task = TaskType.DETECTION

        # Load model
        self._load()

    def _load(self):
        """Load model using appropriate backend."""
        from .device import get_backend_for_device
        from ..runtime import get_runtime

        # Determine backend
        if self._config.backend == "auto":
            backend = get_backend_for_device(self._config.device)
        else:
            backend = self._config.backend

        # Get runtime
        self._runtime = get_runtime(backend)

        # Load model
        logger.info(f"Loading model: {self._path}")
        logger.info(f"Device: {self._config.device}, Backend: {backend}")

        self._handle = self._runtime.load(self._path, self._config)
        self._input_info = self._runtime.get_input_info(self._handle)
        self._output_info = self._runtime.get_output_info(self._handle)

        # Try to load labels
        self._load_labels()

    def _load_labels(self):
        """Try to load labels from accompanying file."""
        path = Path(self._path)
        label_paths = [
            path.with_suffix(".txt"),
            path.with_suffix(".names"),
            path.parent / "labels.txt",
            path.parent / "classes.txt",
        ]

        for label_path in label_paths:
            if label_path.exists():
                with open(label_path) as f:
                    self._labels = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(self._labels)} labels from {label_path}")
                break

    @property
    def name(self) -> str:
        """Get model name."""
        return Path(self._path).stem

    @property
    def task(self) -> TaskType:
        """Get task type."""
        return self._task

    @property
    def device(self) -> str:
        """Get device."""
        return self._config.device

    @property
    def backend(self) -> str:
        """Get backend."""
        return self._config.backend

    @property
    def input_info(self) -> List[TensorInfo]:
        """Get input tensor info."""
        return self._input_info

    @property
    def output_info(self) -> List[TensorInfo]:
        """Get output tensor info."""
        return self._output_info

    @property
    def labels(self) -> List[str]:
        """Get class labels."""
        return self._labels

    def predict(
        self,
        source: Union[str, np.ndarray, List],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        **kwargs
    ) -> Results:
        """
        Run inference.

        Args:
            source: Input image (path, array, or list)
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold

        Returns:
            Inference results
        """
        import cv2

        # Load image if path
        if isinstance(source, str):
            image = cv2.imread(source)
            if image is None:
                raise ValueError(f"Failed to load image: {source}")
        elif isinstance(source, np.ndarray):
            image = source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # Create config
        config = InferConfig(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            **kwargs
        )

        # Preprocess
        input_tensor, preprocess_info = self._preprocess(image)

        # Inference
        import time
        start = time.perf_counter()
        outputs = self._runtime.infer(self._handle, {"input": input_tensor})
        inference_time = (time.perf_counter() - start) * 1000

        # Postprocess
        results = self._postprocess(outputs, image.shape[:2], preprocess_info, config)
        results.inference_time_ms = inference_time
        results.device_used = self._config.device
        results.image_size = image.shape[:2]
        results._original_image = image

        return results

    def predict_batch(
        self,
        sources: List[Union[str, np.ndarray]],
        **kwargs
    ) -> List[Results]:
        """Run batch inference."""
        return [self.predict(src, **kwargs) for src in sources]

    def warmup(self, iterations: int = 3):
        """Warmup model."""
        # Get input shape
        input_shape = self._input_info[0].shape
        dummy = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(iterations):
            self._runtime.infer(self._handle, {"input": dummy})

        logger.info(f"Warmup completed ({iterations} iterations)")

    def _preprocess(
        self,
        image: np.ndarray
    ) -> tuple:
        """Preprocess image for inference."""
        import cv2

        # Get target size from model input
        input_shape = self._input_info[0].shape
        if len(input_shape) == 4:
            _, _, h, w = input_shape
        else:
            h, w = 640, 640

        orig_h, orig_w = image.shape[:2]

        # Resize with letterbox
        scale = min(w / orig_w, h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_w, pad_h = (w - new_w) // 2, (h - new_h) // 2

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((h, w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        # Convert to tensor (NCHW, normalized)
        tensor = padded.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
        tensor = np.expand_dims(tensor, 0)  # Add batch dimension

        preprocess_info = {
            "scale": scale,
            "pad_w": pad_w,
            "pad_h": pad_h,
            "orig_size": (orig_h, orig_w),
        }

        return tensor, preprocess_info

    def _postprocess(
        self,
        outputs: Dict[str, np.ndarray],
        orig_size: tuple,
        preprocess_info: dict,
        config: InferConfig
    ) -> Results:
        """Postprocess model outputs."""
        from .types import Detection, BBox

        results = Results()

        # Get first output
        output = list(outputs.values())[0]

        # Detect output format and postprocess accordingly
        # This is a simplified YOLO postprocessing
        if len(output.shape) == 3:
            # Shape: (1, num_detections, 4+1+num_classes) or (1, 4+1+num_classes, num_detections)
            if output.shape[1] < output.shape[2]:
                output = output.transpose(0, 2, 1)

            output = output[0]  # Remove batch dimension

            scale = preprocess_info["scale"]
            pad_w = preprocess_info["pad_w"]
            pad_h = preprocess_info["pad_h"]

            detections = []
            for det in output:
                # YOLO format: [cx, cy, w, h, obj_conf, class_scores...]
                cx, cy, w, h = det[:4]
                scores = det[4:]

                if len(scores) == 1:
                    # Object confidence only
                    obj_conf = scores[0]
                    class_id = 0
                    confidence = obj_conf
                else:
                    # Multiple classes
                    class_id = int(np.argmax(scores))
                    confidence = float(scores[class_id])

                if confidence < config.conf_threshold:
                    continue

                # Convert to original coordinates
                x1 = (cx - w / 2 - pad_w) / scale
                y1 = (cy - h / 2 - pad_h) / scale
                x2 = (cx + w / 2 - pad_w) / scale
                y2 = (cy + h / 2 - pad_h) / scale

                # Clip to image bounds
                x1 = max(0, min(x1, orig_size[1]))
                y1 = max(0, min(y1, orig_size[0]))
                x2 = max(0, min(x2, orig_size[1]))
                y2 = max(0, min(y2, orig_size[0]))

                label = self._labels[class_id] if class_id < len(self._labels) else str(class_id)

                detections.append(Detection(
                    bbox=BBox(x1, y1, x2, y2),
                    class_id=class_id,
                    label=label,
                    confidence=confidence,
                ))

            # NMS
            detections = self._nms(detections, config.iou_threshold)
            results.detections = detections[:config.max_detections]

        results.raw_outputs = outputs
        return results

    def _nms(
        self,
        detections: List,
        iou_threshold: float
    ) -> List:
        """Non-maximum suppression."""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if d.class_id != best.class_id or best.bbox.iou(d.bbox) < iou_threshold
            ]

        return keep


def load_model(
    path: Union[str, Path],
    device: str = "auto",
    backend: str = "auto",
    task: Optional[str] = None,
    **kwargs
) -> Model:
    """
    Load a model.

    Args:
        path: Model path or Model Zoo name
        device: Target device ("auto", "cpu", "cuda:0", etc.)
        backend: Backend ("auto", "openvino", "tensorrt", "snpe")
        task: Task type hint

    Returns:
        Loaded model

    Examples:
        >>> model = load_model("yolov8n.onnx")
        >>> model = load_model("yolov8n.onnx", device="cuda:0")
        >>> model = load_model("efficientnet_b0", task="classification")
    """
    config = LoadConfig(
        device=device,
        backend=backend,
        task=task,
        **kwargs
    )

    return Model(str(path), config)
