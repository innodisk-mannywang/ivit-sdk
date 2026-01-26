"""
Model loading and management.
"""

from typing import Union, Optional, List, Dict, Any, Generator, Iterable, Callable
from pathlib import Path
import numpy as np
import logging
import time

from .types import LoadConfig, InferConfig, TensorInfo, TaskType
from .result import Results
from .callbacks import CallbackManager, CallbackContext, CallbackEvent
from .runtime_config import OpenVINOConfig, TensorRTConfig, ONNXRuntimeConfig, SNPEConfig
from .processors import (
    BasePreProcessor,
    BasePostProcessor,
    get_preprocessor,
    get_postprocessor,
)

logger = logging.getLogger(__name__)


class VideoSource:
    """
    Video source wrapper for streaming inference.

    Supports video files, camera indices, and RTSP streams.
    """

    def __init__(
        self,
        source: Union[str, int],
        loop: bool = False,
    ):
        """
        Initialize video source.

        Args:
            source: Video file path, camera index (0, 1, ...), or RTSP URL
            loop: Loop video file (only for files)
        """
        import cv2

        self.source = source
        self.loop = loop
        self._cap = None
        self._frame_count = 0
        self._fps = 0
        self._width = 0
        self._height = 0

        self._open()

    def _open(self):
        """Open video source."""
        import cv2

        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.source}")

        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def is_camera(self) -> bool:
        return isinstance(self.source, int) or str(self.source).startswith("rtsp")

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        import cv2

        ret, frame = self._cap.read()

        if not ret:
            if self.loop and not self.is_camera:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()

            if not ret:
                raise StopIteration

        return frame

    def __len__(self) -> int:
        return self._frame_count if self._frame_count > 0 else 0

    def release(self):
        """Release video source."""
        if self._cap:
            self._cap.release()

    def __del__(self):
        self.release()


# COCO class labels (80 classes)
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


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

        # Processors (set after loading based on task type)
        self._preprocessor: Optional[BasePreProcessor] = None
        self._postprocessor: Optional[BasePostProcessor] = None

        # Callback system
        self._callbacks = CallbackManager()

        # Load model
        self._load()

        # Initialize default processors based on task
        self._init_processors()

        # Trigger model loaded callback
        self._trigger_callback(CallbackEvent.MODEL_LOADED)

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

        # Use load_model method (all runtimes implement this)
        self._handle = self._runtime.load_model(
            self._path,
            device=self._config.device,
            precision=getattr(self._config, 'precision', 'fp32'),
        )
        self._input_info = self._handle.get_input_info()
        self._output_info = self._handle.get_output_info()

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
                return

        # Use default COCO labels for YOLO models
        model_name = path.stem.lower()
        if "yolo" in model_name:
            self._labels = COCO_LABELS
            logger.info("Using default COCO labels for YOLO model")

    def _init_processors(self):
        """Initialize default processors based on task type and model name."""
        model_name = Path(self._path).stem.lower()

        # Auto-detect task type from model name
        if any(x in model_name for x in ["yolo", "ssd", "rcnn", "fcos", "retinanet"]):
            self._task = TaskType.DETECTION
            self._preprocessor = get_preprocessor("letterbox")
            self._postprocessor = get_postprocessor("yolo")
            logger.debug("Using letterbox preprocessor and YOLO postprocessor")
        elif any(x in model_name for x in ["resnet", "efficientnet", "mobilenet", "vgg", "densenet", "classifier"]):
            self._task = TaskType.CLASSIFICATION
            self._preprocessor = get_preprocessor("center_crop")
            self._postprocessor = get_postprocessor("classification")
            logger.debug("Using center_crop preprocessor and classification postprocessor")
        else:
            # Default to detection with letterbox
            self._preprocessor = get_preprocessor("letterbox")
            self._postprocessor = get_postprocessor("yolo")
            logger.debug("Using default letterbox preprocessor and YOLO postprocessor")

    def set_preprocessor(self, preprocessor: BasePreProcessor) -> 'Model':
        """
        Set a custom preprocessor.

        Args:
            preprocessor: PreProcessor instance

        Returns:
            self (for method chaining)

        Examples:
            >>> from ivit.core import LetterboxPreProcessor
            >>> model.set_preprocessor(LetterboxPreProcessor(pad_value=0))
        """
        self._preprocessor = preprocessor
        return self

    def set_postprocessor(self, postprocessor: BasePostProcessor) -> 'Model':
        """
        Set a custom postprocessor.

        Args:
            postprocessor: PostProcessor instance

        Returns:
            self (for method chaining)

        Examples:
            >>> from ivit.core import YOLOPostProcessor
            >>> model.set_postprocessor(YOLOPostProcessor())
        """
        self._postprocessor = postprocessor
        return self

    @property
    def preprocessor(self) -> Optional[BasePreProcessor]:
        """Get current preprocessor."""
        return self._preprocessor

    @property
    def postprocessor(self) -> Optional[BasePostProcessor]:
        """Get current postprocessor."""
        return self._postprocessor

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

    @property
    def callbacks(self) -> CallbackManager:
        """Get callback manager."""
        return self._callbacks

    # =========================================================================
    # Callback API
    # =========================================================================

    def on(
        self,
        event: str,
        callback: Callable[[CallbackContext], Any] = None,
        priority: int = 0
    ):
        """
        Register a callback for an event.

        Can be used as a decorator or direct method call.

        Args:
            event: Event name. Options:
                - "pre_process": Before preprocessing
                - "post_process": After postprocessing
                - "infer_start": Before inference
                - "infer_end": After inference
                - "batch_start": Before batch processing
                - "batch_end": After batch processing
                - "stream_start": When stream starts
                - "stream_frame": After each frame
                - "stream_end": When stream ends
            callback: Callback function (optional for decorator usage)
            priority: Priority (higher = called first)

        Returns:
            Decorator function or None

        Examples:
            >>> # As decorator
            >>> @model.on("infer_end")
            ... def log_time(ctx):
            ...     print(f"Took {ctx.latency_ms}ms")
            >>>
            >>> # Direct registration
            >>> model.on("infer_end", lambda ctx: print(f"Latency: {ctx.latency_ms}ms"))
            >>>
            >>> # Using built-in callbacks
            >>> from ivit.core.callbacks import FPSCounter
            >>> fps = FPSCounter()
            >>> model.on("infer_end", fps)
            >>> # Later: print(fps.fps)
        """
        def decorator(fn):
            self._callbacks.register(event, fn, priority)
            return fn

        if callback is not None:
            self._callbacks.register(event, callback, priority)
            return None

        return decorator

    def remove_callback(
        self,
        event: str,
        callback: Callable = None
    ) -> int:
        """
        Remove callbacks.

        Args:
            event: Event name
            callback: Specific callback (None = remove all for event)

        Returns:
            Number of callbacks removed

        Examples:
            >>> model.remove_callback("infer_end")  # Remove all
            >>> model.remove_callback("infer_end", my_callback)  # Remove specific
        """
        return self._callbacks.unregister(event, callback)

    def _trigger_callback(
        self,
        event: str,
        **kwargs
    ) -> CallbackContext:
        """
        Trigger callbacks for an event.

        Args:
            event: Event name
            **kwargs: Context data

        Returns:
            CallbackContext with any modifications
        """
        ctx = CallbackContext(
            event=event,
            model_name=self.name,
            device=self._config.device,
            **kwargs
        )
        self._callbacks.trigger(event, ctx)
        return ctx

    # =========================================================================
    # Hardware Configuration API
    # =========================================================================

    @property
    def runtime(self) -> Any:
        """
        Get the underlying runtime instance.

        Allows direct access to backend-specific functionality.

        Returns:
            Runtime instance (OpenVINORuntime, TensorRTRuntime, ONNXRuntime, etc.)

        Examples:
            >>> model = ivit.load("model.onnx", device="cuda:0")
            >>> runtime = model.runtime
            >>> print(runtime.name)  # "onnxruntime"
        """
        return self._runtime

    @property
    def runtime_handle(self) -> Any:
        """
        Get the underlying model handle from the runtime.

        Returns the raw model object (e.g., ONNX InferenceSession, OpenVINO CompiledModel).

        Examples:
            >>> handle = model.runtime_handle
            >>> # For ONNX Runtime:
            >>> session = handle.session
            >>> # For OpenVINO:
            >>> compiled_model = handle.compiled_model
        """
        return self._handle

    def configure_openvino(
        self,
        performance_mode: str = None,
        num_streams: int = None,
        inference_precision: str = None,
        enable_cpu_pinning: bool = None,
        num_threads: int = None,
        cache_dir: str = None,
        **kwargs
    ) -> 'Model':
        """
        Configure OpenVINO-specific settings.

        Only effective when using OpenVINO backend.

        Args:
            performance_mode: "LATENCY" (default), "THROUGHPUT", or "CUMULATIVE_THROUGHPUT"
            num_streams: Number of inference streams
            inference_precision: "FP32", "FP16", or "INT8"
            enable_cpu_pinning: Pin threads to CPU cores
            num_threads: Number of CPU threads (0 = auto)
            cache_dir: Model compilation cache directory
            **kwargs: Additional device properties

        Returns:
            self (for method chaining)

        Examples:
            >>> model.configure_openvino(
            ...     performance_mode="LATENCY",
            ...     num_streams=4,
            ...     inference_precision="FP16",
            ... )
            >>>
            >>> # Method chaining
            >>> model.configure_openvino(num_threads=8).warmup(5)
        """
        config = OpenVINOConfig()

        if performance_mode is not None:
            config.performance_mode = performance_mode
        if num_streams is not None:
            config.num_streams = num_streams
        if inference_precision is not None:
            config.inference_precision = inference_precision
        if enable_cpu_pinning is not None:
            config.enable_cpu_pinning = enable_cpu_pinning
        if num_threads is not None:
            config.num_threads = num_threads
        if cache_dir is not None:
            config.cache_dir = cache_dir

        config.device_properties.update(kwargs)

        # Store config
        self._openvino_config = config

        # Apply config if runtime supports it
        if hasattr(self._handle, 'apply_config'):
            self._handle.apply_config(config.to_ov_config())
            logger.info(f"Applied OpenVINO config: {config}")
        else:
            logger.warning("OpenVINO configuration stored but runtime does not support apply_config")

        return self

    def configure_tensorrt(
        self,
        workspace_size: int = None,
        dla_core: int = None,
        enable_fp16: bool = None,
        enable_int8: bool = None,
        builder_optimization_level: int = None,
        enable_sparsity: bool = None,
        max_batch_size: int = None,
        enable_profiling: bool = None,
        **kwargs
    ) -> 'Model':
        """
        Configure TensorRT-specific settings.

        Only effective when using TensorRT backend.

        Args:
            workspace_size: Maximum workspace size in bytes (default: 1GB)
            dla_core: DLA core to use (-1 = disabled, 0-1 on Jetson)
            enable_fp16: Enable FP16 inference
            enable_int8: Enable INT8 inference
            builder_optimization_level: Optimization level (0-5)
            enable_sparsity: Enable sparsity optimizations
            max_batch_size: Maximum batch size
            enable_profiling: Enable layer profiling

        Returns:
            self (for method chaining)

        Examples:
            >>> model.configure_tensorrt(
            ...     workspace_size=1 << 30,  # 1GB
            ...     enable_fp16=True,
            ...     builder_optimization_level=5,
            ... )
        """
        config = TensorRTConfig()

        if workspace_size is not None:
            config.workspace_size = workspace_size
        if dla_core is not None:
            config.dla_core = dla_core
        if enable_fp16 is not None:
            config.enable_fp16 = enable_fp16
        if enable_int8 is not None:
            config.enable_int8 = enable_int8
        if builder_optimization_level is not None:
            config.builder_optimization_level = builder_optimization_level
        if enable_sparsity is not None:
            config.enable_sparsity = enable_sparsity
        if max_batch_size is not None:
            config.max_batch_size = max_batch_size
        if enable_profiling is not None:
            config.enable_profiling = enable_profiling

        # Store config
        self._tensorrt_config = config

        # Apply config if runtime supports it
        if hasattr(self._handle, 'apply_config'):
            self._handle.apply_config(config.to_trt_config())
            logger.info(f"Applied TensorRT config: {config}")
        else:
            logger.warning("TensorRT configuration stored but runtime does not support apply_config")

        return self

    def configure_onnxruntime(
        self,
        num_threads: int = None,
        inter_op_num_threads: int = None,
        graph_optimization_level: str = None,
        enable_cuda_graph: bool = None,
        cuda_device_id: int = None,
        cuda_mem_limit: int = None,
        execution_mode: str = None,
        **kwargs
    ) -> 'Model':
        """
        Configure ONNX Runtime-specific settings.

        Args:
            num_threads: Number of intra-op threads (0 = auto)
            inter_op_num_threads: Number of inter-op threads (0 = auto)
            graph_optimization_level: "ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"
            enable_cuda_graph: Enable CUDA graph capture
            cuda_device_id: CUDA device ID
            cuda_mem_limit: CUDA memory limit (0 = no limit)
            execution_mode: "ORT_SEQUENTIAL" or "ORT_PARALLEL"

        Returns:
            self (for method chaining)

        Examples:
            >>> model.configure_onnxruntime(
            ...     num_threads=8,
            ...     enable_cuda_graph=True,
            ... )
        """
        config = ONNXRuntimeConfig()

        if num_threads is not None:
            config.num_threads = num_threads
        if inter_op_num_threads is not None:
            config.inter_op_num_threads = inter_op_num_threads
        if graph_optimization_level is not None:
            config.graph_optimization_level = graph_optimization_level
        if enable_cuda_graph is not None:
            config.enable_cuda_graph = enable_cuda_graph
        if cuda_device_id is not None:
            config.cuda_device_id = cuda_device_id
        if cuda_mem_limit is not None:
            config.cuda_mem_limit = cuda_mem_limit
        if execution_mode is not None:
            config.execution_mode = execution_mode

        # Store config
        self._onnxruntime_config = config

        # Apply config if runtime supports it
        if hasattr(self._handle, 'apply_config'):
            self._handle.apply_config(config.to_ort_config())
            logger.info(f"Applied ONNX Runtime config: {config}")
        else:
            logger.warning("ONNX Runtime configuration stored but runtime does not support apply_config")

        return self

    def configure_snpe(
        self,
        runtime: str = None,
        performance_profile: str = None,
        enable_profiling: bool = None,
        **kwargs
    ) -> 'Model':
        """
        Configure Qualcomm SNPE-specific settings.

        Only effective when using SNPE backend.

        Args:
            runtime: "cpu", "gpu", "dsp", or "aip"
            performance_profile: "DEFAULT", "BALANCED", "HIGH_PERFORMANCE", "POWER_SAVER"
            enable_profiling: Enable profiling

        Returns:
            self (for method chaining)

        Examples:
            >>> model.configure_snpe(
            ...     runtime="dsp",
            ...     performance_profile="HIGH_PERFORMANCE",
            ... )
        """
        config = SNPEConfig()

        if runtime is not None:
            config.runtime = runtime
        if performance_profile is not None:
            config.performance_profile = performance_profile
        if enable_profiling is not None:
            config.enable_profiling = enable_profiling

        # Store config
        self._snpe_config = config

        # Apply config if runtime supports it
        if hasattr(self._handle, 'apply_config'):
            self._handle.apply_config(config.to_snpe_config())
            logger.info(f"Applied SNPE config: {config}")
        else:
            logger.warning("SNPE configuration stored but runtime does not support apply_config")

        return self

    def infer_raw(
        self,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Run raw inference without pre/post processing.

        Allows direct tensor-in, tensor-out inference for advanced use cases.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays

        Examples:
            >>> # Prepare input tensor manually
            >>> input_tensor = preprocess_my_way(image)
            >>> input_name = model.input_info[0]["name"]
            >>> outputs = model.infer_raw({input_name: input_tensor})
            >>>
            >>> # Process outputs manually
            >>> detections = postprocess_my_way(outputs)
        """
        return self._handle.infer(inputs)

    def __call__(
        self,
        source: Union[str, np.ndarray, List],
        **kwargs
    ) -> Results:
        """
        Run inference (callable interface).

        Allows model to be used like Ultralytics: model("image.jpg")

        Args:
            source: Input image (path, array, or list)
            **kwargs: Additional inference arguments

        Returns:
            Inference results

        Examples:
            >>> model = ivit.load("yolov8n.onnx")
            >>> results = model("image.jpg")
            >>> results.show()
        """
        return self.predict(source, **kwargs)

    def predict(
        self,
        source: Union[str, np.ndarray, List],
        conf: float = None,
        conf_threshold: float = None,
        iou: float = None,
        iou_threshold: float = None,
        classes: Optional[List[int]] = None,
        max_det: int = None,
        max_detections: int = None,
        imgsz: int = None,
        half: bool = False,
        **kwargs
    ) -> Results:
        """
        Run inference.

        Args:
            source: Input image (path, array, or list)
            conf: Confidence threshold (alias for conf_threshold)
            conf_threshold: Confidence threshold
            iou: NMS IoU threshold (alias for iou_threshold)
            iou_threshold: NMS IoU threshold
            classes: Filter by class IDs (e.g., [0, 1, 2] for person, bicycle, car)
            max_det: Maximum detections (alias for max_detections)
            max_detections: Maximum number of detections
            imgsz: Input image size (overrides model default)
            half: Use FP16 inference (if supported)

        Returns:
            Inference results

        Examples:
            >>> results = model.predict("image.jpg")
            >>> results = model.predict("image.jpg", conf=0.25, iou=0.45)
            >>> results = model.predict("image.jpg", classes=[0])  # Only detect persons
            >>> results = model.predict("image.jpg", max_det=100)
        """
        import cv2

        # Handle parameter aliases
        conf_threshold = conf if conf is not None else (conf_threshold or 0.5)
        iou_threshold = iou if iou is not None else (iou_threshold or 0.45)
        max_detections = max_det if max_det is not None else (max_detections or 300)

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
            max_detections=max_detections,
            classes=classes,
            imgsz=imgsz,
            half=half,
            **kwargs
        )

        # Track timing
        total_start = time.perf_counter()

        # Trigger pre_process callback
        ctx = self._trigger_callback(
            CallbackEvent.PRE_PROCESS,
            image=image,
            image_shape=image.shape[:2],
        )

        # Preprocess
        preprocess_start = time.perf_counter()
        input_tensor, preprocess_info = self._preprocess(image)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000

        # Trigger infer_start callback
        self._trigger_callback(
            CallbackEvent.INFER_START,
            image=image,
            input_tensor=input_tensor,
        )

        # Inference
        infer_start = time.perf_counter()
        input_name = self._input_info[0]["name"] if self._input_info else "input"
        outputs = self._handle.infer({input_name: input_tensor})
        inference_time = (time.perf_counter() - infer_start) * 1000

        # Postprocess
        postprocess_start = time.perf_counter()
        results = self._postprocess(outputs, image.shape[:2], preprocess_info, config)
        postprocess_time = (time.perf_counter() - postprocess_start) * 1000

        total_time = (time.perf_counter() - total_start) * 1000

        # Set result metadata
        results.inference_time_ms = inference_time
        results.device_used = self._config.device
        results.image_size = image.shape[:2]
        results._original_image = image

        # Trigger post_process callback
        self._trigger_callback(
            CallbackEvent.POST_PROCESS,
            image=image,
            outputs=outputs,
            results=results,
        )

        # Trigger infer_end callback
        self._trigger_callback(
            CallbackEvent.INFER_END,
            image=image,
            results=results,
            latency_ms=total_time,
            preprocess_ms=preprocess_time,
            inference_ms=inference_time,
            postprocess_ms=postprocess_time,
        )

        return results

    def predict_batch(
        self,
        sources: List[Union[str, np.ndarray]],
        conf: float = 0.5,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        **kwargs
    ) -> List[Results]:
        """
        Run batch inference on multiple images.

        Automatically handles models with fixed batch size by processing
        images sequentially when needed.

        Args:
            sources: List of image paths or numpy arrays
            conf: Confidence threshold
            iou: NMS IoU threshold
            classes: Filter by class IDs
            max_det: Maximum detections per image

        Returns:
            List of Results for each image

        Examples:
            >>> images = ["img1.jpg", "img2.jpg", "img3.jpg"]
            >>> results = model.predict_batch(images)
            >>> for i, r in enumerate(results):
            ...     print(f"Image {i}: {len(r)} detections")
            >>>
            >>> # With parameters
            >>> results = model.predict_batch(images, conf=0.25, classes=[0])
        """
        import cv2

        # Load all images
        images = []
        for src in sources:
            if isinstance(src, str):
                img = cv2.imread(src)
                if img is None:
                    raise ValueError(f"Failed to load image: {src}")
                images.append(img)
            elif isinstance(src, np.ndarray):
                images.append(src)
            else:
                raise TypeError(f"Unsupported source type: {type(src)}")

        # Trigger batch_start callback
        self._trigger_callback(
            CallbackEvent.BATCH_START,
            batch_size=len(images),
        )

        # Check if model supports dynamic batch
        input_shape = self._input_info[0]["shape"] if self._input_info else [1, 3, 640, 640]
        model_batch_size = input_shape[0] if input_shape[0] > 0 else 1
        supports_dynamic_batch = (model_batch_size == -1 or model_batch_size == 0)

        all_results = []
        total_start = time.perf_counter()

        if supports_dynamic_batch and len(images) > 1:
            # True batch inference
            logger.info(f"Using batch inference for {len(images)} images")
            # TODO: Implement true batch inference when model supports it
            # For now, fall back to sequential processing
            pass

        # Sequential processing (most models have fixed batch size)
        for idx, img in enumerate(images):
            results = self.predict(
                img,
                conf=conf,
                iou=iou,
                classes=classes,
                max_det=max_det,
                **kwargs
            )
            all_results.append(results)

        total_time = (time.perf_counter() - total_start) * 1000
        avg_time = total_time / len(images)

        # Trigger batch_end callback
        self._trigger_callback(
            CallbackEvent.BATCH_END,
            batch_size=len(images),
            latency_ms=total_time,
        )

        logger.info(f"Batch inference: {len(images)} images, {total_time:.1f}ms total, {avg_time:.1f}ms/img")

        return all_results

    def stream(
        self,
        source: Union[str, int],
        conf: float = 0.5,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        show: bool = False,
        save: bool = False,
        save_dir: str = "runs/detect",
        loop: bool = False,
        **kwargs
    ) -> Generator[Results, None, None]:
        """
        Stream inference on video or camera.

        Yields results frame by frame using a generator.

        Args:
            source: Video file path, camera index (0, 1, ...), or RTSP URL
            conf: Confidence threshold
            iou: NMS IoU threshold
            classes: Filter by class IDs (e.g., [0, 1, 2])
            max_det: Maximum detections per frame
            show: Display results in window
            save: Save results to video file
            save_dir: Directory to save results
            loop: Loop video file
            **kwargs: Additional arguments

        Yields:
            Results for each frame

        Examples:
            >>> model = ivit.load("yolov8n.onnx")
            >>>
            >>> # Process video file
            >>> for results in model.stream("video.mp4"):
            ...     print(f"Frame: {len(results)} detections")
            ...     results.show()
            >>>
            >>> # Process camera with display
            >>> for results in model.stream(0, show=True):
            ...     if cv2.waitKey(1) == ord('q'):
            ...         break
            >>>
            >>> # Save results
            >>> for results in model.stream("input.mp4", save=True):
            ...     pass  # Results saved automatically
        """
        import cv2

        # Open video source
        video = VideoSource(source, loop=loop)
        logger.info(f"Streaming from: {source}")
        logger.info(f"Resolution: {video.width}x{video.height}, FPS: {video.fps:.1f}")

        # Trigger stream_start callback
        self._trigger_callback(
            CallbackEvent.STREAM_START,
            source_fps=video.fps,
        )

        # Setup video writer if saving
        writer = None
        if save:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            if isinstance(source, int):
                output_name = f"camera_{source}.mp4"
            else:
                output_name = Path(source).stem + "_result.mp4"

            output_path = save_path / output_name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                video.fps,
                (video.width, video.height)
            )
            logger.info(f"Saving to: {output_path}")

        frame_idx = 0
        try:
            for frame in video:
                # Run inference
                results = self.predict(
                    frame,
                    conf_threshold=conf,
                    iou_threshold=iou,
                    classes=classes,
                    max_detections=max_det,
                    **kwargs
                )

                # Add frame info
                results.frame_idx = frame_idx
                results.source_fps = video.fps

                # Trigger stream_frame callback
                self._trigger_callback(
                    CallbackEvent.STREAM_FRAME,
                    frame_idx=frame_idx,
                    source_fps=video.fps,
                    results=results,
                    latency_ms=results.inference_time_ms,
                )

                # Show if requested
                if show:
                    vis = results.visualize()
                    cv2.imshow("iVIT Stream", vis)
                    key = cv2.waitKey(1)
                    if key == ord('q') or key == 27:  # q or ESC
                        break

                # Save if requested
                if writer:
                    vis = results.visualize()
                    writer.write(vis)

                frame_idx += 1
                yield results

        finally:
            # Trigger stream_end callback
            self._trigger_callback(
                CallbackEvent.STREAM_END,
                frame_idx=frame_idx,
                source_fps=video.fps,
            )

            # Cleanup
            video.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

            logger.info(f"Processed {frame_idx} frames")

    def warmup(self, iterations: int = 3, input_shape: tuple = None):
        """
        Warmup model by running dummy inferences.

        Args:
            iterations: Number of warmup iterations (default: 3)
            input_shape: Optional input shape override for dynamic models.
                         If None, uses model's input shape with dynamic dims
                         replaced by defaults (1 for batch, 640 for spatial).

        Examples:
            >>> model.warmup()  # Use default shape
            >>> model.warmup(5)  # 5 iterations
            >>> model.warmup(3, input_shape=(1, 3, 640, 640))  # Override shape
        """
        # Get input info
        input_info = self._input_info[0] if self._input_info else {"shape": [1, 3, 640, 640], "name": "input"}
        input_name = input_info["name"]

        # Determine input shape
        if input_shape is not None:
            # Use provided shape
            shape = list(input_shape)
        else:
            # Get shape from model and handle dynamic dimensions
            shape = list(input_info["shape"])

            # Replace dynamic dimensions (-1, 0) with sensible defaults
            for i, dim in enumerate(shape):
                if dim <= 0:
                    if i == 0:
                        # Batch dimension
                        shape[i] = 1
                    elif i == 1 and len(shape) == 4:
                        # Channel dimension (typically 3 for RGB)
                        shape[i] = 3
                    else:
                        # Spatial dimensions (default to 640)
                        shape[i] = 640

        # Validate shape
        if any(d <= 0 for d in shape):
            logger.warning(f"Cannot create warmup tensor with shape {shape}, skipping warmup")
            return

        # Create dummy input
        dummy = np.random.randn(*shape).astype(np.float32)

        for _ in range(iterations):
            self._handle.infer({input_name: dummy})

        logger.info(f"Warmup completed ({iterations} iterations, shape={tuple(shape)})")

    def _preprocess(
        self,
        image: np.ndarray
    ) -> tuple:
        """Preprocess image for inference using configured preprocessor."""
        # Get target size from model input
        input_shape = self._input_info[0]["shape"] if self._input_info else [1, 3, 640, 640]
        if len(input_shape) == 4:
            _, _, h, w = input_shape
            # Handle dynamic dimensions
            if h <= 0:
                h = 640
            if w <= 0:
                w = 640
        else:
            h, w = 640, 640

        # Use configured preprocessor
        if self._preprocessor is not None:
            return self._preprocessor.process(image, (h, w))

        # Fallback: inline letterbox preprocessing
        import cv2
        orig_h, orig_w = image.shape[:2]
        scale = min(w / orig_w, h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_w, pad_h = (w - new_w) // 2, (h - new_h) // 2

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((h, w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        tensor = padded.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, 0)

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
        """Postprocess model outputs using configured postprocessor."""
        # Use configured postprocessor
        if self._postprocessor is not None:
            return self._postprocessor.process(
                outputs,
                orig_size,
                preprocess_info,
                config,
                labels=self._labels,
            )

        # Fallback: inline YOLO postprocessing
        from .types import Detection, BBox

        results = Results()
        output = list(outputs.values())[0]

        if len(output.shape) == 3:
            if output.shape[1] < output.shape[2]:
                output = output.transpose(0, 2, 1)

            output = output[0]
            scale = preprocess_info["scale"]
            pad_w = preprocess_info["pad_w"]
            pad_h = preprocess_info["pad_h"]

            detections = []
            for det in output:
                cx, cy, w, h = det[:4]
                scores = det[4:]

                if len(scores) == 1:
                    class_id = 0
                    confidence = float(scores[0])
                else:
                    class_id = int(np.argmax(scores))
                    confidence = float(scores[class_id])

                if confidence < config.conf_threshold:
                    continue
                if config.classes is not None and class_id not in config.classes:
                    continue

                x1 = (cx - w / 2 - pad_w) / scale
                y1 = (cy - h / 2 - pad_h) / scale
                x2 = (cx + w / 2 - pad_w) / scale
                y2 = (cy + h / 2 - pad_h) / scale

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

    # =========================================================================
    # Test Time Augmentation (TTA)
    # =========================================================================

    def predict_tta(
        self,
        source: Union[str, np.ndarray],
        conf: float = None,
        iou: float = None,
        classes: Optional[List[int]] = None,
        max_det: int = None,
        augments: Optional[List[str]] = None,
        scales: Optional[List[float]] = None,
        merge_iou: float = 0.6,
        **kwargs
    ) -> Results:
        """
        Run inference with Test Time Augmentation (TTA).

        TTA improves accuracy by running inference on multiple augmented
        versions of the input and merging the results.

        Args:
            source: Input image (path or numpy array)
            conf: Confidence threshold (default: 0.5)
            iou: NMS IoU threshold (default: 0.45)
            classes: Filter by class IDs
            max_det: Maximum detections (default: 300)
            augments: List of augmentations to apply. Options:
                - "hflip": Horizontal flip
                - "vflip": Vertical flip
                - "rotate90": Rotate 90 degrees
                - "rotate180": Rotate 180 degrees
                - "rotate270": Rotate 270 degrees
                Default: ["original", "hflip"]
            scales: List of scale factors for multi-scale inference.
                Default: [1.0] (no scaling)
                Example: [0.8, 1.0, 1.2] for multi-scale
            merge_iou: IoU threshold for merging overlapping boxes (default: 0.6)
            **kwargs: Additional arguments passed to predict()

        Returns:
            Merged inference results with improved accuracy

        Examples:
            >>> # Basic TTA with horizontal flip
            >>> results = model.predict_tta("image.jpg")
            >>>
            >>> # Multi-scale TTA
            >>> results = model.predict_tta("image.jpg", scales=[0.8, 1.0, 1.2])
            >>>
            >>> # Custom augmentations
            >>> results = model.predict_tta(
            ...     "image.jpg",
            ...     augments=["original", "hflip", "rotate90"],
            ...     scales=[0.8, 1.0, 1.2]
            ... )
        """
        import cv2

        # Handle parameters
        conf = conf if conf is not None else 0.5
        iou = iou if iou is not None else 0.45
        max_det = max_det if max_det is not None else 300
        augments = augments if augments is not None else ["original", "hflip"]
        scales = scales if scales is not None else [1.0]

        # Ensure "original" is always first if not specified
        if "original" not in augments:
            augments = ["original"] + augments

        # Load image if path
        if isinstance(source, str):
            image = cv2.imread(source)
            if image is None:
                raise ValueError(f"Failed to load image: {source}")
        elif isinstance(source, np.ndarray):
            image = source.copy()
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        orig_h, orig_w = image.shape[:2]
        all_detections = []

        logger.info(f"TTA: {len(augments)} augmentations x {len(scales)} scales = {len(augments) * len(scales)} inferences")

        # Run inference for each augmentation and scale combination
        for scale in scales:
            # Scale image if needed
            if scale != 1.0:
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                scaled_image = cv2.resize(image, (new_w, new_h))
            else:
                scaled_image = image
                new_w, new_h = orig_w, orig_h

            for aug_name in augments:
                # Apply augmentation
                aug_image, inverse_fn = self._apply_tta_augmentation(scaled_image, aug_name)

                # Run inference with lower threshold for TTA
                results = self.predict(
                    aug_image,
                    conf=conf * 0.8,  # Lower threshold, merge will filter
                    iou=iou,
                    classes=classes,
                    max_det=max_det * 2,  # More detections, merge will reduce
                    **kwargs
                )

                # Transform detections back to original coordinates
                for det in results.detections:
                    # Inverse augmentation transform
                    x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
                    x1, y1, x2, y2 = inverse_fn(x1, y1, x2, y2, new_w, new_h)

                    # Inverse scale transform
                    if scale != 1.0:
                        x1 = x1 / scale
                        y1 = y1 / scale
                        x2 = x2 / scale
                        y2 = y2 / scale

                    # Clip to original image bounds
                    x1 = max(0, min(x1, orig_w))
                    y1 = max(0, min(y1, orig_h))
                    x2 = max(0, min(x2, orig_w))
                    y2 = max(0, min(y2, orig_h))

                    from .types import Detection, BBox
                    all_detections.append(Detection(
                        bbox=BBox(x1, y1, x2, y2),
                        class_id=det.class_id,
                        label=det.label,
                        confidence=det.confidence,
                    ))

        # Merge overlapping detections using Weighted Box Fusion (WBF) style merging
        merged_detections = self._merge_tta_detections(
            all_detections,
            merge_iou=merge_iou,
            conf_threshold=conf,
            max_detections=max_det,
        )

        # Create final results
        final_results = Results()
        final_results.detections = merged_detections
        final_results.image_size = (orig_h, orig_w)
        final_results._original_image = image
        final_results.device_used = self._config.device

        logger.info(f"TTA: {len(all_detections)} raw detections -> {len(merged_detections)} merged")

        return final_results

    def _apply_tta_augmentation(
        self,
        image: np.ndarray,
        aug_name: str
    ) -> tuple:
        """
        Apply TTA augmentation and return inverse transform function.

        Args:
            image: Input image
            aug_name: Augmentation name

        Returns:
            (augmented_image, inverse_transform_fn)
        """
        import cv2

        h, w = image.shape[:2]

        if aug_name == "original":
            # No augmentation
            def inverse(x1, y1, x2, y2, w, h):
                return x1, y1, x2, y2
            return image, inverse

        elif aug_name == "hflip":
            # Horizontal flip
            aug_image = cv2.flip(image, 1)

            def inverse(x1, y1, x2, y2, w, h):
                return w - x2, y1, w - x1, y2
            return aug_image, inverse

        elif aug_name == "vflip":
            # Vertical flip
            aug_image = cv2.flip(image, 0)

            def inverse(x1, y1, x2, y2, w, h):
                return x1, h - y2, x2, h - y1
            return aug_image, inverse

        elif aug_name == "rotate90":
            # Rotate 90 degrees clockwise
            aug_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            def inverse(x1, y1, x2, y2, w, h):
                # After 90 CW: (x,y) -> (h-1-y, x), so inverse is (x,y) -> (y, w-1-x)
                return y1, w - x2, y2, w - x1
            return aug_image, inverse

        elif aug_name == "rotate180":
            # Rotate 180 degrees
            aug_image = cv2.rotate(image, cv2.ROTATE_180)

            def inverse(x1, y1, x2, y2, w, h):
                return w - x2, h - y2, w - x1, h - y1
            return aug_image, inverse

        elif aug_name == "rotate270":
            # Rotate 270 degrees clockwise (90 counter-clockwise)
            aug_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            def inverse(x1, y1, x2, y2, w, h):
                return h - y2, x1, h - y1, x2
            return aug_image, inverse

        else:
            logger.warning(f"Unknown augmentation: {aug_name}, using original")
            def inverse(x1, y1, x2, y2, w, h):
                return x1, y1, x2, y2
            return image, inverse

    def _merge_tta_detections(
        self,
        detections: List,
        merge_iou: float = 0.6,
        conf_threshold: float = 0.5,
        max_detections: int = 300,
    ) -> List:
        """
        Merge TTA detections using a Weighted Box Fusion style approach.

        Detections from different augmentations that overlap significantly
        are merged, and their confidence scores are boosted.

        Args:
            detections: All detections from TTA
            merge_iou: IoU threshold for considering boxes as the same object
            conf_threshold: Final confidence threshold
            max_detections: Maximum number of detections to return

        Returns:
            List of merged detections
        """
        from .types import Detection, BBox

        if not detections:
            return []

        # Group detections by class
        class_detections = {}
        for det in detections:
            if det.class_id not in class_detections:
                class_detections[det.class_id] = []
            class_detections[det.class_id].append(det)

        merged = []

        for class_id, class_dets in class_detections.items():
            # Sort by confidence
            class_dets = sorted(class_dets, key=lambda x: x.confidence, reverse=True)

            while class_dets:
                best = class_dets.pop(0)

                # Find overlapping boxes
                cluster = [best]
                remaining = []

                for det in class_dets:
                    if best.bbox.iou(det.bbox) >= merge_iou:
                        cluster.append(det)
                    else:
                        remaining.append(det)

                class_dets = remaining

                # Merge cluster using weighted average
                if len(cluster) == 1:
                    merged_det = cluster[0]
                else:
                    # Weighted average of boxes (weighted by confidence)
                    total_weight = sum(d.confidence for d in cluster)
                    avg_x1 = sum(d.bbox.x1 * d.confidence for d in cluster) / total_weight
                    avg_y1 = sum(d.bbox.y1 * d.confidence for d in cluster) / total_weight
                    avg_x2 = sum(d.bbox.x2 * d.confidence for d in cluster) / total_weight
                    avg_y2 = sum(d.bbox.y2 * d.confidence for d in cluster) / total_weight

                    # Boost confidence based on number of detections
                    # More detections = higher confidence (up to a limit)
                    boost_factor = min(len(cluster) / 2.0, 1.5)
                    merged_conf = min(best.confidence * boost_factor, 0.99)

                    merged_det = Detection(
                        bbox=BBox(avg_x1, avg_y1, avg_x2, avg_y2),
                        class_id=class_id,
                        label=best.label,
                        confidence=merged_conf,
                    )

                # Filter by confidence threshold
                if merged_det.confidence >= conf_threshold:
                    merged.append(merged_det)

        # Sort by confidence and limit
        merged = sorted(merged, key=lambda x: x.confidence, reverse=True)
        return merged[:max_detections]


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
