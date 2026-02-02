"""
Model Zoo Registry - Model catalog and download management.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model information in the zoo."""
    name: str
    task: str  # detect, classify, segment, pose
    description: str
    input_size: tuple  # (H, W)
    num_classes: int
    formats: List[str]  # Available formats: onnx, openvino, tensorrt
    url: Optional[str] = None
    source: str = "ultralytics"  # ultralytics, torchvision, custom
    license: str = "AGPL-3.0"
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


# Model Zoo Registry
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # =========================================================================
    # YOLOv8 Detection Models
    # =========================================================================
    "yolov8n": ModelInfo(
        name="yolov8n",
        task="detect",
        description="YOLOv8 Nano - Fastest, smallest model for edge devices",
        input_size=(640, 640),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        source="ultralytics",
        metrics={"mAP50-95": 37.3, "params_m": 3.2, "flops_g": 8.7},
        tags=["yolo", "detection", "fast", "edge"],
    ),
    "yolov8s": ModelInfo(
        name="yolov8s",
        task="detect",
        description="YOLOv8 Small - Balance of speed and accuracy",
        input_size=(640, 640),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
        source="ultralytics",
        metrics={"mAP50-95": 44.9, "params_m": 11.2, "flops_g": 28.6},
        tags=["yolo", "detection", "balanced"],
    ),
    "yolov8m": ModelInfo(
        name="yolov8m",
        task="detect",
        description="YOLOv8 Medium - Higher accuracy",
        input_size=(640, 640),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
        source="ultralytics",
        metrics={"mAP50-95": 50.2, "params_m": 25.9, "flops_g": 78.9},
        tags=["yolo", "detection", "accurate"],
    ),
    "yolov8l": ModelInfo(
        name="yolov8l",
        task="detect",
        description="YOLOv8 Large - High accuracy for server deployment",
        input_size=(640, 640),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt",
        source="ultralytics",
        metrics={"mAP50-95": 52.9, "params_m": 43.7, "flops_g": 165.2},
        tags=["yolo", "detection", "accurate", "server"],
    ),
    "yolov8x": ModelInfo(
        name="yolov8x",
        task="detect",
        description="YOLOv8 Extra Large - Maximum accuracy",
        input_size=(640, 640),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
        source="ultralytics",
        metrics={"mAP50-95": 53.9, "params_m": 68.2, "flops_g": 257.8},
        tags=["yolo", "detection", "accurate", "server"],
    ),

    # =========================================================================
    # YOLOv8 Segmentation Models
    # =========================================================================
    "yolov8n-seg": ModelInfo(
        name="yolov8n-seg",
        task="segment",
        description="YOLOv8 Nano Segmentation",
        input_size=(640, 640),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt",
        source="ultralytics",
        metrics={"mAP50-95_box": 36.7, "mAP50-95_mask": 30.5},
        tags=["yolo", "segmentation", "fast", "edge"],
    ),
    "yolov8s-seg": ModelInfo(
        name="yolov8s-seg",
        task="segment",
        description="YOLOv8 Small Segmentation",
        input_size=(640, 640),
        num_classes=80,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.pt",
        source="ultralytics",
        metrics={"mAP50-95_box": 44.6, "mAP50-95_mask": 36.8},
        tags=["yolo", "segmentation", "balanced"],
    ),

    # =========================================================================
    # YOLOv8 Classification Models
    # =========================================================================
    "yolov8n-cls": ModelInfo(
        name="yolov8n-cls",
        task="classify",
        description="YOLOv8 Nano Classification",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-cls.pt",
        source="ultralytics",
        metrics={"top1": 69.0, "top5": 88.3},
        tags=["yolo", "classification", "fast", "edge"],
    ),
    "yolov8s-cls": ModelInfo(
        name="yolov8s-cls",
        task="classify",
        description="YOLOv8 Small Classification",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-cls.pt",
        source="ultralytics",
        metrics={"top1": 73.8, "top5": 91.7},
        tags=["yolo", "classification", "balanced"],
    ),

    # =========================================================================
    # YOLOv8 Pose Models
    # =========================================================================
    "yolov8n-pose": ModelInfo(
        name="yolov8n-pose",
        task="pose",
        description="YOLOv8 Nano Pose Estimation",
        input_size=(640, 640),
        num_classes=1,  # person
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt",
        source="ultralytics",
        metrics={"mAP50-95_pose": 50.4},
        tags=["yolo", "pose", "fast", "edge"],
    ),
    "yolov8s-pose": ModelInfo(
        name="yolov8s-pose",
        task="pose",
        description="YOLOv8 Small Pose Estimation",
        input_size=(640, 640),
        num_classes=1,
        formats=["onnx", "openvino", "tensorrt"],
        url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt",
        source="ultralytics",
        metrics={"mAP50-95_pose": 60.0},
        tags=["yolo", "pose", "balanced"],
    ),

    # =========================================================================
    # Classic Models
    # =========================================================================
    "resnet50": ModelInfo(
        name="resnet50",
        task="classify",
        description="ResNet-50 ImageNet Classification",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"top1": 76.1, "top5": 92.9},
        tags=["classification", "imagenet", "classic"],
    ),
    "mobilenetv3": ModelInfo(
        name="mobilenetv3",
        task="classify",
        description="MobileNetV3 Large - Efficient mobile classification",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"top1": 74.0, "top5": 91.3},
        tags=["classification", "mobile", "efficient"],
    ),
    "efficientnet-b0": ModelInfo(
        name="efficientnet-b0",
        task="classify",
        description="EfficientNet-B0 - Efficient scaling",
        input_size=(224, 224),
        num_classes=1000,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="Apache-2.0",
        metrics={"top1": 77.1, "top5": 93.3},
        tags=["classification", "efficient"],
    ),

    # =========================================================================
    # Semantic Segmentation Models
    # =========================================================================
    "deeplabv3-resnet50": ModelInfo(
        name="deeplabv3-resnet50",
        task="segment",
        description="DeepLabV3 ResNet-50 - Semantic segmentation",
        input_size=(520, 520),
        num_classes=21,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"mIoU": 66.4},
        tags=["segmentation", "semantic", "voc", "classic"],
    ),
    "deeplabv3-mobilenetv3": ModelInfo(
        name="deeplabv3-mobilenetv3",
        task="segment",
        description="DeepLabV3 MobileNetV3 - Lightweight semantic segmentation",
        input_size=(520, 520),
        num_classes=21,
        formats=["onnx", "openvino"],
        source="torchvision",
        license="BSD-3-Clause",
        metrics={"mIoU": 60.3},
        tags=["segmentation", "semantic", "voc", "mobile", "efficient"],
    ),
}


def _get_cache_dir() -> Path:
    """Get model cache directory."""
    cache_dir = os.environ.get("IVIT_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)

    # Default: ~/.cache/ivit/models
    return Path.home() / ".cache" / "ivit" / "models"


def list_models(task: Optional[str] = None) -> List[str]:
    """
    List available models in the zoo.

    Args:
        task: Filter by task type (detect, classify, segment, pose)

    Returns:
        List of model names

    Examples:
        >>> ivit.zoo.list_models()
        ['yolov8n', 'yolov8s', 'yolov8m', ...]
        >>> ivit.zoo.list_models(task="detect")
        ['yolov8n', 'yolov8s', ...]
    """
    models = []
    for name, info in MODEL_REGISTRY.items():
        if task is None or info.task == task:
            models.append(name)
    return sorted(models)


def search(query: str) -> List[str]:
    """
    Search models by name or tag.

    Args:
        query: Search query

    Returns:
        List of matching model names

    Examples:
        >>> ivit.zoo.search("yolo")
        ['yolov8n', 'yolov8s', 'yolov8m', ...]
        >>> ivit.zoo.search("edge")
        ['yolov8n', 'yolov8n-seg', 'yolov8n-cls', ...]
    """
    query = query.lower()
    results = []

    for name, info in MODEL_REGISTRY.items():
        # Match name
        if query in name.lower():
            results.append(name)
            continue

        # Match tags
        if any(query in tag.lower() for tag in info.tags):
            results.append(name)
            continue

        # Match description
        if query in info.description.lower():
            results.append(name)

    return sorted(set(results))


def get_model_info(name: str) -> ModelInfo:
    """
    Get detailed model information.

    Args:
        name: Model name

    Returns:
        ModelInfo object

    Raises:
        KeyError: If model not found

    Examples:
        >>> info = ivit.zoo.get_model_info("yolov8n")
        >>> print(f"Task: {info.task}, Input: {info.input_size}")
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model not found: {name}. Use ivit.zoo.list_models() to see available models.")
    return MODEL_REGISTRY[name]


def download(
    name: str,
    format: str = "onnx",
    force: bool = False,
) -> Path:
    """
    Download model from zoo.

    Args:
        name: Model name
        format: Model format (onnx, openvino, tensorrt)
        force: Force re-download

    Returns:
        Path to downloaded model

    Examples:
        >>> path = ivit.zoo.download("yolov8n")
        >>> print(path)
        /home/user/.cache/ivit/models/yolov8n.onnx
    """
    import requests
    from tqdm import tqdm

    info = get_model_info(name)

    if format not in info.formats:
        raise ValueError(f"Format '{format}' not available for {name}. Available: {info.formats}")

    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine file path
    if format == "onnx":
        model_path = cache_dir / f"{name}.onnx"
    elif format == "openvino":
        model_path = cache_dir / f"{name}_openvino" / f"{name}.xml"
    elif format == "tensorrt":
        model_path = cache_dir / f"{name}.engine"
    else:
        model_path = cache_dir / f"{name}.{format}"

    # Check if already exists
    if model_path.exists() and not force:
        logger.info(f"Model already cached: {model_path}")
        return model_path

    # Download PT file first, then convert
    if info.url and info.source == "ultralytics":
        pt_path = cache_dir / f"{name}.pt"

        if not pt_path.exists() or force:
            logger.info(f"Downloading {name} from {info.url}...")

            response = requests.get(info.url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(pt_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"Downloaded to: {pt_path}")

        # Convert to requested format
        if format == "onnx":
            model_path = _convert_to_onnx(pt_path, model_path, info)

    elif info.source == "torchvision":
        if format == "onnx":
            model_path = _export_torchvision_onnx(name, model_path, info)

    return model_path


def _convert_to_onnx(pt_path: Path, onnx_path: Path, info: ModelInfo) -> Path:
    """Convert PyTorch model to ONNX."""
    try:
        from ultralytics import YOLO

        logger.info(f"Converting to ONNX: {onnx_path}")
        model = YOLO(str(pt_path))
        model.export(format="onnx", imgsz=info.input_size[0])

        # Move to cache location
        exported = pt_path.with_suffix(".onnx")
        if exported != onnx_path:
            exported.rename(onnx_path)

        logger.info(f"Converted: {onnx_path}")
        return onnx_path

    except ImportError:
        raise ImportError(
            "Ultralytics required for model conversion. "
            "Install with: pip install ultralytics"
        )


_TORCHVISION_MODEL_MAP = {
    "resnet50": ("resnet50", "ResNet50_Weights.DEFAULT"),
    "mobilenetv3": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights.DEFAULT"),
    "efficientnet-b0": ("efficientnet_b0", "EfficientNet_B0_Weights.DEFAULT"),
    "deeplabv3-resnet50": ("deeplabv3_resnet50", "DeepLabV3_ResNet50_Weights.DEFAULT"),
    "deeplabv3-mobilenetv3": ("deeplabv3_mobilenet_v3_large", "DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT"),
}


def _export_torchvision_onnx(name: str, onnx_path: Path, info: ModelInfo) -> Path:
    """Export a torchvision model to ONNX."""
    try:
        import torch
        import torchvision.models as models
        import torchvision.models.segmentation as seg_models
    except ImportError:
        raise ImportError(
            "PyTorch and torchvision required for torchvision model export. "
            "Install with: pip install torch torchvision"
        )

    if name not in _TORCHVISION_MODEL_MAP:
        raise ValueError(f"No torchvision mapping for model: {name}")

    func_name, weights_name = _TORCHVISION_MODEL_MAP[name]

    logger.info(f"Loading torchvision model: {func_name}")

    # Segmentation models live in torchvision.models.segmentation
    if hasattr(seg_models, func_name):
        module = seg_models
    else:
        module = models

    weights_enum = None
    for part in weights_name.split("."):
        weights_enum = getattr(module if weights_enum is None else weights_enum, part)

    model = getattr(module, func_name)(weights=weights_enum)
    model.eval()

    h, w = info.input_size
    dummy = torch.randn(1, 3, h, w)

    logger.info(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    logger.info(f"Exported: {onnx_path}")
    return onnx_path


def load(
    name: str,
    device: str = "auto",
    format: str = "onnx",
    **kwargs
):
    """
    Load model from zoo.

    Args:
        name: Model name or path
        device: Target device
        format: Model format
        **kwargs: Additional arguments for ivit.load()

    Returns:
        Loaded model ready for inference

    Examples:
        >>> model = ivit.zoo.load("yolov8n")
        >>> results = model("image.jpg")
        >>>
        >>> model = ivit.zoo.load("yolov8n", device="cuda:0")
        >>> model = ivit.zoo.load("yolov8n", device="npu")
    """
    import ivit

    # Check if it's a zoo model or a path
    if name in MODEL_REGISTRY:
        # Download if needed
        model_path = download(name, format=format)
    else:
        # Treat as path
        model_path = Path(name)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {name}")

    # Load with ivit
    return ivit.load(str(model_path), device=device, **kwargs)


def print_models(task: Optional[str] = None):
    """
    Print formatted model list.

    Args:
        task: Filter by task type
    """
    models = list_models(task)

    print()
    print("=" * 80)
    print("iVIT Model Zoo")
    print("=" * 80)
    print()
    print(f"{'Model':<20} {'Task':<12} {'Input':<12} {'Description':<30}")
    print("-" * 80)

    for name in models:
        info = MODEL_REGISTRY[name]
        input_str = f"{info.input_size[0]}x{info.input_size[1]}"
        desc = info.description[:28] + ".." if len(info.description) > 30 else info.description
        print(f"{name:<20} {info.task:<12} {input_str:<12} {desc:<30}")

    print("-" * 80)
    print(f"Total: {len(models)} models")
    print()
    print("Usage: model = ivit.zoo.load('yolov8n')")
    print()
