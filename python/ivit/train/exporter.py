"""
Model exporter for converting trained models to deployment formats.

Supports ONNX, TorchScript, and backend-specific formats.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Export trained models to various deployment formats.

    Supports:
    - ONNX (cross-platform)
    - TorchScript (PyTorch native)
    - OpenVINO IR (Intel optimization)
    - TensorRT Engine (NVIDIA optimization)

    Args:
        model: PyTorch model to export
        device: Device the model is on

    Examples:
        >>> exporter = ModelExporter(model, device)
        >>> exporter.export("model.onnx", format="onnx")
        >>> exporter.export("model.xml", format="openvino", quantize="int8")
    """

    def __init__(self, model: Any, device: Any):
        self.model = model
        self.device = device

    def export(
        self,
        path: str,
        format: str = "onnx",
        optimize_for: Optional[str] = None,
        quantize: Optional[str] = None,
        input_shape: tuple = (1, 3, 224, 224),
        opset_version: int = 17,
        class_names: Optional[List[str]] = None,
        calibration_data: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> str:
        """
        Export model to specified format.

        Args:
            path: Output path
            format: Export format ("onnx", "torchscript", "openvino", "tensorrt")
            optimize_for: Target hardware optimization
            quantize: Quantization mode ("fp16", "int8")
            input_shape: Model input shape
            opset_version: ONNX opset version
            class_names: Class names for metadata
            calibration_data: Data for INT8 calibration

        Returns:
            Path to exported model
        """
        format = format.lower()

        # Create output directory
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if format == "onnx":
            return self._export_onnx(
                path, input_shape, opset_version, quantize, class_names
            )
        elif format == "torchscript":
            return self._export_torchscript(path, input_shape)
        elif format == "openvino":
            return self._export_openvino(
                path, input_shape, quantize, calibration_data
            )
        elif format == "tensorrt":
            return self._export_tensorrt(
                path, input_shape, quantize
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_onnx(
        self,
        path: str,
        input_shape: tuple,
        opset_version: int,
        quantize: Optional[str],
        class_names: Optional[List[str]],
    ) -> str:
        """Export to ONNX format."""
        import torch

        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)

        # Get input/output names
        input_names = ["input"]
        output_names = ["output"]

        # Dynamic axes for batch size
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

        # Export to ONNX
        logger.info(f"Exporting to ONNX: {path}")
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

        # Apply quantization if requested
        if quantize == "fp16":
            path = self._quantize_onnx_fp16(path)
        elif quantize == "int8":
            logger.warning("INT8 quantization requires calibration data")

        # Save class names
        if class_names:
            labels_path = Path(path).with_suffix(".txt")
            with open(labels_path, 'w') as f:
                f.write('\n'.join(class_names))
            logger.info(f"Saved labels to {labels_path}")

        logger.info(f"ONNX export complete: {path}")
        return path

    def _quantize_onnx_fp16(self, path: str) -> str:
        """Convert ONNX model to FP16."""
        try:
            from onnxconverter_common import float16
            import onnx

            model = onnx.load(path)
            model_fp16 = float16.convert_float_to_float16(model)

            fp16_path = str(Path(path).with_suffix('')) + "_fp16.onnx"
            onnx.save(model_fp16, fp16_path)

            logger.info(f"FP16 quantization complete: {fp16_path}")
            return fp16_path

        except ImportError:
            logger.warning("onnxconverter-common not installed, skipping FP16 conversion")
            return path

    def _export_torchscript(
        self,
        path: str,
        input_shape: tuple,
    ) -> str:
        """Export to TorchScript format."""
        import torch

        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)

        # Trace model
        logger.info(f"Exporting to TorchScript: {path}")
        traced = torch.jit.trace(self.model, dummy_input)
        traced.save(path)

        logger.info(f"TorchScript export complete: {path}")
        return path

    def _export_openvino(
        self,
        path: str,
        input_shape: tuple,
        quantize: Optional[str],
        calibration_data: Optional[List[np.ndarray]],
    ) -> str:
        """Export to OpenVINO IR format."""
        import torch

        # First export to ONNX
        onnx_path = str(Path(path).with_suffix('.onnx'))
        self._export_onnx(onnx_path, input_shape, 17, None, None)

        try:
            import openvino as ov
            from openvino import Core
            from openvino.runtime import serialize

            core = Core()
            logger.info(f"Converting to OpenVINO IR: {path}")

            # Read ONNX model
            model = core.read_model(onnx_path)

            # Apply quantization if requested
            if quantize == "int8" and calibration_data is not None:
                model = self._quantize_openvino_int8(model, calibration_data)
            elif quantize == "fp16":
                # Compress weights to FP16 during serialization
                xml_path = str(Path(path).with_suffix('.xml'))
                ov.save_model(model, xml_path, compress_to_fp16=True)
                logger.info(f"OpenVINO IR export complete (FP16): {xml_path}")
                return xml_path

            # Serialize to IR format
            xml_path = str(Path(path).with_suffix('.xml'))
            serialize(model, xml_path)

            logger.info(f"OpenVINO IR export complete: {xml_path}")
            return xml_path

        except ImportError:
            logger.warning("OpenVINO not installed, returning ONNX path")
            return onnx_path

    def _quantize_openvino_int8(self, model, calibration_data: List[np.ndarray]):
        """Apply INT8 quantization using NNCF."""
        try:
            import nncf

            # Create calibration dataset
            def transform_fn(data_item):
                return np.expand_dims(data_item, 0)

            calibration_dataset = nncf.Dataset(calibration_data, transform_fn)

            # Quantize model
            quantized_model = nncf.quantize(
                model,
                calibration_dataset,
                preset=nncf.QuantizationPreset.MIXED,
            )

            logger.info("INT8 quantization complete")
            return quantized_model

        except ImportError:
            logger.warning("NNCF not installed, skipping INT8 quantization")
            return model

    def _export_tensorrt(
        self,
        path: str,
        input_shape: tuple,
        quantize: Optional[str],
    ) -> str:
        """Export to TensorRT engine format."""
        # First export to ONNX
        onnx_path = str(Path(path).with_suffix('.onnx'))
        self._export_onnx(onnx_path, input_shape, 17, None, None)

        try:
            import tensorrt as trt

            logger.info(f"Converting to TensorRT: {path}")

            # Create builder and network
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)

            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    raise RuntimeError("Failed to parse ONNX model")

            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

            if quantize == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Enabled FP16 precision")
            elif quantize == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                logger.warning("INT8 requires calibrator, falling back to FP16")
                config.set_flag(trt.BuilderFlag.FP16)

            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")

            # Save engine
            engine_path = str(Path(path).with_suffix('.engine'))
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)

            logger.info(f"TensorRT export complete: {engine_path}")
            return engine_path

        except ImportError:
            logger.warning("TensorRT not installed, returning ONNX path")
            return onnx_path


def export_model(
    model: Any,
    path: str,
    format: str = "onnx",
    input_shape: tuple = (1, 3, 224, 224),
    **kwargs
) -> str:
    """
    Convenience function to export a model.

    Args:
        model: Model to export
        path: Output path
        format: Export format
        input_shape: Input shape
        **kwargs: Additional arguments

    Returns:
        Path to exported model

    Examples:
        >>> from ivit.train import export_model
        >>> export_model(model, "model.onnx", input_shape=(1, 3, 640, 640))
    """
    import torch
    device = next(model.parameters()).device
    exporter = ModelExporter(model, device)
    return exporter.export(path, format, input_shape=input_shape, **kwargs)
