"""
Runtime configuration classes for hardware-specific settings.

Each backend has its own configuration class with tunable parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class OpenVINOConfig:
    """
    OpenVINO runtime configuration.

    Provides access to Intel-specific optimizations.

    Examples:
        >>> model.configure_openvino(
        ...     performance_mode="LATENCY",
        ...     num_streams=4,
        ...     inference_precision="FP16",
        ... )
    """
    # Performance hint
    performance_mode: str = "LATENCY"
    """Performance hint: "LATENCY" (default), "THROUGHPUT", or "CUMULATIVE_THROUGHPUT" """

    # Number of inference streams
    num_streams: int = 1
    """Number of inference streams for parallel execution"""

    # Inference precision
    inference_precision: str = "FP32"
    """Inference precision: "FP32", "FP16", or "INT8" """

    # CPU-specific settings
    enable_cpu_pinning: bool = False
    """Pin threads to CPU cores for better cache utilization"""

    num_threads: int = 0
    """Number of CPU threads (0 = auto)"""

    # NPU-specific settings
    npu_compilation_mode: str = "DefaultHW"
    """NPU compilation mode: "DefaultHW", "DefaultSW" """

    # Cache settings
    cache_dir: Optional[str] = None
    """Directory for model compilation cache"""

    # Device-specific properties
    device_properties: Dict[str, Any] = field(default_factory=dict)
    """Additional device-specific properties"""

    def to_ov_config(self) -> Dict[str, Any]:
        """Convert to OpenVINO config dict."""
        config = {}

        # Performance hint
        config["PERFORMANCE_HINT"] = self.performance_mode

        # Streams
        if self.num_streams > 0:
            config["NUM_STREAMS"] = str(self.num_streams)

        # Precision
        if self.inference_precision != "FP32":
            config["INFERENCE_PRECISION_HINT"] = self.inference_precision

        # CPU pinning
        if self.enable_cpu_pinning:
            config["ENABLE_CPU_PINNING"] = "YES"

        # Threads
        if self.num_threads > 0:
            config["INFERENCE_NUM_THREADS"] = str(self.num_threads)

        # Cache
        if self.cache_dir:
            config["CACHE_DIR"] = self.cache_dir

        # Additional properties
        config.update(self.device_properties)

        return config


@dataclass
class TensorRTConfig:
    """
    TensorRT runtime configuration.

    Provides access to NVIDIA-specific optimizations.

    Examples:
        >>> model.configure_tensorrt(
        ...     workspace_size=1 << 30,  # 1GB
        ...     dla_core=0,
        ...     enable_fp16=True,
        ... )
    """
    # Workspace size in bytes
    workspace_size: int = 1 << 30  # 1GB default
    """Maximum workspace size for TensorRT (bytes)"""

    # DLA (Deep Learning Accelerator) settings
    dla_core: int = -1
    """DLA core to use (-1 = disabled, 0-1 on Jetson)"""

    allow_gpu_fallback: bool = True
    """Allow GPU fallback for unsupported DLA layers"""

    # Precision settings
    enable_fp16: bool = False
    """Enable FP16 inference"""

    enable_int8: bool = False
    """Enable INT8 inference (requires calibration)"""

    strict_type_constraints: bool = False
    """Enforce strict type constraints"""

    # Optimization settings
    builder_optimization_level: int = 3
    """Builder optimization level (0-5, higher = more optimization)"""

    enable_sparsity: bool = False
    """Enable sparsity optimizations (Ampere+)"""

    # Timing cache
    timing_cache_path: Optional[str] = None
    """Path to timing cache file"""

    # Engine settings
    max_batch_size: int = 1
    """Maximum batch size"""

    # Profiling
    enable_profiling: bool = False
    """Enable layer profiling"""

    def to_trt_config(self) -> Dict[str, Any]:
        """Convert to TensorRT config dict."""
        return {
            "workspace_size": self.workspace_size,
            "dla_core": self.dla_core,
            "allow_gpu_fallback": self.allow_gpu_fallback,
            "enable_fp16": self.enable_fp16,
            "enable_int8": self.enable_int8,
            "strict_type_constraints": self.strict_type_constraints,
            "builder_optimization_level": self.builder_optimization_level,
            "enable_sparsity": self.enable_sparsity,
            "timing_cache_path": self.timing_cache_path,
            "max_batch_size": self.max_batch_size,
            "enable_profiling": self.enable_profiling,
        }


@dataclass
class ONNXRuntimeConfig:
    """
    ONNX Runtime configuration.

    Provides access to cross-platform optimizations.

    Examples:
        >>> model.configure_onnxruntime(
        ...     num_threads=8,
        ...     graph_optimization_level="ORT_ENABLE_ALL",
        ...     enable_cuda_graph=True,
        ... )
    """
    # Thread settings
    num_threads: int = 0
    """Number of intra-op threads (0 = auto)"""

    inter_op_num_threads: int = 0
    """Number of inter-op threads (0 = auto)"""

    # Graph optimization
    graph_optimization_level: str = "ORT_ENABLE_ALL"
    """Graph optimization level: "ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL" """

    # Memory settings
    enable_mem_pattern: bool = True
    """Enable memory pattern optimization"""

    enable_cpu_mem_arena: bool = True
    """Enable CPU memory arena"""

    # CUDA settings
    cuda_device_id: int = 0
    """CUDA device ID"""

    cuda_mem_limit: int = 0
    """CUDA memory limit (0 = no limit)"""

    arena_extend_strategy: int = 0
    """Memory arena extend strategy (0 = default)"""

    enable_cuda_graph: bool = False
    """Enable CUDA graph capture (reduces kernel launch overhead)"""

    # Execution mode
    execution_mode: str = "ORT_SEQUENTIAL"
    """Execution mode: "ORT_SEQUENTIAL" or "ORT_PARALLEL" """

    def to_ort_config(self) -> Dict[str, Any]:
        """Convert to ONNX Runtime config dict."""
        return {
            "num_threads": self.num_threads,
            "inter_op_num_threads": self.inter_op_num_threads,
            "graph_optimization_level": self.graph_optimization_level,
            "enable_mem_pattern": self.enable_mem_pattern,
            "enable_cpu_mem_arena": self.enable_cpu_mem_arena,
            "cuda_device_id": self.cuda_device_id,
            "cuda_mem_limit": self.cuda_mem_limit,
            "arena_extend_strategy": self.arena_extend_strategy,
            "enable_cuda_graph": self.enable_cuda_graph,
            "execution_mode": self.execution_mode,
        }


@dataclass
class SNPEConfig:
    """
    Qualcomm SNPE runtime configuration.

    Provides access to Hexagon DSP/NPU optimizations.

    Examples:
        >>> model.configure_snpe(
        ...     runtime="dsp",
        ...     performance_profile="HIGH_PERFORMANCE",
        ... )
    """
    # Runtime selection
    runtime: str = "dsp"
    """Runtime: "cpu", "gpu", "dsp", "aip" """

    # Performance profile
    performance_profile: str = "DEFAULT"
    """Performance profile: "DEFAULT", "BALANCED", "HIGH_PERFORMANCE", "POWER_SAVER", "SUSTAINED_HIGH_PERFORMANCE" """

    # Buffer type
    use_user_supplied_buffers: bool = False
    """Use user-supplied buffers for zero-copy"""

    # Profiling
    enable_profiling: bool = False
    """Enable profiling"""

    profiling_level: str = "BASIC"
    """Profiling level: "OFF", "BASIC", "MODERATE", "DETAILED" """

    def to_snpe_config(self) -> Dict[str, Any]:
        """Convert to SNPE config dict."""
        return {
            "runtime": self.runtime,
            "performance_profile": self.performance_profile,
            "use_user_supplied_buffers": self.use_user_supplied_buffers,
            "enable_profiling": self.enable_profiling,
            "profiling_level": self.profiling_level,
        }
