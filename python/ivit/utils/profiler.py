"""
Performance profiling utilities.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import time
import json
import numpy as np


@dataclass
class ProfileReport:
    """Performance profile report."""

    model_name: str = ""
    device: str = ""
    backend: str = ""
    precision: str = ""
    input_shape: Tuple[int, ...] = ()
    iterations: int = 0

    # Latency (milliseconds)
    latency_mean: float = 0.0
    latency_median: float = 0.0
    latency_std: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    # Throughput
    throughput_fps: float = 0.0

    # Memory
    memory_mb: float = 0.0

    def __str__(self) -> str:
        """Format report as string."""
        lines = [
            "",
            "=" * 60,
            "                   Benchmark Report",
            "=" * 60,
            f"  Model:        {self.model_name}",
            f"  Device:       {self.device}",
            f"  Backend:      {self.backend}",
            f"  Precision:    {self.precision}",
            f"  Input Shape:  {self.input_shape}",
            "-" * 60,
            "  Latency:",
            f"    Mean:       {self.latency_mean:.2f} ms",
            f"    Median:     {self.latency_median:.2f} ms",
            f"    Std:        {self.latency_std:.2f} ms",
            f"    Min:        {self.latency_min:.2f} ms",
            f"    Max:        {self.latency_max:.2f} ms",
            f"    P95:        {self.latency_p95:.2f} ms",
            f"    P99:        {self.latency_p99:.2f} ms",
            "-" * 60,
            f"  Throughput:   {self.throughput_fps:.1f} FPS",
            f"  Memory:       {self.memory_mb:.1f} MB",
            "=" * 60,
            "",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "backend": self.backend,
            "precision": self.precision,
            "input_shape": list(self.input_shape),
            "iterations": self.iterations,
            "latency": {
                "mean_ms": self.latency_mean,
                "median_ms": self.latency_median,
                "std_ms": self.latency_std,
                "min_ms": self.latency_min,
                "max_ms": self.latency_max,
                "p95_ms": self.latency_p95,
                "p99_ms": self.latency_p99,
            },
            "throughput_fps": self.throughput_fps,
            "memory_mb": self.memory_mb,
        }

    def save(self, path: str) -> None:
        """Save report to file."""
        with open(path, "w") as f:
            f.write(self.to_json())


class Profiler:
    """
    Performance profiler.

    Examples:
        >>> profiler = Profiler()
        >>> report = profiler.benchmark(model, input_shape=(1, 3, 640, 640))
        >>> print(report)
    """

    def __init__(self):
        self._times: List[float] = []
        self._start_time: Optional[float] = None

    def benchmark(
        self,
        model,
        input_shape: Tuple[int, ...],
        iterations: int = 100,
        warmup: int = 10,
    ) -> ProfileReport:
        """
        Run benchmark on model.

        Args:
            model: Model to benchmark
            input_shape: Input shape (N, C, H, W)
            iterations: Number of iterations
            warmup: Warmup iterations

        Returns:
            Profile report
        """
        # Create dummy input
        dummy = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            model._runtime.infer(model._handle, {"input": dummy})

        # Benchmark
        self.reset()
        for _ in range(iterations):
            self.start()
            model._runtime.infer(model._handle, {"input": dummy})
            self.stop()

        # Calculate stats
        mean, median, std, min_t, max_t, p95, p99 = self.calculate_stats()

        report = ProfileReport(
            model_name=model.name,
            device=model.device,
            backend=model.backend,
            precision="fp32",  # TODO: detect from model
            input_shape=input_shape,
            iterations=iterations,
            latency_mean=mean,
            latency_median=median,
            latency_std=std,
            latency_min=min_t,
            latency_max=max_t,
            latency_p95=p95,
            latency_p99=p99,
            throughput_fps=1000.0 / mean if mean > 0 else 0,
        )

        return report

    def start(self) -> None:
        """Start timing."""
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop timing and record."""
        if self._start_time is not None:
            elapsed = (time.perf_counter() - self._start_time) * 1000
            self._times.append(elapsed)
            self._start_time = None

    def elapsed_ms(self) -> float:
        """Get last elapsed time in milliseconds."""
        return self._times[-1] if self._times else 0.0

    def reset(self) -> None:
        """Reset profiler."""
        self._times = []
        self._start_time = None

    @property
    def times(self) -> List[float]:
        """Get all recorded times."""
        return self._times.copy()

    def calculate_stats(self) -> Tuple[float, float, float, float, float, float, float]:
        """
        Calculate statistics.

        Returns:
            (mean, median, std, min, max, p95, p99)
        """
        if not self._times:
            return (0, 0, 0, 0, 0, 0, 0)

        times = np.array(self._times)
        return (
            float(np.mean(times)),
            float(np.median(times)),
            float(np.std(times)),
            float(np.min(times)),
            float(np.max(times)),
            float(np.percentile(times, 95)),
            float(np.percentile(times, 99)),
        )


class ScopedTimer:
    """
    Scoped timer for automatic timing.

    Examples:
        >>> profiler = Profiler()
        >>> with ScopedTimer(profiler):
        ...     model.predict(image)
        >>> print(f"Elapsed: {profiler.elapsed_ms():.2f} ms")
    """

    def __init__(self, profiler: Profiler):
        self._profiler = profiler

    def __enter__(self):
        self._profiler.start()
        return self

    def __exit__(self, *args):
        self._profiler.stop()
