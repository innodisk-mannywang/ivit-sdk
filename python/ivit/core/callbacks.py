"""
Callback system for iVIT-SDK.

Provides extensibility through event-driven callbacks.
"""

from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class CallbackEvent(str, Enum):
    """Supported callback events."""
    # Inference lifecycle
    PRE_PROCESS = "pre_process"
    POST_PROCESS = "post_process"
    INFER_START = "infer_start"
    INFER_END = "infer_end"

    # Batch processing
    BATCH_START = "batch_start"
    BATCH_END = "batch_end"

    # Stream processing
    STREAM_START = "stream_start"
    STREAM_FRAME = "stream_frame"
    STREAM_END = "stream_end"

    # Model lifecycle
    MODEL_LOADED = "model_loaded"
    MODEL_UNLOADED = "model_unloaded"


@dataclass
class CallbackContext:
    """
    Context passed to callbacks.

    Contains useful information about the current operation.
    """
    event: str
    model_name: str = ""
    device: str = ""

    # Inference context
    image: Any = None
    image_shape: tuple = None
    input_tensor: Any = None
    outputs: Dict[str, Any] = None
    results: Any = None

    # Timing
    latency_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    # Batch context
    batch_size: int = 0
    batch_index: int = 0

    # Stream context
    frame_idx: int = 0
    source_fps: float = 0.0

    # Custom data
    custom: Dict[str, Any] = field(default_factory=dict)


CallbackFn = Callable[[CallbackContext], Optional[Any]]


class CallbackManager:
    """
    Manages callback registration and execution.

    Allows registering multiple callbacks per event.
    """

    def __init__(self):
        self._callbacks: Dict[str, List[CallbackFn]] = {}
        self._enabled = True

    def register(
        self,
        event: str,
        callback: CallbackFn,
        priority: int = 0
    ) -> None:
        """
        Register a callback for an event.

        Args:
            event: Event name (use CallbackEvent enum)
            callback: Callback function
            priority: Priority (higher = called first)

        Examples:
            >>> def my_callback(ctx):
            ...     print(f"Inference took {ctx.latency_ms}ms")
            >>> manager.register("infer_end", my_callback)
        """
        if event not in self._callbacks:
            self._callbacks[event] = []

        self._callbacks[event].append((priority, callback))
        # Sort by priority (descending)
        self._callbacks[event].sort(key=lambda x: x[0], reverse=True)

        logger.debug(f"Registered callback for '{event}'")

    def unregister(
        self,
        event: str,
        callback: Optional[CallbackFn] = None
    ) -> int:
        """
        Unregister callbacks.

        Args:
            event: Event name
            callback: Specific callback to remove (None = remove all)

        Returns:
            Number of callbacks removed
        """
        if event not in self._callbacks:
            return 0

        if callback is None:
            count = len(self._callbacks[event])
            self._callbacks[event] = []
            return count

        original_len = len(self._callbacks[event])
        self._callbacks[event] = [
            (p, cb) for p, cb in self._callbacks[event]
            if cb != callback
        ]
        return original_len - len(self._callbacks[event])

    def trigger(
        self,
        event: str,
        context: CallbackContext
    ) -> Optional[Any]:
        """
        Trigger callbacks for an event.

        Args:
            event: Event name
            context: Callback context

        Returns:
            Modified data from callbacks (if any)
        """
        if not self._enabled:
            return None

        if event not in self._callbacks:
            return None

        result = None
        for priority, callback in self._callbacks[event]:
            try:
                ret = callback(context)
                if ret is not None:
                    result = ret
            except Exception as e:
                logger.error(f"Callback error for '{event}': {e}")

        return result

    def has_callbacks(self, event: str) -> bool:
        """Check if event has any registered callbacks."""
        return event in self._callbacks and len(self._callbacks[event]) > 0

    def list_callbacks(self, event: Optional[str] = None) -> Dict[str, int]:
        """
        List registered callbacks.

        Args:
            event: Specific event (None = all events)

        Returns:
            Dict mapping event names to callback counts
        """
        if event:
            return {event: len(self._callbacks.get(event, []))}
        return {e: len(cbs) for e, cbs in self._callbacks.items()}

    def enable(self) -> None:
        """Enable callback execution."""
        self._enabled = True

    def disable(self) -> None:
        """Disable callback execution (for performance)."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if callbacks are enabled."""
        return self._enabled

    def clear(self) -> None:
        """Remove all callbacks."""
        self._callbacks.clear()


# Decorator for creating callbacks
def callback(event: str, priority: int = 0):
    """
    Decorator to mark a function as a callback.

    Args:
        event: Event name
        priority: Callback priority

    Examples:
        >>> @callback("infer_end")
        ... def log_latency(ctx):
        ...     print(f"Latency: {ctx.latency_ms}ms")
    """
    def decorator(func: CallbackFn) -> CallbackFn:
        func._callback_event = event
        func._callback_priority = priority
        return func
    return decorator


# Built-in callbacks

class LatencyLogger:
    """Callback that logs inference latency."""

    def __init__(self, log_every: int = 1):
        """
        Args:
            log_every: Log every N inferences
        """
        self.log_every = log_every
        self._count = 0
        self._total_ms = 0.0

    def __call__(self, ctx: CallbackContext) -> None:
        self._count += 1
        self._total_ms += ctx.latency_ms

        if self._count % self.log_every == 0:
            avg = self._total_ms / self._count
            logger.info(
                f"Inference #{self._count}: {ctx.latency_ms:.2f}ms "
                f"(avg: {avg:.2f}ms)"
            )

    @property
    def average_latency(self) -> float:
        """Get average latency in milliseconds."""
        if self._count == 0:
            return 0.0
        return self._total_ms / self._count

    @property
    def count(self) -> int:
        """Get total inference count."""
        return self._count

    @property
    def total_latency(self) -> float:
        """Get total latency in milliseconds."""
        return self._total_ms

    def reset(self) -> None:
        """Reset the logger."""
        self._count = 0
        self._total_ms = 0.0


class FPSCounter:
    """Callback that tracks FPS using a sliding window."""

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of frames for moving average
        """
        self.window_size = window_size
        # Use deque with maxlen for O(1) append and automatic size limit
        self._latencies: deque = deque(maxlen=window_size)

    def __call__(self, ctx: CallbackContext) -> None:
        # deque with maxlen automatically removes oldest when full
        self._latencies.append(ctx.latency_ms)

    @property
    def fps(self) -> float:
        """Get current FPS."""
        if not self._latencies:
            return 0.0
        avg_ms = sum(self._latencies) / len(self._latencies)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Get average latency."""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def reset(self) -> None:
        """Reset the counter."""
        self._latencies.clear()


class DetectionFilter:
    """Callback that filters detections by criteria."""

    def __init__(
        self,
        min_confidence: float = 0.0,
        min_area: float = 0.0,
        max_area: float = float('inf'),
        classes: Optional[List[str]] = None,
    ):
        """
        Args:
            min_confidence: Minimum confidence threshold
            min_area: Minimum bounding box area
            max_area: Maximum bounding box area
            classes: Allowed class labels
        """
        self.min_confidence = min_confidence
        self.min_area = min_area
        self.max_area = max_area
        self.classes = classes

    def __call__(self, ctx: CallbackContext) -> None:
        if ctx.results is None or not hasattr(ctx.results, 'detections'):
            return

        filtered = []
        for det in ctx.results.detections:
            # Check confidence
            if det.confidence < self.min_confidence:
                continue

            # Check area
            area = det.bbox.area
            if area < self.min_area or area > self.max_area:
                continue

            # Check class
            if self.classes is not None and det.label not in self.classes:
                continue

            filtered.append(det)

        ctx.results.detections = filtered
