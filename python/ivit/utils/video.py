"""
Video stream utilities.
"""

from typing import Union, Optional, Tuple, Iterator
import numpy as np


class VideoStream:
    """
    Video stream reader.

    Supports video files, cameras, and RTSP streams.

    Examples:
        >>> stream = VideoStream("video.mp4")
        >>> for frame in stream:
        ...     results = model.predict(frame)
        >>> stream.release()

        >>> # Or with context manager
        >>> with VideoStream(0) as stream:  # Camera
        ...     for frame in stream:
        ...         process(frame)
    """

    def __init__(
        self,
        source: Union[str, int],
        fps: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None,
    ):
        """
        Open video stream.

        Args:
            source: Video file path, camera ID, or RTSP URL
            fps: Target FPS (None = original)
            resolution: Target resolution (width, height)
        """
        import cv2

        self._source = source
        self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")

        # Set resolution if specified
        if resolution:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # Set FPS if specified (for cameras)
        if fps and isinstance(source, int):
            self._cap.set(cv2.CAP_PROP_FPS, fps)

        self._target_fps = fps
        self._frame_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    def read(self) -> Optional[np.ndarray]:
        """
        Read next frame.

        Returns:
            Frame as numpy array (BGR) or None if end of stream
        """
        ret, frame = self._cap.read()
        if not ret:
            return None

        self._frame_count += 1
        return frame

    def release(self) -> None:
        """Release video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        """Get stream FPS."""
        import cv2
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def width(self) -> int:
        """Get frame width."""
        import cv2
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Get frame height."""
        import cv2
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        """Get total frame count (for video files)."""
        import cv2
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def current_frame(self) -> int:
        """Get current frame number."""
        return self._frame_count

    def seek(self, frame_number: int) -> bool:
        """
        Seek to specific frame (video files only).

        Args:
            frame_number: Target frame number

        Returns:
            True if successful
        """
        import cv2
        return self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    @property
    def is_opened(self) -> bool:
        """Check if stream is opened."""
        return self._cap is not None and self._cap.isOpened()


class VideoWriter:
    """
    Video writer.

    Examples:
        >>> writer = VideoWriter("output.mp4", fps=30, resolution=(1920, 1080))
        >>> for frame in frames:
        ...     writer.write(frame)
        >>> writer.release()
    """

    def __init__(
        self,
        path: str,
        fps: float,
        resolution: Tuple[int, int],
        codec: str = "mp4v",
    ):
        """
        Create video writer.

        Args:
            path: Output file path
            fps: Frames per second
            resolution: (width, height)
            codec: FourCC codec
        """
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(path, fourcc, fps, resolution)

        if not self._writer.isOpened():
            raise ValueError(f"Failed to create video writer: {path}")

        self._path = path
        self._fps = fps
        self._resolution = resolution
        self._frame_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def write(self, frame: np.ndarray) -> None:
        """
        Write frame to video.

        Args:
            frame: Frame as numpy array (BGR)
        """
        import cv2

        # Resize if needed
        h, w = frame.shape[:2]
        if (w, h) != self._resolution:
            frame = cv2.resize(frame, self._resolution)

        self._writer.write(frame)
        self._frame_count += 1

    def release(self) -> None:
        """Release video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    @property
    def frame_count(self) -> int:
        """Get number of frames written."""
        return self._frame_count

    @property
    def is_opened(self) -> bool:
        """Check if writer is opened."""
        return self._writer is not None and self._writer.isOpened()
