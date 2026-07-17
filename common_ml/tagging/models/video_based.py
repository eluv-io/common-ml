
from abc import ABC, abstractmethod
from typing import List, Optional


class VideoVectorModel(ABC):
    """A model that embeds an entire video (or a time window of it) into a single vector.

    Unlike `BatchFrameModel`, which embeds frames independently, implementations
    are free to use a temporal-aware path (native video encoder, optical flow,
    3D-CNN, ...) so that motion/ordering information is preserved in the vector.

    Use with `AVModel.from_video_vector_model` to obtain a tagger that emits a
    pooled whole-video `VectorTag` (and, optionally, one `VectorTag` per segment).
    """

    @abstractmethod
    def embed_video(
        self,
        fpath: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> List[float]:
        """Embed a whole video, or the time window ``[start_ms, end_ms]`` of it, into one vector.

        Segmentation is driven by the caller (the AVModel factory): it passes a
        ``(start_ms, end_ms)`` window per segment so the model can sample frames
        densely within that window instead of sparsely across the whole video.
        This is what lets the approach scale to multi-hour videos without losing
        temporal detail.

        Times are in milliseconds, matching `VectorTag.start_time`/`end_time`.
        ``None`` means the media boundary (start of media / end of media).

        Parameters
        ----------
        fpath : str
            Path to a video file.
        start_ms : Optional[int]
            Window start in milliseconds (inclusive). ``None`` -> start of media.
        end_ms : Optional[int]
            Window end in milliseconds (exclusive). ``None`` -> end of media.
        normalize : Optional[bool]
            Whether to L2-normalize the returned vector. ``None`` -> use the
            implementation's own default (typically normalized (True)). The factory
            threads its own ``normalize`` here so each segment is normalized (or
            raw) before pooling.

        Returns
        -------
        List[float]
            A single embedding vector representing the requested window or whole video by default.
        """
