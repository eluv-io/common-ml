from typing import List, Protocol, runtime_checkable

import numpy as np

from common_ml.tagging.tag_types import Tag, FrameTag

@runtime_checkable
class VideoModel(Protocol):
    def tag_video(self, fpath: str) -> List[Tag]: ...
    
@runtime_checkable
class FrameModel(Protocol):
    def tag_frame(self, img: np.ndarray) -> List[FrameTag]: ...  # img: (H, W, 3)

@runtime_checkable
class BatchFrameModel(Protocol):
    def tag_frames(self, imgs: np.ndarray) -> List[List[FrameTag]]: ...  # imgs: (N, H, W, 3)