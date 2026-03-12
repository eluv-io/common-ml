from typing import List, Protocol

import numpy as np

from common_ml.tagging.messages import Tag, FrameTag

class VideoModel(Protocol):
    def tag(self, fpath: str) -> List[Tag]: ...

class FrameModel(Protocol):
    def tag(self, img: np.ndarray) -> List[FrameTag]: ...  # img: (H, W, 3)

class BatchFrameModel(Protocol):
    def tag(self, imgs: np.ndarray) -> List[List[FrameTag]]: ...  # imgs: (N, H, W, 3)