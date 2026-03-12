from typing import List, Protocol

import numpy as np

from common_ml.tagging.messages import Tag, FrameTag

class VideoModel(Protocol):
    def tag(self, fpath: str) -> List[Tag]: ...

class FrameModel(Protocol):
    def tag(self, img: np.ndarray) -> List[FrameTag]: ...