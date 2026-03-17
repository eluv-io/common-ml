

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from common_ml.tagging.models.tag_types import FrameTag
    
class FrameModel(ABC):
    @abstractmethod
    def tag_frame(self, img: np.ndarray) -> List[FrameTag]: pass  # img: (H, W, 3)


class BatchFrameModel(ABC):
    @abstractmethod
    def tag_frames(self, imgs: np.ndarray) -> List[List[FrameTag]]: pass  # imgs: (N, H, W, 3)

    @staticmethod
    def from_frame_model(model: 'FrameModel') -> 'BatchFrameModel':
        class NewModel(BatchFrameModel):
            def tag_frames(self, imgs: np.ndarray) -> List[List[FrameTag]]:
                return [model.tag_frame(img) for img in imgs]
        return NewModel()
