

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from common_ml.tagging.models.tag_types import BaseFrameTag

class FrameModel(ABC):
    @abstractmethod
    def tag_frame(self, img: np.ndarray) -> List[BaseFrameTag]:
        """
        Parameters
        ----------
        img : np.ndarray, shape (H, W, 3), dtype uint8
            RGB image.
        """


class BatchFrameModel(ABC):
    @abstractmethod
    def tag_frames(self, imgs: np.ndarray) -> List[List[BaseFrameTag]]:
        """
        Parameters
        ----------
        imgs : np.ndarray, shape (N, H, W, 3), dtype uint8
            Batch of RGB images.
        """

    @staticmethod
    def from_frame_model(model: 'FrameModel') -> 'BatchFrameModel':
        class NewModel(BatchFrameModel):
            def tag_frames(self, imgs: np.ndarray) -> List[List[BaseFrameTag]]:
                return [model.tag_frame(img) for img in imgs]
        return NewModel()
