from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

from tags import VideoTag, FrameTag

class VideoModel(ABC):
    @abstractmethod
    def tag(self, data: Any) -> List[VideoTag]:
        pass

class VideoFrameModel(ABC):
    @abstractmethod
    def tag(self, video: Any) -> Tuple[List[VideoTag], Dict[int, List[FrameTag]]]:
        pass