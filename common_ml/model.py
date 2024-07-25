from abc import ABC, abstractmethod
from typing import List, IO, Tuple, Dict, Any

from tags import VideoTag, FrameTag

class VideoModel(ABC):
    @abstractmethod
    def tag(self, data: Any) -> List[VideoTag]:
        pass

class VideoFrameModel(ABC):
    @abstractmethod
    def tag_frame(self, video: Any) -> Tuple[List[VideoTag], Dict[int, List[FrameTag]]]:
        pass