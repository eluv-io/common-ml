from abc import ABC, abstractmethod
from typing import List, IO, Tuple, Dict

from tags import VideoTag, FrameTag

class VideoModel(ABC):
    @abstractmethod
    def tag(self, data: IO) -> List[VideoTag]:
        pass

class VideoFrameModel(ABC):
    @abstractmethod
    def tag_frame(self, video: IO) -> Tuple[List[VideoTag], Dict[int, List[FrameTag]]]:
        pass