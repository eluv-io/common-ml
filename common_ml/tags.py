#
# Type definitions
#

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

from common_ml.types import Data

@dataclass
class VideoTag(Data):
    # VideoTag represents a single tag in a video, possibly containing a text label
    #
    # Has attributes
    # - start_time: int (required) (in milliseconds)
    # - end_time: int (required) (in milliseconds)
    # - text: str (optional) (the text of the tag, sometimes this is not relevant (i.e shot detection))
    # - confidence: float (optional) (the confidence of the tag)
    start_time: int
    end_time: int
    text: str
    confidence: Optional[float]=None

    @staticmethod
    def from_dict(data: dict) -> 'VideoTag':
        return VideoTag(start_time=data['start_time'], end_time=data['end_time'], text=data['text'], confidence=data.get('confidence'))

@dataclass
class _Box:
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass
class FrameTag(Data):
    text: str
    box: _Box
    confidence: Optional[float]=None

    @staticmethod
    def from_dict(data: dict) -> 'FrameTag':
        return FrameTag(text=data['text'], box=_Box(**data['box']), confidence=data.get('confidence'))

@dataclass
class AggTag:
    start_time: int
    end_time: int
    tags: Dict[str, List[VideoTag]]

    # Given a feature, merges all the tags into a single tag by joining the text 
    # Used for STT only right now
    def coalesce(self, feature: str) -> None:
        if feature not in self.tags:
            return
        tags = self.tags[feature]
        if len(tags) == 0:
            return
        text = " ".join([tag.text for tag in tags])
        self.tags[feature] = [VideoTag(start_time=tags[0].start_time, end_time=tags[-1].end_time, text=text, confidence=tags[0].confidence)]