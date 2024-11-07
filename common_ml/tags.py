#
# Type definitions
#

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

@dataclass
class VideoTag:
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

@dataclass
class FrameTag:
    text: str
    box: Tuple[float, float, float, float]
    confidence: Optional[float]=None

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