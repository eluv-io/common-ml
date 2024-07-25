#
# Type definitions
#

from dataclasses import dataclass
from typing import Optional

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