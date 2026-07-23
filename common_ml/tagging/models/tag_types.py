
from dataclasses import dataclass
from typing import Dict, List, Optional
# these used to be in this file and I don't want to break stuff
from common_ml.tagging.messages import Tag, FrameInfo

@dataclass(frozen=True, kw_only=True)
class FrameTag:
    tag: str
    vector: Optional[List[float]] = None
    box: Dict[str, float]
    additional_info: Optional[Dict] = None

    def to_tag(self, **video_fields) -> Tag:
        """Build the tag of the matching payload type.
        video_fields contains start_time, end_time, source_media, track, frame_info."""
        return Tag(tag=self.tag, vector=self.vector, additional_info=self.additional_info, **video_fields)
