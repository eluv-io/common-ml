
from dataclasses import dataclass, field
from typing import Dict, List, Optional
# these used to be in this file and I don't want to break stuff
from common_ml.tagging.messages import Tag, VectorTag, BaseTag, FrameInfo

@dataclass(frozen=True, kw_only=True)
class BaseFrameTag:
    box: Dict[str, float]
    additional_info: Optional[Dict] = None

    def to_tag(self, **video_fields) -> BaseTag:
        """Build the video-level tag of the matching payload type.

        video_fields carries the payload-agnostic fields (start_time, end_time,
        source_media, track, frame_info); the payload is injected by the subclass."""
        raise NotImplementedError

@dataclass(frozen=True, kw_only=True)
class FrameTag(BaseFrameTag):
    tag: str

    def to_tag(self, **video_fields) -> Tag:
        return Tag(tag=self.tag, additional_info=self.additional_info, **video_fields)

@dataclass(frozen=True, kw_only=True)
class VectorFrameTag(BaseFrameTag):
    vector: List[float] = field(default_factory=list)

    def to_tag(self, **video_fields) -> VectorTag:
        return VectorTag(vector=self.vector, additional_info=self.additional_info, **video_fields)
