from typing import Optional, Dict, List, ClassVar, Hashable
from dataclasses import dataclass, field

class Message:
    message_type: ClassVar[str] = "message"

@dataclass(frozen=True)
class FrameInfo:
    frame_idx: int
    box: Dict[str, float]

@dataclass(frozen=True, kw_only=True)
class BaseTag(Message):
    start_time: int
    end_time: int
    source_media: str
    track: str = ""
    additional_info: Optional[Dict] = None
    frame_info: Optional[FrameInfo] = None

    def grouping_key(self) -> Optional[Hashable]:
        """Key used by AVModel._combine_adjacent to merge consecutive frame
        detections into a single time-ranged tag. Return None to opt out of
        combination entirely (each detection stands on its own)."""
        return None

@dataclass(frozen=True, kw_only=True)
class Tag(BaseTag):
    message_type: ClassVar[str] = "tag"
    tag: str

    def grouping_key(self) -> Optional[Hashable]:
        return self.tag

@dataclass(frozen=True, kw_only=True)
class VectorTag(BaseTag):
    message_type: ClassVar[str] = "vector_tag"
    vector: List[float] = field(default_factory=list)
    # inherits grouping_key() -> None: vectors are never run-length combined

@dataclass(frozen=True)
class Progress(Message):
    message_type: ClassVar[str] = "progress"
    source_media: str

@dataclass(frozen=True)
class ProgressRatio(Message):
    message_type: ClassVar[str] = "progress_ratio"
    progress: float

@dataclass(frozen=True)
class Error(Message):
    message_type: ClassVar[str] = "error"
    message: str
    source_media: Optional[str] = None
