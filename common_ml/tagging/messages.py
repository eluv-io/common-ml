from typing import Optional, Dict, List, ClassVar
from dataclasses import dataclass, field

class Message:
    message_type: ClassVar[str]

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

@dataclass(frozen=True, kw_only=True)
class Tag(BaseTag):
    message_type: ClassVar[str] = "tag"
    tag: str

@dataclass(frozen=True, kw_only=True)
class Vector(BaseTag):
    message_type: ClassVar[str] = "vector"
    vector: List[float] = field(default_factory=list)

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
