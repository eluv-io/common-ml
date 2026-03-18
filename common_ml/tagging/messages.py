from typing import List, Literal, Optional, Dict, Protocol
from dataclasses import dataclass

from common_ml.tagging.models.tag_types import Tag

class Message: ...

@dataclass
class Progress:
    source_media: str

@dataclass
class Error:
    message: str
    source_media: Optional[str] = None

@dataclass
class TagMessage(Message):
    type: Literal['tag']
    data: Tag

@dataclass
class ProgressMessage(Message):
    type: Literal['progress']
    data: Progress

@dataclass
class ErrorMessage(Message):
    type: Literal['error']
    data: Error