from typing import Literal, Optional, Dict
from dataclasses import dataclass

from common_ml.tagging.tag_types import Tag

class Message: ...

class TagMessage(Message):
    type: Literal['tag']
    data: Tag

class ProgressMessage(Message):
    type: Literal['progress']
    data: Dict[str, float]

class ErrorMessage(Message):
    type: Literal['error']
    data: Dict[str, str]

@dataclass
class Progress:
    source_media: str

@dataclass
class Error:
    message: str