from typing import List, Protocol

from common_ml.tagging.models.tag_types import Tag
from common_ml.tagging.messages import Message

class MessageProducer(Protocol):
    def produce_messages(self, files: List[str]) -> List[Message]:
        ...

class FileTagger(Protocol):
    def tag(self, file: str) -> List[Tag]:
        ...