
from dataclasses import dataclass
from typing import Dict, Optional
# these used to be in this file and I don't want to break stuff
from common_ml.tagging.messages import Tag, FrameInfo

@dataclass(frozen=True)
class FrameTag:
    tag: str
    box: Dict[str, float]
    additional_info: Optional[Dict] = None
