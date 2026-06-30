from collections.abc import Iterator
from abc import ABC, abstractmethod
from typing import Optional

from common_ml.tagging.messages import Message
from common_ml.tagging.models.tag_types import FrameInfo, FrameTag, Tag
from common_ml.tagging.models.frame_based import BatchFrameModel
from common_ml.video_processing import get_frames, get_fps

class TagProcessor(ABC):

    """ Whether this processor produces tags where start_time is in increasing order
        True: This processor produces tags in content order.  This means the run helper will output progress messages normally.
        False: This processor produces tags that may be out of order.  The run helper will not emit progress messages until the end.
        None: Unknown.  Processor will behave as "True" unless tags go backwards, then will hold all future progress messages as "False"
    
        Note this function must return a constant value, and will be called at the very start of processing.
    """
    def outputs_in_order(self) -> Optional[bool]:
        return None
    
    """ Called at the very start of processing, before any tags are produced.  Can be used to do any setup or validation that requires access to the content. """        
    def preflight(self, iq: str, auth: str):
        pass

    """ Generate (yield) tags for a given iq (with auth) in the specified time range.
        May also yield progress messages"""
    @abstractmethod
    def process(self, iq: str, auth: str, start_time: int, end_time: int) -> Iterator[Message]:
        pass

    """ Any final tags to generate."""
    def on_completion(self) -> Iterator[Message]:
        yield from ()
