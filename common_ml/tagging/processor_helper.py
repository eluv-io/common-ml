

from dataclasses import dataclass
import json
import os
from typing import Iterator, List
from loguru import logger

from common_ml.tagging.messages import Message, Progress, Tag
from common_ml.tagging.models.processor import TagProcessor

@dataclass
class InputRangeInfo:
    input: str
    start_time: float
    end_time: float

## logic written in a separate class to avoid circular dependency between producer and processor
class TagProcessorAdapterLogic:

    def __init__(self, tag_processor: TagProcessor):
        self.tag_processor = tag_processor
        self.in_order = tag_processor.outputs_in_order()
        self.iq = os.environ.get("ELV_CONTENT", None)
        self.auth = os.environ.get("ELV_TOKEN", None)
        self.last_time = 0
        self.last_source_media = None
        self.all_rangeinfos = []
        self.progress_bolus = []
        self.preflight_passed = False
        
    def produce(self, files: List[str]) -> Iterator[Message]:

        if self.iq is None or self.auth is None:
            raise ValueError("ELV_CONTENT and ELV_TOKEN environment variables must be set to use TagProcessorAdapter")

        if not self.preflight_passed:
            self.tag_processor.preflight(self.iq, self.auth)
            self.preflight_passed = True
            
        rangeinfos = []
        for fname in files:
            with open(fname, 'r') as f:
                data = json.load(f)
               
                rangeinfo = InputRangeInfo(
                    input=fname,
                    start_time=data['start_time'],
                    end_time=data['end_time'],
                )
                rangeinfos.append(rangeinfo)

        if not rangeinfos: return

        rangeinfos = sorted(rangeinfos, key=lambda info: info.start_time)
        self.all_rangeinfos = sorted(self.all_rangeinfos + rangeinfos, key=lambda info: info.start_time)
        
        consolidated = []
        current_start = rangeinfos[0].start_time
        current_end = rangeinfos[0].end_time

        for info in rangeinfos[1:]:
            if info.start_time <= current_end:
                current_end = max(current_end, info.end_time)
            elif info.start_time == current_end:
                current_end = info.end_time
            else:
                consolidated.append([current_start, current_end])
                current_start = info.start_time
                current_end = info.end_time

        consolidated.append([current_start, current_end])

        next_progress_rangeinfo = 0
        for start, end in consolidated:
            logger.info(f"Processing range {start} to {end}")
            for message in self.tag_processor.process(self.iq, self.auth, start, end):
                source_media_info = None
                if isinstance(message, Tag):
                    source_media_info = find_input_range(rangeinfos, message.start_time)
                    if source_media_info is None:
                        source_media_info = find_input_range(self.all_rangeinfos, message.start_time)
                        
                    if source_media_info is None:
                        ## since there's no source media that satisfies this tag, it's okay if its "out of order"
                        logger.warning(f"Tag produced with start_time {message.start_time} that does not fall within any input range, crediting to first json")
                        source_media_info = self.all_rangeinfos[0]
                    elif self.in_order == False:
                        ## out of order okay because processor says it's out of order 
                        pass
                    elif self.in_order is None and message.start_time < self.last_time:
                        ## in-between case where processor doesn't specify
                        logger.warning("TagProcessor was unspecified whether it was in order or not.  Out of order tags detected.")
                        logger.warning("Check code to verify out-of-order tags are expected; if so consider having outputs_in_order() return False")
                        self.in_order = False
                    elif self.in_order == True and message.start_time < self.last_time:
                        raise Exception("TagProcessor declared itself in order but produced out of order tags")
                    
                    yield Tag(
                        start_time=message.start_time,
                        end_time=message.end_time,
                        tag=message.tag,
                        source_media=source_media_info.input,
                        track=message.track,
                        additional_info=message.additional_info,
                        frame_info=message.frame_info,
                    )
                                                            
                    if self.in_order:
                        while next_progress_rangeinfo < len(rangeinfos) and rangeinfos[next_progress_rangeinfo].input != source_media_info.input:
                            yield Progress(source_media=rangeinfos[next_progress_rangeinfo].input)
                            next_progress_rangeinfo += 1

                    self.last_time = message.start_time
                else:
                    yield message

        if self.in_order:
            ## finish up the progress for the rest of this block
            ## (this handles sparse processors)
            while next_progress_rangeinfo < len(rangeinfos):
                yield Progress(source_media=rangeinfos[next_progress_rangeinfo].input)
                next_progress_rangeinfo += 1
        else:
            ## out of order, keep track of these so we can dump them at the end
            self.progress_bolus += rangeinfos[next_progress_rangeinfo:]

    def on_completion(self) -> Iterator[Message]:
        for message in self.tag_processor.on_completion():
            source_media_info = None
            if isinstance(message, Tag):
                source_media_info = find_input_range(self.all_rangeinfos, message.start_time)                    
                if source_media_info is None:
                    logger.warning(f"Tag produced with start_time {message.start_time} that does not fall within any input range, crediting to first json")
                    source_media_info = self.all_rangeinfos[0]
                yield Tag(
                    start_time=message.start_time,
                    end_time=message.end_time,
                    tag=message.tag,
                    source_media=source_media_info.input,
                    track=message.track,
                    additional_info=message.additional_info,
                    frame_info=message.frame_info,
                )
            else:
                yield message

        ## progress message for everything now for out-of-order processors
        for info in self.progress_bolus:
            yield Progress(source_media=info.input)

def find_input_range(infos: List[InputRangeInfo], timestamp: float) -> Optional[InputRangeInfo]:
    """Return the most specific (latest-starting) range that contains timestamp."""
    best: Optional[InputRangeInfo] = None
    for info in infos:
        if info.start_time <= timestamp <= info.end_time:
            if best is None or info.start_time > best.start_time:
                best = info
    return best



