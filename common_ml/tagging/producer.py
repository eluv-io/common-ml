    
from abc import ABC, abstractmethod

from loguru import logger
from typing import Union

from common_ml.tagging.messages import *
from common_ml.tagging.models.abstract import *
from common_ml.tagging.file_tagger import FileTagger

class TagMessageProducer(ABC):
    @abstractmethod
    def produce(self, files: List[str]) -> List[Message]:
        pass

    @staticmethod
    def from_file_tagger(file_tagger: FileTagger, continue_on_error: bool = False) -> 'TagMessageProducer':
        class NewTagMessageProducer(TagMessageProducer):
            def produce(self, files: List[str]) -> List[Message]:
                res = []

                for fname in files:
                    try:
                        tags = file_tagger.tag(fname)
                    except Exception as e:
                        logger.opt(exception=e).error(f"Error processing file {fname}")
                        res.append(ErrorMessage(type='error', data=Error(message=str(e), source_media=fname)))
                        if not continue_on_error:
                            return res
                        else:
                            continue

                    for tag in tags:
                        res.append(TagMessage(type="tag", data=tag))

                    # finished fname, add progress
                    res.append(ProgressMessage(type='progress', data=Progress(source_media=fname)))

                return res

        return NewTagMessageProducer()
    
    @staticmethod
    def from_model(
        model: Union[AVModel, FrameModel, BatchFrameModel], 
        fps: float=1.0, 
        allow_single_frame: bool=False,
        continue_on_error: bool=False
    ) -> 'TagMessageProducer':
        if isinstance(model, AVModel):
            file_tagger = FileTagger.from_video_model(model)
        elif isinstance(model, (FrameModel, BatchFrameModel)):
            file_tagger = FileTagger.from_frame_model(model, fps, allow_single_frame)
        else:
            raise ValueError("Model must be either AVModel, FrameModel, or BatchFrameModel")

        return TagMessageProducer.from_file_tagger(file_tagger, continue_on_error)