    
from abc import ABC, abstractmethod

from typing import Union, Iterator

from common_ml.tagging.messages import *
from common_ml.tagging.models.frame_based import *
from common_ml.tagging.models.av import AVModel
from common_ml.tagging.file_tagger import FileTagger

class TagMessageProducer(ABC):
    @abstractmethod
    def produce(self, files: List[str]) -> Iterator[Message]:
        pass

    def on_completion(self) -> Iterator[Message]:
        """
        If specified this will run any finalization logic to be run when all files have been received
        """
        yield from ()
    
    @staticmethod
    def from_file_tagger(file_tagger: FileTagger) -> "TagMessageProducer":
        class NewTagMessageProducer(TagMessageProducer):
            def produce(self, files: List[str]) -> Iterator[Message]:
                for fname in files:
                    tags = file_tagger.tag(fname)

                    for tag in tags:
                        yield tag

                    yield Progress(source_media=fname)

        return NewTagMessageProducer()
    
    @staticmethod
    def from_model(
        model: Union[AVModel, FrameModel, BatchFrameModel], 
        fps: float=1.0, 
        allow_single_frame: bool=True
    ) -> 'TagMessageProducer':
        if isinstance(model, AVModel):
            file_tagger = FileTagger.from_video_model(model)
        elif isinstance(model, (FrameModel, BatchFrameModel)):
            file_tagger = FileTagger.from_frame_model(model, fps, allow_single_frame)
        else:
            raise ValueError("Model must be either AVModel, FrameModel, or BatchFrameModel")

        return TagMessageProducer.from_file_tagger(file_tagger)