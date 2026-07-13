    
from abc import ABC, abstractmethod

from typing import Union, Iterator

from common_ml.tagging.messages import *
from common_ml.tagging.models.frame_based import *
from common_ml.tagging.models.av import AVModel
from common_ml.tagging.models.processor import TagProcessor
from common_ml.tagging.file_tagger import FileTagger
from common_ml.tagging.processor_helper import TagProcessorAdapterLogic

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
        class FileTaggerToTagMessageProducerAdapter(TagMessageProducer):
            def produce(self, files: List[str]) -> Iterator[Message]:
                for fname in files:
                    tags = file_tagger.tag(fname)

                    for tag in tags:
                        yield tag

                    yield Progress(source_media=fname)

        return FileTaggerToTagMessageProducerAdapter()

    @staticmethod
    def from_tag_processor(tag_processor: TagProcessor) -> "TagMessageProducer":
        class TagProcessorToTagMessageProducerAdapter(TagMessageProducer):
            ## this is written like this to avoid circular dependency
            ## (and avoiding all the tag processor -> producer code being here in this class, since it's a lot)
            def __init__(self, tag_processor: TagProcessor):
                self.tag_processor_adapter_logic = TagProcessorAdapterLogic(tag_processor)

            def produce(self, files: List[str]) -> Iterator[Message]:
                return self.tag_processor_adapter_logic.produce(files)

            def on_completion(self) -> Iterator[Message]:
                return self.tag_processor_adapter_logic.on_completion()

        return TagProcessorToTagMessageProducerAdapter(tag_processor)
    
    @staticmethod
    def from_model(
        model: Union[AVModel, FrameModel, BatchFrameModel, TagProcessor],
        fps: float=1.0, 
        allow_single_frame: bool=True
    ) -> 'TagMessageProducer':
        if isinstance(model, AVModel):
            return TagMessageProducer.from_file_tagger(FileTagger.from_video_model(model))
        elif isinstance(model, (FrameModel, BatchFrameModel)):
            return TagMessageProducer.from_file_tagger(FileTagger.from_frame_model(model, fps, allow_single_frame))
        elif isinstance(model, TagProcessor):
            return TagMessageProducer.from_tag_processor(model)
        else:
            raise ValueError("Model must be either AVModel, FrameModel, BatchFrameModel, or TagProcessor")
