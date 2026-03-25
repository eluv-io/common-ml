
from typing import List

import pytest

from common_ml.tagging.messages import *
from common_ml.tagging.file_tagger import FileTagger
from common_ml.tagging.models.frame_based import FrameModel
from common_ml.tagging.producer import *


def test_message_producer(frame_model: FrameModel, test_videos: List[str], test_images: List[str]):
    producer = TagMessageProducer.from_model(frame_model, fps=1.0, allow_single_frame=True)
    messages = list(producer.produce(test_videos))

    status_messages = [msg for msg in messages if isinstance(msg, ProgressMessage)]
    assert len(status_messages) == 2
    assert status_messages[0].type == "progress"
    assert status_messages[0].data.source_media == test_videos[0]
    assert status_messages[1].type == "progress"
    assert status_messages[1].data.source_media == test_videos[1]

    tag_messages = [msg for msg in messages if isinstance(msg, TagMessage)]
    assert len(tag_messages) > 0
    for msg in tag_messages:
        assert msg.type == "tag"
        if msg.data.frame_info:
            assert msg.data.start_time == msg.data.end_time
        else:
            assert msg.data.end_time > msg.data.start_time
        assert msg.data.source_media in test_videos
        assert msg.data.tag in ["a", "b"]

    error_messages = [msg for msg in messages if isinstance(msg, ErrorMessage)]
    assert len(error_messages) == 0

    # test on images
    messages = list(producer.produce(test_images))
    status_messages = [msg for msg in messages if isinstance(msg, ProgressMessage)]
    tag_messages = [msg for msg in messages if isinstance(msg, TagMessage)]

    assert len(status_messages) == 2
    assert len(tag_messages) > 0

    for msg in tag_messages:
        assert msg.type == "tag"
        assert msg.data.start_time == msg.data.end_time
        assert msg.data.frame_info
        assert msg.data.source_media in test_images

def test_producer_error(frame_model: FrameModel, test_videos: List[str]):
    file_tagger = FileTagger.from_frame_model(frame_model, fps=1, allow_single_frame=True)
    class ErrorTagger(FileTagger):
        def tag(self, file: str) -> List[Tag]:
            if file == test_videos[1]:
                raise RuntimeError("i'm panicking")
            else:
                return file_tagger.tag(file)
            
    new_tagger = ErrorTagger()

    producer = TagMessageProducer.from_file_tagger(new_tagger)
    with pytest.raises(Exception):
        # it should error the whole thing
        list(producer.produce(test_videos))