
from typing import List

from common_ml.tagging.messages import *
from common_ml.tagging.abstract import FileTagger
from common_ml.tagging.file_tagger_adapt import get_file_tagger_from_frame_model
from common_ml.tagging.models.abstract import FrameModel
from common_ml.tagging.producer_adapt import *


def test_message_producer(frame_model: FrameModel, test_videos: List[str], test_images: List[str]):
    producer = get_message_producer_from_model(frame_model)
    messages = producer.produce_messages(test_videos)

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
    messages = producer.produce_messages(test_images)
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
    file_tagger = get_file_tagger_from_frame_model(frame_model, fps=1, allow_single_frame=True)
    class ErrorTagger(FileTagger):
        def tag(self, file: str) -> List[Tag]:
            if file == test_videos[0]:
                raise RuntimeError("i'm panicking")
            else:
                return file_tagger.tag(file)
            
    new_tagger = ErrorTagger()

    # test with continue on error
    producer = get_message_producer_from_file_tagger(new_tagger, continue_on_error=True)
    messages = producer.produce_messages(test_videos)

    status_messages = [msg for msg in messages if isinstance(msg, ProgressMessage)]
    assert len(status_messages) == 1
    assert status_messages[0].data.source_media == test_videos[1]

    tag_messages = [msg for msg in messages if isinstance(msg, TagMessage)]
    assert len(tag_messages) > 0
    for msg in tag_messages:
        # make sure we don't get anything from the errored file
        assert msg.data.source_media == test_videos[1]

    error_messages = [msg for msg in messages if isinstance(msg, ErrorMessage)]
    assert len(error_messages) == 1
    assert error_messages[0].type == "error"
    assert error_messages[0].data.source_media == test_videos[0]

    # test without continue on error
    producer = get_message_producer_from_file_tagger(new_tagger, continue_on_error=False)
    messages = producer.produce_messages(test_videos)

    status_messages = [msg for msg in messages if isinstance(msg, ProgressMessage)]
    tag_messages = [msg for msg in messages if isinstance(msg, TagMessage)]
    error_messages = [msg for msg in messages if isinstance(msg, ErrorMessage)]

    assert len(status_messages) == len(tag_messages) == 0

    assert len(error_messages) == 1