import os
import pytest
from common_ml.tagging.model_types import VideoModel, FrameModel
from common_ml.tagging.messages import Tag, FrameTag
from common_ml.tagging.running import *


TEST_DATA = os.path.join(os.path.dirname(__file__), "test-data")

class FakeVideoModel(VideoModel):
    def tag_video(self, fpath):
        return [
            Tag(tag="action", start_time=0, end_time=1000, source_media=fpath, track="", frame_info=None),
            Tag(tag="dialog", start_time=1000, end_time=2000, source_media=fpath, track="", frame_info=None),
        ]


class FakeFrameModel(FrameModel):
    def __init__(self):
        self.call_count = 0

    def tag_frame(self, img):
        if self.call_count % 2 == 0:
            return [FrameTag(tag="a", box={"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}), FrameTag(tag="b", box={"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4})]
        elif (self.call_count // 2) % 2 == 0:
            return [FrameTag(tag="a", box={"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4})]
        else:
            return [FrameTag(tag="b", box={"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4})]


@pytest.fixture
def video_model():
    return FakeVideoModel()


@pytest.fixture
def frame_model():
    return FakeFrameModel()


@pytest.fixture
def batch_frame_model(frame_model):
    return batchify_frame_model(frame_model)

@pytest.fixture
def test_videos():
    return [os.path.join(TEST_DATA, "test1.mp4"), os.path.join(TEST_DATA, "test2.mp4")]


@pytest.fixture
def test_images():
    return [os.path.join(TEST_DATA, "test1.png"), os.path.join(TEST_DATA, "test2.png")]