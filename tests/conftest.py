import os
import shutil
import tempfile
import pytest
from common_ml.tagging.models.av import AVModel
from common_ml.tagging.models.frame_based import *
from common_ml.tagging.models.tag_types import *
from common_ml.tagging.run_helpers import *


TEST_DATA = os.path.join(os.path.dirname(__file__), "test-data")

class FakeAVModel(AVModel):
    def tag(self, fpath):
        return [
            Tag(tag="action", start_time=0, end_time=1000, source_media=fpath, track="", frame_info=None),
            Tag(tag="dialog", start_time=1000, end_time=2000, source_media=fpath, track="", frame_info=None),
        ]


class FakeFrameModel(FrameModel):
    def __init__(self):
        self.call_count = 0

    def tag_frame(self, img):
        if self.call_count % 2 == 0:
            res = [FrameTag(tag="a", box={"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}), FrameTag(tag="b", box={"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4})]
        elif (self.call_count // 2) % 2 == 0:
            res = [FrameTag(tag="a", box={"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4})]
        else:
            res = [FrameTag(tag="b", box={"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4})]
        self.call_count += 1
        # add some random additional_info
        out = []
        for r in res:
            out.append(
                FrameTag(
                    tag=r.tag,
                    box=r.box,
                    additional_info={"hello": "world"}
                )
            )
        return out


@pytest.fixture
def video_model():
    return FakeAVModel()


@pytest.fixture
def frame_model():
    return FakeFrameModel()


@pytest.fixture
def batch_frame_model(frame_model):
    return BatchFrameModel.from_frame_model(frame_model)

@pytest.fixture
def test_videos():
    return [os.path.join(TEST_DATA, "1.mp4"), os.path.join(TEST_DATA, "2.mp4")]


@pytest.fixture
def test_images():
    return [os.path.join(TEST_DATA, "1.jpg"), os.path.join(TEST_DATA, "2.jpg")]

@pytest.fixture
def test_folder():
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)