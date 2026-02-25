from typing import List
from unittest.mock import patch
import numpy as np

from common_ml.model import FrameModel, VideoModel, get_video_model_from_frame_model
from common_ml.tags import Tag, FrameInfo


class DumbFrameModel(FrameModel):
    """Returns a fixed list of tags for every frame, regardless of content."""

    def __init__(self, tags: List[Tag]):
        self._tags = tags

    def tag_frame(self, img: np.ndarray) -> List[Tag]:
        return self._tags

# Fake frames returned by get_frames: two key frames at positions 0 and 1
FAKE_FRAMES = [np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8)]
FAKE_FPOS = [0, 1]


def test_returns_video_model_instance():
    frame_model = DumbFrameModel([])
    video_model = get_video_model_from_frame_model(frame_model, fps=1.0, allow_single_frame=True)
    assert isinstance(video_model, VideoModel)
    assert isinstance(video_model, FrameModel)  # should also implement FrameModel interface


def test_tag_frame_delegates_to_frame_model(make_tag):
    tags = [make_tag("dog"), make_tag("cat")]
    frame_model = DumbFrameModel(tags)
    video_model = get_video_model_from_frame_model(frame_model, fps=1.0, allow_single_frame=True)
    assert isinstance(video_model, FrameModel)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    result = video_model.tag_frame(img)
    assert result == tags


def test_tag_returns_frame_tags_and_combined_tags(make_tag):
    """tag() should return both per-frame tags (with frame_info) and combined video tags (without)."""
    frame_model = DumbFrameModel([make_tag("dog")])

    with patch("common_ml.model.get_frames", return_value=(FAKE_FRAMES, FAKE_FPOS, None)):
        video_model = get_video_model_from_frame_model(frame_model, fps=1.0, allow_single_frame=True)
        result = video_model.tag("dummy.mp4")

    frame_level = [t for t in result if t.frame_info is not None]
    combined = [t for t in result if t.frame_info is None]

    assert len(frame_level) == 2
    assert all(t.text == "dog" for t in frame_level)

    assert len(combined) == 1
    assert combined[0].text == "dog"


def test_tag_frame_info_populated_correctly(make_tag):
    """Frame-level tags should have frame_idx matching the key frame position."""
    frame_model = DumbFrameModel([make_tag("cat")])

    with patch("common_ml.model.get_frames", return_value=(FAKE_FRAMES, FAKE_FPOS, None)):
        video_model = get_video_model_from_frame_model(frame_model, fps=1.0, allow_single_frame=True)
        result = video_model.tag("dummy.mp4")

    frame_level = [t for t in result if t.frame_info is not None]
    assert {t.frame_info.frame_idx for t in frame_level} == {0, 1}


def test_tag_no_single_frame_suppresses_non_adjacent(make_tag):
    """With allow_single_frame=False, tags that span only one frame are excluded from combined output."""
    frame_model = DumbFrameModel([make_tag("dog")])

    # Only one frame at position 0 — should produce no combined tag
    single_frame = [np.zeros((10, 10, 3), dtype=np.uint8)]
    single_fpos = [0]

    with patch("common_ml.model.get_frames", return_value=(single_frame, single_fpos, None)):
        video_model = get_video_model_from_frame_model(frame_model, fps=1.0, allow_single_frame=False)
        result = video_model.tag("dummy.mp4")

    combined = [t for t in result if t.frame_info is None]
    assert combined == []


def test_tag_empty_frame_model(make_tag):
    """A frame model that returns no tags should result in an empty tag list."""
    frame_model = DumbFrameModel([])

    with patch("common_ml.model.get_frames", return_value=(FAKE_FRAMES, FAKE_FPOS, None)):
        video_model = get_video_model_from_frame_model(frame_model, fps=1.0, allow_single_frame=True)
        result = video_model.tag("dummy.mp4")

    assert result == []
