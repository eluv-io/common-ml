import json
import os
from typing import List
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from common_ml.model import FrameModel, VideoModel, default_tag
from common_ml.tags import Tag, FrameInfo


class DumbFrameModel(FrameModel):
    def __init__(self, tags: List[Tag]):
        self._tags = tags

    def tag_frame(self, img: np.ndarray) -> List[Tag]:
        return self._tags


class DumbVideoModel(VideoModel):
    def __init__(self, tags: List[Tag]):
        self._tags = tags

    def tag(self, fpath: str) -> List[Tag]:
        return self._tags


FAKE_IMAGE = np.zeros((10, 10, 3), dtype=np.uint8)


def read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_empty_files_does_nothing(tmp_path):
    output = str(tmp_path / "out.jsonl")
    default_tag(DumbFrameModel([]), [], output)
    assert not os.path.exists(output)


def test_image_file_writes_tags(tmp_path, make_tag):
    output = str(tmp_path / "out.jsonl")
    tags = [make_tag("dog", frame_idx=0), make_tag("cat", frame_idx=0)]
    model = DumbFrameModel(tags)

    with patch("common_ml.model.get_file_type", return_value="image"), \
         patch("os.path.exists", return_value=True), \
         patch("cv2.imread", return_value=FAKE_IMAGE):
        default_tag(model, ["fake.jpg"], output)

    rows = read_jsonl(output)
    assert len(rows) == 2
    assert rows[0]["text"] == "dog"
    assert rows[1]["text"] == "cat"


def test_video_file_writes_tags(tmp_path, make_tag):
    output = str(tmp_path / "out.jsonl")
    tags = [make_tag("car", frame_idx=0)]
    model = DumbVideoModel(tags)

    with patch("common_ml.model.get_file_type", return_value="video"):
        default_tag(model, ["fake.mp4"], output)

    rows = read_jsonl(output)
    assert len(rows) == 1
    assert rows[0]["text"] == "car"


def test_multiple_image_files_all_written(tmp_path, make_tag):
    output = str(tmp_path / "out.jsonl")
    model = DumbFrameModel([make_tag("dog", frame_idx=0)])

    with patch("common_ml.model.get_file_type", return_value="image"), \
         patch("os.path.exists", return_value=True), \
         patch("cv2.imread", return_value=FAKE_IMAGE):
        default_tag(model, ["a.jpg", "b.jpg", "c.jpg"], output)

    rows = read_jsonl(output)
    assert len(rows) == 3


def test_multiple_video_files_all_written(tmp_path, make_tag):
    output = str(tmp_path / "out.jsonl")
    model = DumbVideoModel([make_tag("cat", frame_idx=0)])

    with patch("common_ml.model.get_file_type", return_value="video"):
        default_tag(model, ["a.mp4", "b.mp4"], output)

    rows = read_jsonl(output)
    assert len(rows) == 2


def test_unknown_file_type_raises(tmp_path):
    output = str(tmp_path / "out.jsonl")
    model = DumbFrameModel([])

    with patch("common_ml.model.get_file_type", return_value="unknown"), \
         pytest.raises(AssertionError):
        default_tag(model, ["fake.xyz"], output)


def test_missing_image_file_raises(tmp_path, make_tag):
    output = str(tmp_path / "out.jsonl")
    model = DumbFrameModel([make_tag("dog", frame_idx=0)])

    with patch("common_ml.model.get_file_type", return_value="image"), \
         patch("os.path.exists", return_value=False), \
         pytest.raises(FileNotFoundError):
        default_tag(model, ["missing.jpg"], output)
