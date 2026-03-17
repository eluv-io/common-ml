
import os
from dacite import from_dict
import multiprocessing

from common_ml.tagging.running import *
from common_ml.tagging.messages import *
from common_ml.tagging.models.abstract import VideoModel, FrameModel, BatchFrameModel
from common_ml.tagging.models.tag_types import Tag
from common_ml.tagging.file_tagger_adapt import get_file_tagger_from_video_model, get_file_tagger_from_frame_model


def test_video_tag(video_model: VideoModel, test_videos: List[str]):
    file_tagger = get_file_tagger_from_video_model(video_model)

    all_tags = []
    for fname in test_videos:
        all_tags.extend(file_tagger.tag(fname))

    assert len(all_tags) == 4

    assert all_tags[0].tag == "action"
    assert all_tags[0].start_time == 0 and all_tags[0].end_time == 1000
    assert all_tags[0].source_media == test_videos[0]

    assert all_tags[1].tag == "dialog"

def test_frame_tag(frame_model: FrameModel, test_images: List[str]):
    file_tagger = get_file_tagger_from_frame_model(frame_model, fps=1.0, allow_single_frame=True)

    all_tags = []
    for fname in test_images:
        all_tags.extend(file_tagger.tag(fname))

    assert len(all_tags) == 3

    assert all_tags[0].tag == "a"
    assert all_tags[0].frame_info
    assert len(all_tags[0].frame_info.box) == 4
    assert all_tags[0].source_media == test_images[0]

    assert all_tags[1].tag == "b"
    assert all_tags[1].source_media == test_images[0]

    assert all_tags[2].tag == "a"
    assert all_tags[2].source_media == test_images[1]

def test_frame_tag_videos(frame_model: FrameModel, test_videos: List[str]):
    file_tagger = get_file_tagger_from_frame_model(frame_model, fps=1, allow_single_frame=True)

    all_tags = []
    for fname in test_videos:
        all_tags.extend(file_tagger.tag(fname))

    assert len(all_tags) == 122
    frame_tags = [t for t in all_tags if t.frame_info is not None]
    video_tags = [t for t in all_tags if t.frame_info is None]

    for tag in all_tags:
        assert tag.source_media in test_videos
        assert tag.tag in ["a", "b"]

    for tag in frame_tags:
        assert len(tag.frame_info.box) == 4
        assert tag.start_time == tag.end_time

    for tag in video_tags:
        assert tag.end_time > tag.start_time

def test_frame_tag_videos_single_false(frame_model: FrameModel, test_videos: List[str]):
    file_tagger = get_file_tagger_from_frame_model(frame_model, fps=1, allow_single_frame=False)

    all_tags = []
    for fname in test_videos:
        all_tags.extend(file_tagger.tag(fname))

    video_tags = [t for t in all_tags if t.frame_info is None]

    for tag in video_tags:
        # we shouldn't have single frame video tags.
        assert tag.end_time > tag.start_time + 1000