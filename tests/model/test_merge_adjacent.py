from typing import List, Dict

from common_ml.model import _combine_adjacent
from common_ml.tags import Tag


def make_frame_tags_dict(tags: List[Tag], f_intv: int):
    max_frame_idx = max(t.frame_info.frame_idx for t in tags if t.frame_info)
    f_idx_to_tags = {i: [] for i in range(0, max_frame_idx + 1, f_intv)}
    for tag in tags:
        assert tag.frame_info
        f_idx = tag.frame_info.frame_idx
        f_idx_to_tags[f_idx].append(tag)

    return f_idx_to_tags

def test_combine_adjacent_empty():
    assert _combine_adjacent({}, allow_single_frame=True, fps=1.0) == []

def test_combine_adjacent_single_frame_allowed(make_tag):
    tags = [make_tag("dog", 10), make_tag("cat", 10)]
    result = _combine_adjacent(make_frame_tags_dict(tags, 1), allow_single_frame=True, fps=1.0)
    assert len(result) == 2
    assert result[0].text == "dog"
    assert result[0].start_time == 10 * 1000
    assert result[0].end_time == 11 * 1000
    assert result[1].text == "cat"
    assert all(t.frame_info is None for t in result)


def test_combine_adjacent_single_frame_disallowed(make_tag):
    tags = [make_tag("dog", 10), make_tag("cat", 10)]
    result = _combine_adjacent(make_frame_tags_dict(tags, 1), allow_single_frame=False, fps=1.0)
    assert result == []

def test_combine_range(make_tag):
    tags = [make_tag("dog", 10), make_tag("dog", 11)]
    result = _combine_adjacent(make_frame_tags_dict(tags, 1), allow_single_frame=True, fps=1.0)
    assert len(result) == 1
    assert result[0].text == "dog"
    assert result[0].start_time == 10 * 1000
    assert result[0].end_time == 12 * 1000


def test_combine_adjacent_gap_produces_two_intervals(make_tag):
    # id_map: 0→1, 1→2, 2→3, 3→3; "dog" at [0,1,3]: id_map[1]=2≠3 → split
    tags = [make_tag("dog", 0), make_tag("dog", 1), make_tag("cat", 2), make_tag("dog", 3), make_tag("dog", 4)]
    result = _combine_adjacent(make_frame_tags_dict(tags, 1), allow_single_frame=False, fps=10.0)
    dog_tags = [t for t in result if t.text == "dog"]
    assert len(dog_tags) == 2

def test_combine_adjacent_preserves_source_media_and_track(make_tag):
    tags = [
        make_tag("dog", 0, source_media="myvideo.mp4", track="objects"),
        make_tag("dog", 1, source_media="myvideo.mp4", track="objects"),
    ]
    result = _combine_adjacent(make_frame_tags_dict(tags, 1), allow_single_frame=False, fps=10.0)
    assert len(result) == 1
    assert result[0].source_media == "myvideo.mp4"
    assert result[0].track == "objects"


def test_combine_far_apart(make_tag):
    # fps=10 → f_intv=0.1s; "dog" at frames 0 and 10
    # start_time = round(0/10 * 1000) = 0
    # end_time   = round((10/10 + 0.1) * 1000) = round(1.1 * 1000) = 1100
    tags = [make_tag("dog", 0), make_tag("dog", 10)]
    result = _combine_adjacent(make_frame_tags_dict(tags, 1), allow_single_frame=True, fps=10.0)
    assert len(result) == 2
