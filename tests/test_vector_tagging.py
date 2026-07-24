import io
import json
from typing import List

from common_ml.tagging.messages import Tag
from common_ml.tagging.models.frame_based import FrameModel, BatchFrameModel
from common_ml.tagging.models.av import AVModel
from common_ml.tagging.models.tag_types import FrameTag
from common_ml.tagging.file_tagger import FileTagger
from common_ml.tagging.run_helpers import write_message

_BOX = {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}


class _MixedFrameModel(FrameModel):
    """Emits a constant-label string tag (mergeable) plus a per-frame vector."""
    dim = 4

    def __init__(self):
        self.call_count = 0

    def tag_frame(self, img):
        base = float(self.call_count)
        self.call_count += 1
        return [
            FrameTag(tag="x", box=_BOX),                                    # string-only -> mergeable
            FrameTag(tag="", vector=[base + i for i in range(self.dim)], box=_BOX),  # vector -> per-frame
        ]


def test_vector_type_basics():
    v = Tag(tag="", vector=[0.1, 0.2, 0.3], start_time=0, end_time=1, source_media="m")
    # vector tag v is a Tag with a vector field; _combine_adjacent keys off
    # this (only Tag instances without vector (vector=None) run-length merged)
    assert isinstance(v, Tag)
    assert v.message_type == "tag" and v.vector is not None


def test_vector_frame_tag_images(vector_frame_model: FrameModel, test_images: List[str]):
    file_tagger = FileTagger.from_frame_model(vector_frame_model, fps=1.0, allow_single_frame=True)

    all_tags = []
    for fname in test_images:
        all_tags.extend(file_tagger.tag(fname))

    # one vector per image
    assert len(all_tags) == len(test_images)
    for tag, fname in zip(all_tags, test_images):
        assert isinstance(tag, Tag)
        assert len(tag.vector) == vector_frame_model.dim
        assert tag.source_media == fname
        assert tag.frame_info is not None
        assert len(tag.frame_info.box) == 4
        assert tag.additional_info == {"hello": "world"}


def test_vector_frame_tag_videos(vector_frame_model: FrameModel, test_videos: List[str]):
    file_tagger = FileTagger.from_frame_model(vector_frame_model, fps=1, allow_single_frame=True)

    all_tags = []
    for fname in test_videos:
        all_tags.extend(file_tagger.tag(fname))

    assert len(all_tags) > 0
    # vectors are not string Tags, so _combine_adjacent skips them => no
    # "combined" tags are produced; every emitted tag is a per-frame vector.
    for tag in all_tags:
        assert isinstance(tag, Tag)
        assert tag.source_media in test_videos
        assert tag.frame_info is not None      # frame-level only
        assert tag.start_time == tag.end_time  # instantaneous
        assert len(tag.vector) == vector_frame_model.dim

    combined = [t for t in all_tags if t.frame_info is None]
    assert combined == []


def test_vector_serialization():
    v = Tag(
        tag="",
        vector=[1.0, 2.0, 3.0],
        start_time=0,
        end_time=1000,
        source_media="m.mp4",
        track="",
        additional_info={"k": "v"},
    )
    buf = io.StringIO()
    write_message(v, buf)

    record = json.loads(buf.getvalue())
    assert record["type"] == "tag"
    assert record["data"]["vector"] == [1.0, 2.0, 3.0]
    assert record["data"]["source_media"] == "m.mp4"
    # the discriminator is a ClassVar, not a field -> must not leak into data
    assert "message_type" not in record["data"]


def test_string_and_vector_serialize_differently():
    t = Tag(tag="dog", start_time=0, end_time=1, source_media="m")
    v = Tag(tag="", vector=[0.5], start_time=0, end_time=1, source_media="m")

    tbuf, vbuf = io.StringIO(), io.StringIO()
    write_message(t, tbuf)
    write_message(v, vbuf)

    assert json.loads(tbuf.getvalue())["type"] == "tag"
    assert json.loads(vbuf.getvalue())["type"] == "tag"


# merge-skip pinned to AVModel.from_frame_model (no FileTagger layer)
def test_av_from_frame_model_vectors_skip_merge(vector_frame_model: FrameModel, test_videos: List[str]):
    # tests AVModel.from_frame_model with vectors
    # output shape: per-frame vectors, frame_info set, no combined tags
    batch = BatchFrameModel.from_frame_model(vector_frame_model)
    model = AVModel.from_frame_model(batch, fps=1, allow_single_frame=True)
    tags = model.tag(test_videos[0])

    assert len(tags) > 0
    assert all(isinstance(t, Tag) and t.vector is not None for t in tags)
    assert all(t.frame_info is not None for t in tags)     # per-frame only
    assert all(t.start_time == t.end_time for t in tags)
    assert [t for t in tags if t.frame_info is None] == []  # nothing merged


# in one stream, strings merge while vectors pass through
def test_av_from_frame_model_mixed_string_and_vector(test_videos: List[str]):
    batch = BatchFrameModel.from_frame_model(_MixedFrameModel())
    model = AVModel.from_frame_model(batch, fps=1, allow_single_frame=True)
    tags = model.tag(test_videos[0])

    vector_tags = [t for t in tags if t.vector is not None]
    string_tags = [t for t in tags if t.vector is None]
    assert vector_tags and string_tags

    # vectors never merge -> only per-frame
    assert all(t.frame_info is not None for t in vector_tags)

    # the constant "x" label produces at least one combined (frame_info None) interval
    string_combined = [t for t in string_tags if t.frame_info is None]
    assert string_combined
    for t in string_combined:
        assert t.tag == "x"
        assert t.end_time > t.start_time
