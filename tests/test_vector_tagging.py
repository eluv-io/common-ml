import io
import json
from typing import List

from common_ml.tagging.messages import VectorTag, Tag, BaseTag
from common_ml.tagging.models.frame_based import FrameModel
from common_ml.tagging.models.tag_types import VectorFrameTag
from common_ml.tagging.file_tagger import FileTagger
from common_ml.tagging.run_helpers import write_message


def test_vector_type_basics():
    v = VectorTag(vector=[0.1, 0.2, 0.3], start_time=0, end_time=1, source_media="m")
    # a vector tag is a BaseTag but not a string Tag
    assert isinstance(v, BaseTag)
    assert not isinstance(v, Tag)
    # vectors opt out of run-length combination
    assert v.grouping_key() is None
    assert v.message_type == "vector_tag"


def test_vector_frame_tag_images(vector_frame_model: FrameModel, test_images: List[str]):
    file_tagger = FileTagger.from_frame_model(vector_frame_model, fps=1.0, allow_single_frame=True)

    all_tags = []
    for fname in test_images:
        all_tags.extend(file_tagger.tag(fname))

    # one vector per image
    assert len(all_tags) == len(test_images)
    for tag, fname in zip(all_tags, test_images):
        assert isinstance(tag, VectorTag)
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
    # grouping_key() is None => no run-length "combined" tags are produced;
    # every emitted tag is a per-frame vector.
    for tag in all_tags:
        assert isinstance(tag, VectorTag)
        assert tag.source_media in test_videos
        assert tag.frame_info is not None      # frame-level only
        assert tag.start_time == tag.end_time  # instantaneous
        assert len(tag.vector) == vector_frame_model.dim

    combined = [t for t in all_tags if t.frame_info is None]
    assert combined == []


def test_vector_tag_serialization():
    v = VectorTag(
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
    assert record["type"] == "vector_tag"
    assert record["data"]["vector"] == [1.0, 2.0, 3.0]
    assert record["data"]["source_media"] == "m.mp4"
    # the discriminator is a ClassVar, not a field -> must not leak into data
    assert "message_type" not in record["data"]


def test_string_and_vector_serialize_differently():
    t = Tag(tag="dog", start_time=0, end_time=1, source_media="m")
    v = VectorTag(vector=[0.5], start_time=0, end_time=1, source_media="m")

    tbuf, vbuf = io.StringIO(), io.StringIO()
    write_message(t, tbuf)
    write_message(v, vbuf)

    assert json.loads(tbuf.getvalue())["type"] == "tag"
    assert json.loads(vbuf.getvalue())["type"] == "vector_tag"
