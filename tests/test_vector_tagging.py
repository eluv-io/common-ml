import io
import json
from typing import List

import numpy as np

from common_ml.tagging.messages import VectorTag, Tag, BaseTag
from common_ml.tagging.models.frame_based import FrameModel, BatchFrameModel
from common_ml.tagging.models.av import AVModel
from common_ml.tagging.models.tag_types import FrameTag, VectorFrameTag
from common_ml.tagging.file_tagger import FileTagger
from common_ml.tagging.run_helpers import write_message
from common_ml.video_processing import get_duration

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
            FrameTag(tag="x", box=_BOX),
            VectorFrameTag(vector=[base + i for i in range(self.dim)], box=_BOX),
        ]


class _EmptyFrameModel(FrameModel):
    def tag_frame(self, img):
        return []


class _ZeroVectorFrameModel(FrameModel):
    dim = 4

    def tag_frame(self, img):
        return [VectorFrameTag(vector=[0.0] * self.dim, box=_BOX)]


class _MultiVectorFrameModel(FrameModel):
    """Emits more than one vector per frame."""
    dim = 4
    per_frame = 2

    def __init__(self):
        self.call_count = 0

    def tag_frame(self, img):
        base = float(self.call_count)
        self.call_count += 1
        return [
            VectorFrameTag(vector=[base + k + i for i in range(self.dim)], box=_BOX)
            for k in range(self.per_frame)
        ]


def test_vector_type_basics():
    v = VectorTag(vector=[0.1, 0.2, 0.3], start_time=0, end_time=1, source_media="m")
    # a vector tag is a BaseTag but not a string Tag; _combine_adjacent keys off
    # this (only Tag instances are run-length merged)
    assert isinstance(v, BaseTag)
    assert not isinstance(v, Tag)
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
    # vectors are not string Tags, so _combine_adjacent skips them => no
    # "combined" tags are produced; every emitted tag is a per-frame vector.
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


def test_vector_av_pooling(vector_frame_model: FrameModel, test_videos: List[str]):
    # tests AVMode.from_vector_frame_model
    # output shape: one pooled vector, frame_info=None, end_time==duration
    batch = BatchFrameModel.from_frame_model(vector_frame_model)
    model = AVModel.from_vector_frame_model(batch, fps=1.0)

    fpath = test_videos[0]
    tags = model.tag(fpath)

    # exactly one video-level vector, no per-frame vectors by default
    assert len(tags) == 1
    v = tags[0]
    assert isinstance(v, VectorTag)
    assert v.frame_info is None
    assert v.start_time == 0
    # spans the whole media duration, not just the sampled range
    assert v.end_time == round(get_duration(fpath) * 1000)
    assert len(v.vector) == vector_frame_model.dim
    # normalized by default
    assert abs(float(np.linalg.norm(v.vector)) - 1.0) < 1e-6


def test_vector_av_pooling_emit_frames_no_normalize(vector_frame_model: FrameModel, test_videos: List[str]):
    batch = BatchFrameModel.from_frame_model(vector_frame_model)
    model = AVModel.from_vector_frame_model(batch, fps=1.0, normalize=False, emit_frame_vectors=True)

    fpath = test_videos[0]
    tags = model.tag(fpath)

    frame_tags = [t for t in tags if t.frame_info is not None]
    pooled_tags = [t for t in tags if t.frame_info is None]

    assert len(pooled_tags) == 1
    assert tags[-1].frame_info is None      # pooled vector is emitted last
    n = len(frame_tags)
    assert n > 0

    # per-frame vectors are instantaneous
    for t in frame_tags:
        assert isinstance(t, VectorTag)
        assert t.start_time == t.end_time

    # FakeVectorFrameModel yields frame i -> [i, i+1, i+2, i+3]; unnormalized mean
    # over n frames is [(n-1)/2 + j].
    expected = [(n - 1) / 2 + j for j in range(vector_frame_model.dim)]
    assert np.allclose(pooled_tags[0].vector, expected)


def test_string_and_vector_serialize_differently():
    t = Tag(tag="dog", start_time=0, end_time=1, source_media="m")
    v = VectorTag(vector=[0.5], start_time=0, end_time=1, source_media="m")

    tbuf, vbuf = io.StringIO(), io.StringIO()
    write_message(t, tbuf)
    write_message(v, vbuf)

    assert json.loads(tbuf.getvalue())["type"] == "tag"
    assert json.loads(vbuf.getvalue())["type"] == "vector_tag"


# merge-skip pinned to AVModel.from_frame_model (no FileTagger layer)
def test_av_from_frame_model_vectors_skip_merge(vector_frame_model: FrameModel, test_videos: List[str]):
    # tests AVModel.from_frame_model with vectors
    # output shape: per-frame vectors, frame_info set, no combined tags
    batch = BatchFrameModel.from_frame_model(vector_frame_model)
    model = AVModel.from_frame_model(batch, fps=1, allow_single_frame=True)
    tags = model.tag(test_videos[0])

    assert len(tags) > 0
    assert all(isinstance(t, VectorTag) for t in tags)
    assert all(t.frame_info is not None for t in tags)     # per-frame only
    assert all(t.start_time == t.end_time for t in tags)
    assert [t for t in tags if t.frame_info is None] == []  # nothing merged


# in one stream, strings merge while vectors pass through
def test_av_from_frame_model_mixed_string_and_vector(test_videos: List[str]):
    batch = BatchFrameModel.from_frame_model(_MixedFrameModel())
    model = AVModel.from_frame_model(batch, fps=1, allow_single_frame=True)
    tags = model.tag(test_videos[0])

    string_tags = [t for t in tags if isinstance(t, Tag)]
    vector_tags = [t for t in tags if isinstance(t, VectorTag)]
    assert string_tags and vector_tags

    # vectors never merge -> only per-frame
    assert all(t.frame_info is not None for t in vector_tags)

    # the constant "x" label produces at least one combined (frame_info None) interval
    string_combined = [t for t in string_tags if t.frame_info is None]
    assert string_combined
    for t in string_combined:
        assert t.tag == "x"
        assert t.end_time > t.start_time


# pooling edge cases
def test_vector_av_pooling_no_frames(test_videos: List[str]):
    # no vectors produced -> no pooled tag emitted (all_vecs empty branch)
    batch = BatchFrameModel.from_frame_model(_EmptyFrameModel())
    model = AVModel.from_vector_frame_model(batch, fps=1)
    assert model.tag(test_videos[0]) == []


def test_vector_av_pooling_zero_norm(test_videos: List[str]):
    # all-zero vectors: the norm>0 guard prevents division -> no NaN/inf
    batch = BatchFrameModel.from_frame_model(_ZeroVectorFrameModel())
    model = AVModel.from_vector_frame_model(batch, fps=1, normalize=True)
    tags = model.tag(test_videos[0])

    assert len(tags) == 1
    assert all(x == 0.0 for x in tags[0].vector)


def test_vector_av_pooling_multiple_per_frame(test_videos: List[str]):
    # 1 vector per video when there are multiple vectors per frame
    m = _MultiVectorFrameModel()
    batch = BatchFrameModel.from_frame_model(m)
    model = AVModel.from_vector_frame_model(batch, fps=1, normalize=False, emit_frame_vectors=True)
    tags = model.tag(test_videos[0])

    frame_tags = [t for t in tags if t.frame_info is not None]
    pooled = [t for t in tags if t.frame_info is None]

    assert len(pooled) == 1
    # every sampled frame contributes per_frame vectors
    assert len(frame_tags) % m.per_frame == 0
    assert len(frame_tags) // m.per_frame > 0
    # pooled == mean over ALL frame vectors (flattened across frames)
    expected = np.mean([t.vector for t in frame_tags], axis=0)
    assert np.allclose(pooled[0].vector, expected)
