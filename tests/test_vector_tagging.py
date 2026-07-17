import io
import json
from typing import List, Optional

import numpy as np
import pytest

from common_ml.tagging.messages import VectorTag, Tag, BaseTag
from common_ml.tagging.models.frame_based import FrameModel, BatchFrameModel
from common_ml.tagging.models.av import AVModel
from common_ml.tagging.models.tag_types import FrameTag, FrameVectorTag
from common_ml.tagging.models.video_based import VideoVectorModel
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
            FrameVectorTag(vector=[base + i for i in range(self.dim)], box=_BOX),
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


# ===========================================================================
# VideoVectorModel + AVModel.from_video_vector_model
#
# These exercise the temporal-aware path: the whole video (or each segment
# window) is embedded by the model itself, rather than embedding frames
# independently and mean-pooling. Windows are passed to embed_video in
# milliseconds, matching VectorTag.start_time/end_time.
# ===========================================================================

class _RecordingVideoModel(VideoVectorModel):
    """Records every (start_ms, end_ms) window and the normalize flag it is asked
    for, and returns a deterministic vector per call, so segmentation / pooling /
    normalization threading are checkable without loading a real embedding model.

    Window i -> raw vector [i, i+1, ..., i+dim-1], L2-normalized iff normalize.
    """
    dim = 4

    def __init__(self):
        self.calls: List[tuple] = []
        self.normalize_seen: List[Optional[bool]] = []

    def embed_video(self, fpath: str, start_ms: Optional[int] = None,
                    end_ms: Optional[int] = None, normalize: Optional[bool] = None) -> List[float]:
        idx = len(self.calls)
        self.calls.append((start_ms, end_ms))
        self.normalize_seen.append(normalize)
        raw = np.asarray([float(idx) + i for i in range(self.dim)], dtype=np.float64)
        # None -> the implementation's default (normalized), matching the contract
        if normalize is None or normalize:
            n = np.linalg.norm(raw)
            if n > 0:
                raw = raw / n
        return raw.tolist()


class _EmptyVideoModel(VideoVectorModel):
    def __init__(self):
        self.calls: List[tuple] = []

    def embed_video(self, fpath, start_ms=None, end_ms=None, normalize=None):
        self.calls.append((start_ms, end_ms))
        return []


class _ZeroVideoModel(VideoVectorModel):
    dim = 4

    def embed_video(self, fpath, start_ms=None, end_ms=None, normalize=None):
        return [0.0] * self.dim


class _FlakyVideoModel(VideoVectorModel):
    """Raises on chosen window index/indices; returns a deterministic raw vector
    otherwise. Used to check that a bad segment is skipped, not fatal."""
    dim = 4

    def __init__(self, fail_on):
        self.fail_on = set(fail_on) if not isinstance(fail_on, int) else {fail_on}
        self.calls: List[tuple] = []

    def embed_video(self, fpath, start_ms=None, end_ms=None, normalize=None):
        idx = len(self.calls)
        self.calls.append((start_ms, end_ms))
        if idx in self.fail_on:
            raise RuntimeError(f"boom on window {idx}")
        return [float(idx) + i for i in range(self.dim)]


def _expected_windows(duration_ms: int, segment_length_s: Optional[float]) -> List[tuple]:
    """Mirror the factory's windowing so tests assert against an independent computation."""
    if segment_length_s is None or segment_length_s <= 0 or duration_ms <= round(segment_length_s * 1000):
        return [(0, duration_ms)]
    seg_ms = round(segment_length_s * 1000)
    windows = []
    start = 0
    while start < duration_ms:
        windows.append((start, min(start + seg_ms, duration_ms)))
        start += seg_ms
    return windows


def test_video_vector_model_is_abstract():
    # embed_video is abstract -> cannot instantiate without implementing it
    with pytest.raises(TypeError):
        VideoVectorModel()  # type: ignore[abstract]


def test_video_vector_whole_video_single_window(test_videos: List[str]):
    fpath = test_videos[0]
    duration_ms = round(get_duration(fpath) * 1000)

    fake = _RecordingVideoModel()
    model = AVModel.from_video_vector_model(fake, segment_length_s=None)
    tags = model.tag(fpath)

    # embedded exactly once, over the whole media, in milliseconds
    assert fake.calls == [(0, duration_ms)]

    assert len(tags) == 1
    v = tags[0]
    assert isinstance(v, VectorTag)
    assert v.frame_info is None
    assert v.start_time == 0
    assert v.end_time == duration_ms          # spans whole media, in ms
    assert len(v.vector) == fake.dim
    # normalized by default
    assert abs(float(np.linalg.norm(v.vector)) - 1.0) < 1e-6


def test_video_vector_short_video_collapses_to_one_window(test_videos: List[str]):
    # 2.mp4 is < 30s, so the default 30s segment length yields a single window
    fpath = test_videos[1]
    duration_ms = round(get_duration(fpath) * 1000)
    assert duration_ms < 30_000  # guard the premise of this test

    fake = _RecordingVideoModel()
    model = AVModel.from_video_vector_model(fake)  # default segment_length_s=30
    tags = model.tag(fpath)

    assert fake.calls == [(0, duration_ms)]
    assert len(tags) == 1
    assert tags[0].end_time == duration_ms


def test_video_vector_segmentation_windows_and_pooling(test_videos: List[str]):
    fpath = test_videos[0]
    duration_ms = round(get_duration(fpath) * 1000)
    windows = _expected_windows(duration_ms, 10.0)
    assert len(windows) >= 2  # 30s video / 10s segments -> multiple windows

    fake = _RecordingVideoModel()
    model = AVModel.from_video_vector_model(fake, segment_length_s=10.0)
    tags = model.tag(fpath)

    # each segment embedded once, with contiguous ms windows covering the media
    assert fake.calls == windows
    assert windows[0][0] == 0
    assert windows[-1][1] == duration_ms
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert next_start == prev_end  # contiguous, no gaps/overlap

    # pooled-only output by default
    assert len(tags) == 1
    pooled = tags[0]
    assert pooled.frame_info is None
    assert pooled.start_time == 0
    assert pooled.end_time == duration_ms

    # normalize=True is threaded to embed_video, so each segment is normalized
    # before pooling; the pooled vector is then re-normalized.
    assert all(n is True for n in fake.normalize_seen)
    raw = [np.asarray([float(i) + j for j in range(fake.dim)], dtype=np.float64) for i in range(len(windows))]
    units = [v / np.linalg.norm(v) for v in raw]
    expected = np.mean(units, axis=0)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(pooled.vector, expected)


def test_video_vector_emit_segment_vectors(test_videos: List[str]):
    fpath = test_videos[0]
    duration_ms = round(get_duration(fpath) * 1000)
    windows = _expected_windows(duration_ms, 10.0)

    fake = _RecordingVideoModel()
    model = AVModel.from_video_vector_model(
        fake, segment_length_s=10.0, normalize=False, emit_segment_vectors=True
    )
    tags = model.tag(fpath)

    # one tag per segment, plus a trailing pooled tag
    assert len(tags) == len(windows) + 1

    segment_tags = tags[:-1]
    pooled = tags[-1]

    # segment tags carry the real window timestamps
    for tag, (start_ms, end_ms) in zip(segment_tags, windows):
        assert isinstance(tag, VectorTag)
        assert tag.frame_info is None
        assert tag.start_time == start_ms
        assert tag.end_time == end_ms

    # pooled tag is emitted last and spans the whole media
    assert pooled.start_time == 0
    assert pooled.end_time == duration_ms

    # with normalize=False, pooled == plain mean of the segment vectors
    expected = np.mean([t.vector for t in segment_tags], axis=0)
    assert np.allclose(pooled.vector, expected)


def test_video_vector_empty_embeddings_emit_nothing(test_videos: List[str]):
    fake = _EmptyVideoModel()
    model = AVModel.from_video_vector_model(fake, segment_length_s=10.0)
    # every window returns an empty vector -> nothing to pool -> no tags
    assert model.tag(test_videos[0]) == []
    assert len(fake.calls) >= 1  # windows were still attempted


def test_video_vector_zero_norm_guard(test_videos: List[str]):
    fake = _ZeroVideoModel()
    model = AVModel.from_video_vector_model(fake, segment_length_s=10.0, normalize=True)
    tags = model.tag(test_videos[0])

    # all-zero vectors: norm>0 guard prevents division -> no NaN/inf, one pooled tag
    assert len(tags) == 1
    assert all(x == 0.0 for x in tags[0].vector)
    assert not np.isnan(np.asarray(tags[0].vector)).any()


def test_video_vector_result_is_avmodel(test_videos: List[str]):
    # the factory returns an AVModel, so run_default's AVModel branch consumes it
    fake = _RecordingVideoModel()
    model = AVModel.from_video_vector_model(fake)
    assert isinstance(model, AVModel)
    tags = model.tag(test_videos[0])
    assert all(isinstance(t, BaseTag) for t in tags)


def test_video_vector_threads_normalize_flag(test_videos: List[str]):
    fpath = test_videos[0]

    # normalize=True -> segments requested normalized -> each emitted segment is unit
    fake = _RecordingVideoModel()
    model = AVModel.from_video_vector_model(
        fake, segment_length_s=10.0, normalize=True, emit_segment_vectors=True
    )
    tags = model.tag(fpath)
    assert fake.normalize_seen and all(n is True for n in fake.normalize_seen)
    for t in tags[:-1]:  # segment tags (pooled is last)
        assert abs(float(np.linalg.norm(t.vector)) - 1.0) < 1e-6

    # normalize=False -> segments requested raw
    fake2 = _RecordingVideoModel()
    model2 = AVModel.from_video_vector_model(
        fake2, segment_length_s=10.0, normalize=False, emit_segment_vectors=True
    )
    model2.tag(fpath)
    assert fake2.normalize_seen and all(n is False for n in fake2.normalize_seen)


def test_video_vector_bad_segment_is_skipped_not_fatal(test_videos: List[str]):
    fpath = test_videos[0]
    duration_ms = round(get_duration(fpath) * 1000)
    windows = _expected_windows(duration_ms, 10.0)
    assert len(windows) >= 2

    fake = _FlakyVideoModel(fail_on=1)  # second window raises
    model = AVModel.from_video_vector_model(
        fake, segment_length_s=10.0, normalize=False, emit_segment_vectors=True
    )
    tags = model.tag(fpath)  # must NOT raise

    # every window was attempted despite the mid-stream failure
    assert len(fake.calls) == len(windows)

    pooled = tags[-1]
    segment_tags = tags[:-1]
    assert pooled.start_time == 0 and pooled.end_time == duration_ms

    # the failed window (index 1) is skipped -> its timestamps are absent
    surviving_windows = [w for i, w in enumerate(windows) if i != 1]
    assert [(t.start_time, t.end_time) for t in segment_tags] == surviving_windows

    # pooled == plain mean over the surviving raw segment vectors
    surviving_vecs = [[float(i) + j for j in range(fake.dim)]
                      for i in range(len(windows)) if i != 1]
    assert np.allclose(pooled.vector, np.mean(surviving_vecs, axis=0))


def test_video_vector_all_segments_fail_emit_nothing(test_videos: List[str]):
    fake = _FlakyVideoModel(fail_on=range(10_000))  # every window raises
    model = AVModel.from_video_vector_model(fake, segment_length_s=10.0)
    # all segments fail -> no usable vectors -> no tags, and no exception propagates
    assert model.tag(test_videos[0]) == []
    assert len(fake.calls) >= 1
