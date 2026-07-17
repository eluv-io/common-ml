from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger

from common_ml.tagging.messages import BaseTag, VectorTag
from common_ml.tagging.models.tag_types import BaseFrameTag, FrameInfo, FrameTag, Tag, FrameVectorTag
from common_ml.tagging.models.frame_based import BatchFrameModel
from common_ml.tagging.models.video_based import VideoVectorModel
from common_ml.video_processing import get_frames, get_fps, get_duration

class AVModel(ABC):
    @abstractmethod
    def tag(self, fpath: str) -> List[BaseTag]:
        pass

    @staticmethod
    def _to_milliseconds(seconds: float) -> int:
        return round(seconds * 1000)

    @staticmethod
    def _frame_tag_to_video_tag(frame_tag: BaseFrameTag, frame_idx: int, source_media: str, time_s: float) -> BaseTag:
        ts = AVModel._to_milliseconds(time_s)
        return frame_tag.to_tag(
            start_time=ts,
            end_time=ts,
            source_media=source_media,
            track="",
            frame_info=FrameInfo(frame_idx=frame_idx, box=frame_tag.box),
        )

    @staticmethod
    def from_frame_model(
        frame_model: BatchFrameModel,
        fps: float,
        allow_single_frame: bool,
    ) -> 'AVModel':
        assert fps > 0

        @dataclass
        class TagWithPos:
            pos: int
            tag: BaseTag

        class NewModel(AVModel):
            def tag(self, fpath: str) -> List[BaseTag]:
                key_frames, frame_indices, _ = get_frames(video_file=fpath, fps=fps)
                video_fps = get_fps(fpath)
                tagged_w_pos: List[TagWithPos] = []
                ftag_by_img = frame_model.tag_frames(key_frames)
                for pos, (fidx, ftags) in enumerate(zip(frame_indices, ftag_by_img)):
                    for t in ftags:
                        converted_tag = self._frame_tag_to_video_tag(t, fidx, fpath, fidx / video_fps)
                        tagged_w_pos.append(TagWithPos(pos=pos, tag=converted_tag))

                combined_tags = self._combine_adjacent(tagged_w_pos, allow_single_frame, video_fps)
                frame_level_tags = [t.tag for t in tagged_w_pos]
                return frame_level_tags + combined_tags

            def _combine_adjacent(self, tags: List[TagWithPos], allow_single_frame: bool, fps: float) -> List[BaseTag]:
                if len(tags) == 0:
                    return []

                frame_time = self._to_milliseconds(1 / fps)

                tag_to_items: Dict[str, List[TagWithPos]] = {}
                for twp in tags:
                    if not isinstance(twp.tag, Tag):
                        # run-length merging applies to string tags (Tag) only; other
                        # payloads (e.g. vectors) pass through as per-frame tags
                        continue
                    key = twp.tag.tag
                    if key not in tag_to_items:
                        tag_to_items[key] = []
                    tag_to_items[key].append(twp)

                def combined(left: TagWithPos, right: TagWithPos) -> BaseTag:
                    # rebuild from a representative member so the payload (tag/vector/...)
                    # is carried over generically; drop frame-level-only fields
                    return replace(
                        left.tag,
                        start_time=left.tag.start_time,
                        end_time=right.tag.end_time + frame_time,
                        additional_info=None,
                        frame_info=None,
                    )

                result = []
                for key, items in tag_to_items.items():
                    sorted_items = sorted(items, key=lambda x: x.pos)
                    left = sorted_items[0]
                    right = sorted_items[0]
                    for item in sorted_items[1:]:
                        if item.pos == right.pos + 1:
                            right = item
                        else:
                            if allow_single_frame or right.pos > left.pos:
                                result.append(combined(left, right))
                            left = item
                            right = item

                    if allow_single_frame or right.pos > left.pos:
                        result.append(combined(left, right))

                return result

        return NewModel()

    @staticmethod
    def from_video_vector_model(
        model: VideoVectorModel,
        segment_length_s: Optional[float] = 30.0,
        normalize: bool = True,
        emit_segment_vectors: bool = False,
    ) -> 'AVModel':
        """Produce a whole-video vector via the model's own (temporal-aware) video path.

        Rather than embedding frames independently and mean-pooling (which
        discards temporal/motion information), the embedding of each window is
        delegated to `model.embed_video`, so any temporal structure the model
        captures is preserved.

        The video is split into fixed-length segments and each segment is embedded
        over its own ``(start_ms, end_ms)`` window; this keeps memory bounded and
        frame sampling dense regardless of total length (a 30s window is sampled
        far more densely than one pass over an 8-hour file). The per-segment
        vectors are mean-pooled into a single whole-video `VectorTag`. Set
        `emit_segment_vectors=True` to additionally emit one `VectorTag` per
        segment (with real timestamps) for finer-grained retrieval.

        This factory is model-agnostic: it works with any `VideoVectorModel`
        (native video encoder, QwenVL, 3D-CNN, ...), not just frame-based embedders.

        Args:
            model: any VideoVectorModel implementing
                `embed_video(fpath, start_ms, end_ms, normalize) -> List[float]`.
            segment_length_s: segment duration in seconds. ``None`` (or a value
                >= the media duration) means embed the whole video as one window.
            normalize: threaded to `embed_video` so each segment is normalized (or
                raw) before pooling, and then applied again to the pooled vector. 
                With normalize=True the output is cosine-similarity ready.
            emit_segment_vectors: also emit one VectorTag per segment when True.

        A single failing or empty segment is logged and skipped rather than aborting 
        the whole media; if every segment fails, no vector is emitted.
        """

        class NewModel(AVModel):
            def tag(self, fpath: str) -> List[BaseTag]:
                duration_s = get_duration(fpath)
                duration_ms = self._to_milliseconds(duration_s)

                # Build segment [start_ms, end_ms] windows over the media.
                if segment_length_s is None or segment_length_s <= 0 or duration_s <= segment_length_s:
                    windows = [(0, duration_ms)]
                else:
                    seg_ms = self._to_milliseconds(segment_length_s)
                    windows = []
                    start_ms = 0
                    while start_ms < duration_ms:
                        windows.append((start_ms, min(start_ms + seg_ms, duration_ms)))
                        start_ms += seg_ms

                out: List[BaseTag] = []
                segment_vecs: List[np.ndarray] = []
                for (start_ms, end_ms) in windows:
                    # Guard each segment: one bad window must not abort the whole
                    # media (which would discard every other segment's work and
                    # emit only an Error). Skip it and keep going.
                    try:
                        vec = np.asarray(
                            model.embed_video(fpath, start_ms, end_ms, normalize=normalize),
                            dtype=np.float64,
                        )
                    except Exception as e:
                        logger.warning(
                            f"from_video_vector_model: skipping segment [{start_ms}, {end_ms}]ms "
                            f"of {fpath}: {e!r}"
                        )
                        continue
                    if vec.size == 0:
                        logger.warning(
                            f"from_video_vector_model: empty embedding for segment "
                            f"[{start_ms}, {end_ms}]ms of {fpath}; skipping"
                        )
                        continue

                    # embed_video already applied `normalize`, so segment vectors are
                    # normalized (or raw) as requested -- emit them as-is.
                    segment_vecs.append(vec)
                    if emit_segment_vectors:
                        out.append(VectorTag(
                            vector=vec.tolist(),
                            start_time=start_ms,
                            end_time=end_ms,
                            source_media=fpath,
                            track="",
                            frame_info=None,
                        ))

                if not segment_vecs:  # no usable segments -> emit no vector
                    logger.warning(f"from_video_vector_model: no usable segments for {fpath}; emitting no vector")
                    return out

                # Pool over segments; re-normalize the pooled result when requested
                # (the mean of unit vectors is not itself unit-length).
                pooled = np.mean(np.asarray(segment_vecs, dtype=np.float64), axis=0)
                if normalize:
                    norm = np.linalg.norm(pooled)
                    if norm > 0:
                        pooled = pooled / norm
                out.append(VectorTag(
                    vector=pooled.tolist(),
                    start_time=0,
                    end_time=duration_ms,  # whole media duration
                    source_media=fpath,
                    track="",
                    frame_info=None,
                ))
                return out

        return NewModel()
