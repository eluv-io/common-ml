from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Dict, List, Optional, cast
from abc import ABC, abstractmethod

import numpy as np

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
    def from_frame_vector_model(
        frame_model: BatchFrameModel,
        fps: float,
        normalize: bool = True,
        emit_frame_vectors: bool = False,
    ) -> 'AVModel':
        """Produce a single mean-pooled vector for the whole video.

        Args:
            fps: rate at which to sample frames for pooling.
            normalize: L2-normalize the pooled vector (for cosine-similarity retrieval).
            emit_frame_vectors: additionally emit each per-frame VectorTag when True, 
                reusing per-frame tagging from the above from_frame_model.
        """
        assert fps > 0

        class NewModel(AVModel):
            def tag(self, fpath: str) -> List[BaseTag]:
                key_frames, frame_indices, times = get_frames(video_file=fpath, fps=fps)
                ftags_by_img = frame_model.tag_frames(key_frames)

                out: List[BaseTag] = []
                all_vecs: List[List[float]] = []
                for fidx, time_s, ftags in zip(frame_indices, times, ftags_by_img):
                    for ft in ftags:
                        # this factory requires a vector frame model; narrow the base
                        # BatchFrameModel type for type checking
                        vec_ft = cast(FrameVectorTag, ft)
                        all_vecs.append(vec_ft.vector)
                        if emit_frame_vectors:
                            out.append(self._frame_tag_to_video_tag(vec_ft, fidx, fpath, time_s))

                if not all_vecs:
                    return out

                pooled = np.mean(np.asarray(all_vecs, dtype=np.float64), axis=0)
                if normalize:
                    norm = np.linalg.norm(pooled)
                    if norm > 0:
                        pooled = pooled / norm

                out.append(VectorTag(
                    vector=pooled.tolist(),
                    start_time=0,
                    end_time=self._to_milliseconds(get_duration(fpath)),  # whole media duration
                    source_media=fpath,
                    track="",
                    frame_info=None,
                ))
                return out

        return NewModel()

    @staticmethod
    def from_video_vector_model(
        model: VideoVectorModel,
        segment_length_s: Optional[float] = 30.0,
        normalize: bool = True,
        emit_segment_vectors: bool = False,
    ) -> 'AVModel':
        """Produce a whole-video vector via the model's own (temporal-aware) video path.

        Unlike `from_frame_vector_model` (which embeds frames independently and
        mean-pools, discarding temporal/motion information), the embedding of each
        window is delegated to `model.embed_video`, so any temporal structure the
        model captures is preserved.

        The video is split into fixed-length segments and each segment is embedded
        over its own ``(start_ms, end_ms)`` window; this keeps memory bounded and
        frame sampling dense regardless of total length (a 30s window is sampled
        far more densely than one pass over an 8-hour file). The per-segment
        vectors are mean-pooled into a single whole-video `VectorTag`. Set
        `emit_segment_vectors=True` to additionally emit one `VectorTag` per
        segment (with real timestamps) for finer-grained retrieval.

        This factory is model-agnostic: it works with any `VideoVectorModel`
        (native video encoder, 3D-CNN, ...), not just frame-based embedders.

        Args:
            model: any VideoVectorModel implementing
                `embed_video(fpath, start_ms, end_ms) -> List[float]`.
            segment_length_s: segment duration in seconds. ``None`` (or a value
                >= the media duration) means embed the whole video as one window.
            normalize: L2-normalize emitted vectors (for cosine-similarity retrieval).
            emit_segment_vectors: also emit one VectorTag per segment.
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

                def _l2(vec: np.ndarray) -> np.ndarray:
                    if normalize:
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            return vec / norm
                    return vec

                out: List[BaseTag] = []
                segment_vecs: List[np.ndarray] = []
                for (start_ms, end_ms) in windows:
                    vec = np.asarray(model.embed_video(fpath, start_ms, end_ms), dtype=np.float64)
                    if vec.size == 0:
                        continue
                    segment_vecs.append(vec)
                    if emit_segment_vectors:
                        out.append(VectorTag(
                            vector=_l2(vec).tolist(),
                            start_time=start_ms,
                            end_time=end_ms,
                            source_media=fpath,
                            track="",
                            frame_info=None,
                        ))

                if not segment_vecs: # no vector tags -> return empty list
                    return out

                pooled = _l2(np.mean(np.asarray(segment_vecs, dtype=np.float64), axis=0))
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
