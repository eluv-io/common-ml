from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Dict, Hashable, List
from abc import ABC, abstractmethod

import numpy as np

from common_ml.tagging.messages import BaseTag, VectorTag
from common_ml.tagging.models.tag_types import BaseFrameTag, FrameInfo, FrameTag, Tag
from common_ml.tagging.models.frame_based import BatchFrameModel
from common_ml.video_processing import get_frames, get_fps, get_duration

class AVModel(ABC):
    @abstractmethod
    def tag(self, fpath: str) -> List[Tag]:
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
            tag: Tag

        class NewModel(AVModel):
            def tag(self, fpath: str) -> List[Tag]:
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

            def _combine_adjacent(self, tags: List[TagWithPos], allow_single_frame: bool, fps: float) -> List[Tag]:
                if len(tags) == 0:
                    return []

                frame_time = self._to_milliseconds(1 / fps)

                tag_to_items: Dict[Hashable, List[TagWithPos]] = {}
                for twp in tags:
                    key = twp.tag.grouping_key()
                    if key is None:
                        # payload opts out of run-length combination (e.g. vectors)
                        continue
                    if key not in tag_to_items:
                        tag_to_items[key] = []
                    tag_to_items[key].append(twp)

                def combined(left: TagWithPos, right: TagWithPos) -> Tag:
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
    def from_vector_frame_model(
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
            def tag(self, fpath: str) -> List[Tag]:
                key_frames, frame_indices, times = get_frames(video_file=fpath, fps=fps)
                ftags_by_img = frame_model.tag_frames(key_frames)

                out: List[Tag] = []
                all_vecs: List[List[float]] = []
                for fidx, time_s, ftags in zip(frame_indices, times, ftags_by_img):
                    for ft in ftags:
                        all_vecs.append(ft.vector)
                        if emit_frame_vectors:
                            out.append(self._frame_tag_to_video_tag(ft, fidx, fpath, time_s))

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
