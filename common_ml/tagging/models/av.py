from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Dict, Iterator, List, Optional
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger

from common_ml.tagging.messages import Tag
from common_ml.tagging.models.tag_types import FrameTag, FrameInfo, Tag
from common_ml.tagging.models.frame_based import BatchFrameModel
from common_ml.video_processing import get_frames, get_fps

class AVModel(ABC):
    @abstractmethod
    def tag(self, fpath: str) -> List[Tag]:
        pass

    def on_completion(self) -> Iterator[Tag]:
        """
        Optional finalization hook, called once after all input files have been processed.
        Override to emit any tags that can only be produced once the full input stream is
        known, such as a shot that may span across contiguous input files and
        cannot be closed at the end of one tag() call.
        Defaults to yield nothing.
        """
        yield from ()

    @staticmethod
    def _to_milliseconds(seconds: float) -> int:
        return round(seconds * 1000)

    @staticmethod
    def _frame_tag_to_video_tag(frame_tag: FrameTag, frame_idx: int, source_media: str, time_s: float) -> Tag:
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

                tag_to_items: Dict[str, List[TagWithPos]] = {}
                for twp in tags:
                    if twp.tag.vector is not None:
                        # run-length merging applies to string tags (Tag) only; 
                        # tags with other payloads (e.g. vectors) pass through as per-frame tags
                        continue
                    key = twp.tag.tag
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
