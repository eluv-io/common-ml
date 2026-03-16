
from functools import lru_cache
from typing import Dict, List
from dataclasses import dataclass
import numpy as np

from common_ml.tagging.models.abstract import BatchFrameModel, FrameModel, VideoModel
from common_ml.tagging.models.tag_types import FrameInfo, FrameTag, Tag
from common_ml.video_processing import get_fps, get_frames


def get_video_model_from_frame_model(
    frame_model: BatchFrameModel, 
    fps: float, 
    allow_single_frame: bool
) -> VideoModel:
    assert fps > 0

    @dataclass
    class TagWithPos:
        # index amongst the sampled frames
        pos: int
        tag: Tag

    class NewModel(VideoModel):
        def tag_video(self, fpath: str) -> List[Tag]:
            key_frames, frame_indices, _ = get_frames(video_file=fpath, fps=fps)
            video_fps = get_fps(fpath)
            # flatten frame tags into a list with frame_info populated
            tagged_w_pos: List[TagWithPos] = []
            ftag_by_img = frame_model.tag_frames(key_frames)
            for pos, (fidx, ftags) in enumerate(zip(frame_indices, ftag_by_img)):
                for t in ftags:
                    converted_tag = self._frame_tag_to_video_tag(t, fidx, fpath)
                    tagged_w_pos.append(
                        TagWithPos(
                            pos=pos,
                            tag=converted_tag
                        ))

            # combined adjacent frame level tags with the same value into longer tags
            combined_tags = self._combine_adjacent(tagged_w_pos, allow_single_frame, video_fps)

            frame_level_tags = [t.tag for t in tagged_w_pos]

            return frame_level_tags + combined_tags
        
        def _combine_adjacent(self, tags: List[TagWithPos], allow_single_frame: bool, fps: float) -> List[Tag]:
            if len(tags) == 0:
                return []
            
            frame_time = self._to_milliseconds(1 / fps)

            # group TagWithPos entries by tag value
            tag_to_items: Dict[str, List[TagWithPos]] = {}
            for twp in tags:
                text = twp.tag.tag
                if text not in tag_to_items:
                    tag_to_items[text] = []
                tag_to_items[text].append(twp)

            result = []

            # merge runs of consecutive pos values into a single longer tag
            for text, items in tag_to_items.items():
                sorted_items = sorted(items, key=lambda x: x.pos)
                left = sorted_items[0]
                right = sorted_items[0]
                for item in sorted_items[1:]:
                    if item.pos == right.pos + 1:
                        right = item
                    else:
                        if allow_single_frame or right.pos > left.pos:
                            result.append(Tag(
                                tag=text,
                                start_time=left.tag.start_time,
                                end_time=right.tag.end_time + frame_time,
                                source_media=left.tag.source_media,
                                track=left.tag.track,
                                frame_info=None,
                            ))
                        left = item
                        right = item

                # handle the last run
                if allow_single_frame or right.pos > left.pos:
                    result.append(Tag(
                        tag=text,
                        start_time=left.tag.start_time,
                        end_time=right.tag.end_time + frame_time,
                        source_media=left.tag.source_media,
                        track=left.tag.track,
                        frame_info=None,
                    ))

            return result
        
        @lru_cache(maxsize=1024)
        def _cached_fps(self, source_media: str):
            return get_fps(source_media)
    
        def _frame_tag_to_video_tag(self, frame_tag: FrameTag, frame_idx: int, source_media: str) -> Tag:
            ts = self._to_milliseconds(frame_idx / self._cached_fps(source_media))
            return Tag(
                tag=frame_tag.tag,
                start_time=ts,
                end_time=ts,
                source_media=source_media,
                track="",
                frame_info=FrameInfo(frame_idx=frame_idx, box=frame_tag.box),
            )
        
        def _to_milliseconds(self, seconds: float) -> int:
            return round(seconds * 1000)
        
    return NewModel()

def batchify_frame_model(model: FrameModel) -> BatchFrameModel:
    class NewModel(BatchFrameModel):
        def tag_frames(self, imgs: np.ndarray) -> List[List[FrameTag]]:
            return [model.tag_frame(img) for img in imgs]
    return NewModel()