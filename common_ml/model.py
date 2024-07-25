from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

from .tags import VideoTag, FrameTag

class VideoModel(ABC):
    @abstractmethod
    def tag(self, data: Any) -> List[VideoTag]:
        pass

class VideoFrameModel(ABC):
    @abstractmethod
    def tag(self, video: Any) -> Tuple[List[VideoTag], Dict[int, List[FrameTag]]]:
        pass

    # Converts FrameTags to list of video tags by merging frame tags that occur in successive key frames
    # Args:
    #   frame_tags: map from frame idx to list of FrameTags
    #   fps: frames per second of the video
    #   allow_single_frame: if true, then tags that occur in just a single frame will be included, else they will be removed from the result
    @staticmethod
    def combine_adjacent(frame_tags: Dict[int, List[FrameTag]], fps: float, allow_single_frame: bool) -> List[VideoTag]:
        # maps a tag text value to frames where this tag occurs
        tag_to_frames = {} 
        for f_idx, tags in frame_tags.items():
            for tag in tags:
                text = tag.text
                if text not in tag_to_frames:
                    tag_to_frames[text] = []
                tag_to_frames[text].append(f_idx)

        # get id map: maps a key frame index to the next key frame index
        id_map = {}
        frame_indices = list(sorted(frame_tags.keys()))
        for i in range(len(frame_indices)-1):
            id_map[frame_indices[i]] = frame_indices[i+1]

        result = []

        # merge identical tags contained in adjacent key frames to a single tag
        for tag, frames in tag_to_frames.items():
            left, right = frames[0], frames[0]
            intervals = []
            for f in sorted(frames)[1:]:
                if id_map[right] == f:
                    # the tag occurs in adjacent key frame, so extend the right bound
                    right = f
                else:
                    intervals.append((left, right))
                    left, right = f, f

            # handle the last tag
            intervals.append((left, right))

            # store a tag for each valid interval
            for intv in intervals:
                left, right = intv
                if allow_single_frame or right > left:
                    result.append(VideoTag(
                        text=tag,
                        start_time=_to_milliseconds(left/fps),
                        end_time=_to_milliseconds((right+1)/fps),
                    ))
            
        return result
    
def _to_milliseconds(seconds: float) -> int:
    return round(seconds * 1000)