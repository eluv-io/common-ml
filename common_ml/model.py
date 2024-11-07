from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

from .tags import VideoTag, FrameTag
from .video_processing import get_key_frames

class VideoModel(ABC):
    @abstractmethod
    def tag(self, data: Any) -> List[VideoTag]:
        pass

class FrameModel(ABC):
    @abstractmethod
    def tag(self, img: Any) -> List[FrameTag]:
        pass

    # default method for running frame model on a video file
    def tag_video(self, video: str, allow_single_frame: bool, freq: int) -> Tuple[Dict[int, List[FrameTag]], List[VideoTag]]:
        assert freq > 0, "Frequency must be a positive integer"
        key_frames, fpos, ts = get_key_frames(video_file=video)
        ftags = {pos: self.tag(frame) for i, (pos, frame) in enumerate(zip(fpos, key_frames)) if i % freq == 0}
        ts = [t for i, t in enumerate(ts) if i % freq == 0]
        video_tags = FrameModel._combine_adjacent(ftags, ts, allow_single_frame)
        return ftags, video_tags
    
    @staticmethod
    def _combine_adjacent(frame_tags: Dict[int, List[FrameTag]], timestamps: List[float], allow_single_frame: bool) -> List[VideoTag]:
        if len(frame_tags) == 0:
            return []

        f_idx = sorted(list(frame_tags.keys()))
        f_idx_to_time = {f_idx: t for f_idx, t in zip(f_idx, timestamps)}
        
        _approx_fps = (f_idx[-1] - f_idx[0]) / (timestamps[-1] - timestamps[0])
        f_intv = 1 / _approx_fps
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
                        start_time=FrameModel._to_milliseconds(f_idx_to_time[left]),
                        end_time=FrameModel._to_milliseconds(f_idx_to_time[right]+f_intv),
                    ))
            
        return result
    
    @staticmethod
    def _to_milliseconds(seconds: float) -> int:
        return round(seconds * 1000)
