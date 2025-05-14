from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union
import cv2
import numpy as np
import json
import os
from dataclasses import asdict

from .tags import VideoTag, FrameTag
from .video_processing import get_frames, get_key_frames
from .utils import get_file_type

class VideoModel(ABC):
    @abstractmethod
    def tag(self, fpath: str) -> List[VideoTag]:
        pass

    def get_config(self) -> dict:
        return {}

    def set_config(self, config: dict) -> None:
        pass

class FrameModel(ABC):
    @abstractmethod
    def tag(self, img: np.ndarray) -> List[FrameTag]:
        pass

    def get_config(self) -> dict:
        return {}

    def set_config(self, config: dict) -> None:
        pass

    # default method for running frame model on a video file
    def tag_video(self, fpath: str) -> Tuple[Dict[int, List[FrameTag]], List[VideoTag]]:
        fps = self.get_config().get("fps", 1)
        allow_single_frame = self.get_config().get("allow_single_frame", False)
        assert fps > 0, "Frequency must be a positive integer"
        key_frames, fpos, ts = get_key_frames(fpath)
        ftags = {pos: self.tag(frame) for pos, frame in zip(fpos, key_frames)}
        video_tags = FrameModel._combine_adjacent(ftags, ts, allow_single_frame)
        return ftags, video_tags
    
    @staticmethod
    def _combine_adjacent(frame_tags: Dict[int, List[FrameTag]], timestamps: List[float], allow_single_frame: bool) -> List[VideoTag]:
        if len(frame_tags) == 0:
            return []
        if len(timestamps) <= 1:
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
        for i in range(len(frame_indices)):
            id_map[frame_indices[i]] = frame_indices[i+1] if i+1 < len(frame_indices) else frame_indices[i] # last frame points to itself

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

# A generic tag function which accepts a list of media files and generates output tags for each file. 
# Args:
#   model: the model to use for tagging
#   files: a list of file paths to tag, can be image, video, or audio depending on the model
#   output_path: the path to save the output tags
def default_tag(model: Union[VideoModel, FrameModel], files: List[str], output_path: str) -> None:
    if len(files) == 0:
        return 
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    ftype = get_file_type(files[0])
    assert all(get_file_type(f) == ftype for f in files), "All files must be of the same type"
    assert ftype != "unknown", "Unsupported file type"
    if isinstance(model, VideoModel):
        for fname in files:
            vtags = model.tag(fname)
            with open(os.path.join(output_path, f"{os.path.basename(fname)}_tags.json"), 'w') as fout:
                fout.write(json.dumps([asdict(tag) for tag in vtags]))
    elif isinstance(model, FrameModel):
        if ftype == "video":
            for fname in files:
                ftags, tags = model.tag_video(fname)
                with open(os.path.join(output_path, f"{os.path.basename(fname)}_tags.json"), 'w') as fout:
                    fout.write(json.dumps([asdict(tag) for tag in tags]))
                with open(os.path.join(output_path, f"{os.path.basename(fname)}_frametags.json"), 'w') as fout:
                    ftags = {k: [asdict(tag) for tag in v] for k, v in ftags.items()}
                    fout.write(json.dumps(ftags))
        elif ftype == "image":
            for fname in files:
                if not os.path.exists(fname):
                    raise FileNotFoundError(f"File {fname} not found")
                img = cv2.imread(fname)
                # change color space to RGB
                img = img[:, :, ::-1]
                frametags = model.tag(img)
                with open(os.path.join(output_path, f"{os.path.basename(fname)}_imagetags.json"), 'w') as fout:
                    fout.write(json.dumps([asdict(tag) for tag in frametags]))
    else:
        raise ValueError(f"Unsupported model type {type(model)}, should be either FrameModel or VideoModel")