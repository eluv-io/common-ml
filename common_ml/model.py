from abc import ABC, abstractmethod
import sys
from typing import List, Dict, Union, Callable
import cv2
import numpy as np
import json
import os
from dataclasses import asdict
from queue import Queue
import threading
import time

from .tags import Tag, FrameInfo
from .video_processing import get_frames
from .utils import get_file_type

class VideoModel(ABC):
    @abstractmethod
    def tag(self, fpath: str) -> List[Tag]:
        pass

class FrameModel(ABC):
    @abstractmethod
    def tag_frame(self, img: np.ndarray) -> List[Tag]:
        pass

def get_video_model_from_frame_model(frame_model: FrameModel, fps: float, allow_single_frame: bool) -> VideoModel:
    assert fps > 0

    class VideoFrameModel(VideoModel, FrameModel):
        # default method for running frame model on a video file
        def tag(self, fpath: str) -> List[Tag]:
            key_frames, fpos, _ = get_frames(video_file=fpath, fps=fps)
            # Flatten frame tags into a list with frame_info populated
            tagged: List[Tag] = []
            frame_tags_dict: Dict[int, List[Tag]] = {}
            for pos, frame in zip(fpos, key_frames):
                frame_result = frame_model.tag_frame(frame)
                for t in frame_result:
                    tag = Tag(
                        text=t.text,
                        start_time=t.start_time,
                        end_time=t.end_time,
                        source_media=t.source_media,
                        track=t.track,
                        frame_info=FrameInfo(frame_idx=pos, box=t.frame_info.box if t.frame_info else {}, confidence=t.frame_info.confidence if t.frame_info else None),
                    )
                    tagged.append(tag)
                    frame_tags_dict.setdefault(pos, []).append(tag) 
            video_tags = _combine_adjacent(frame_tags_dict, allow_single_frame, fps)
            return tagged + video_tags
        
        def tag_frame(self, img: np.ndarray) -> List[Tag]:
            return frame_model.tag_frame(img)

    return VideoFrameModel()
    

def _to_milliseconds(seconds: float) -> int:
    return round(seconds * 1000)

def _combine_adjacent(frame_tags_dict: Dict[int, List[Tag]], allow_single_frame: bool, fps: float) -> List[Tag]:
    if len(frame_tags_dict) == 0:
        return []

    f_intv = 1 / fps

    # maps a tag text value to (list of frame indices, representative tag for metadata)
    tag_to_frames: Dict[str, List[int]] = {}
    tag_representative: Dict[str, Tag] = {}
    for idx, tags in frame_tags_dict.items():
        for tag in tags:
            text = tag.text
            if text not in tag_to_frames:
                tag_to_frames[text] = []
                tag_representative[text] = tag
            tag_to_frames[text].append(idx)

    # get id map: maps a key frame index to the next key frame index
    id_map = {}
    frame_indices = list(sorted(frame_tags_dict.keys()))
    for i in range(len(frame_indices)):
        id_map[frame_indices[i]] = frame_indices[i+1] if i+1 < len(frame_indices) else frame_indices[i]  # last frame points to itself

    result = []

    # merge identical tags contained in adjacent key frames to a single tag
    for text, frames in tag_to_frames.items():
        rep = tag_representative[text]
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
                result.append(Tag(
                    text=text,
                    start_time=_to_milliseconds(left / fps),
                    end_time=_to_milliseconds(right / fps + f_intv),
                    source_media=rep.source_media,
                    track=rep.track,
                    frame_info=None,
                ))

    return result

# A generic tag function which accepts a list of media files and generates output tags for each file. 
# Args:
#   model: the model to use for tagging
#   files: a list of file paths to tag, can be image, video, or audio depending on the model
#   output_path: the path to save the output tags
def default_tag(model: Union[VideoModel, FrameModel], files: List[str], output_path: str="tags.jsonl") -> None:
    if len(files) == 0:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as fout:
        for fname in files:
            ftype = get_file_type(fname)
            assert ftype != "unknown", f"Unsupported file type for {fname}"
            if ftype == "image":
                assert isinstance(model, FrameModel), "For image files, model must be a FrameModel"
                if not os.path.exists(fname):
                    raise FileNotFoundError(f"File {fname} not found")
                img = cv2.imread(fname)
                # change color space to RGB
                assert img is not None, f"Failed to read image {fname}"
                img = img[:, :, ::-1]
                frametags = model.tag_frame(img)
                for tag in frametags:
                    fout.write(json.dumps(asdict(tag)) + '\n')
            elif ftype == "video":
                assert isinstance(model, VideoModel)
                vtags = model.tag(fname)
                for tag in vtags:
                    fout.write(json.dumps(asdict(tag)) + '\n')
            else:
                raise ValueError(f"Unsupported file type {ftype} for {fname}")
    
def run_live_mode(
    tag_fn: Callable[[List[str]], None], 
    batch_timeout: float=0.2,
) -> None:
    """
    Live mode: reads file paths from stdin and processes them in batches
    
    Args:
        tag_fn: Function that takes (file_paths: List[str])
    """
    
    file_queue = Queue()
    
    def stdin_reader():
        """Thread function to read from stdin and add files to queue"""
        try:
            for line in sys.stdin:
                line = line.strip()
                if line:
                    file_queue.put(line)
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            file_queue.put(None)  # Signal end of input
    
    def process_batch(files):
        """Process a batch of files using the provided function"""
        valid_files = []
        for f in files:
            if os.path.exists(f):
                valid_files.append(f)
            else:
                print(f"Warning: file {f} does not exist, skipping", file=sys.stderr)
        if valid_files:
            print(f"Processing batch of {len(valid_files)} files...", file=sys.stderr)
            tag_fn(valid_files)
            print(f"Completed batch of {len(valid_files)} files", file=sys.stderr)
    
    reader_thread = threading.Thread(target=stdin_reader, daemon=True)
    reader_thread.start()
    
    current_batch = []
    
    while True:
        try:
            while not file_queue.empty():
                try:
                    file_path = file_queue.get_nowait()
                    
                    if file_path is None:
                        if current_batch:
                            process_batch(current_batch)
                        return
                    
                    current_batch.append(file_path)
                except:
                    break
            
            if current_batch:
                process_batch(current_batch)
                current_batch = []
            
            if not reader_thread.is_alive() and file_queue.empty():
                break
            
            time.sleep(batch_timeout)
                
        except KeyboardInterrupt:
            break