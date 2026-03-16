from typing import List, Dict, Union
from functools import lru_cache
from dataclasses import dataclass
import cv2
import numpy as np

from common_ml.utils.files import get_file_type
from common_ml.tagging.abstract import FileTagger
from common_ml.tagging.models.abstract import BatchFrameModel, FrameModel, VideoModel
from common_ml.tagging.models.conversion import get_video_model_from_frame_model, batchify_frame_model
from common_ml.tagging.models.tag_types import FrameInfo, FrameTag, Tag
from common_ml.video_processing import get_fps, get_frames

def get_file_tagger_from_video_model(video_model: VideoModel) -> FileTagger:
    class NewFileTagger(FileTagger):
        def tag(self, file: str) -> List[Tag]:
            return video_model.tag_video(file)
    
    return NewFileTagger()

def get_file_tagger_from_frame_model(frame_model: Union[FrameModel, BatchFrameModel], fps: float, allow_single_frame: bool) -> FileTagger:
    if isinstance(frame_model, FrameModel):
        batched_frame_model = batchify_frame_model(frame_model)
    else:
        batched_frame_model = frame_model

    video_model = get_video_model_from_frame_model(batched_frame_model, fps, allow_single_frame=allow_single_frame)

    class NewFileTagger(FileTagger):
        def tag(self, file: str) -> List[Tag]:
            file_type = get_file_type(file)
            if file_type == "image":
                # use the frame model directly for images
                img = cv2.imread(file)
                if img is None:
                    raise ValueError(f"Could not read image file {file}")
                img = img[:, :, ::-1]
                frametags = batched_frame_model.tag_frames(np.array([img]))[0]

                tags = []
                for ftag in frametags:
                    out_tag = Tag(
                        start_time=0,
                        end_time=0,
                        tag=ftag.tag,
                        source_media=file,
                        track="",
                        frame_info=FrameInfo(frame_idx=0, box=ftag.box)
                    )
                    tags.append(out_tag)

                return tags
            elif file_type == "video":
                return video_model.tag_video(file)
            else:
                raise ValueError(f"Unsupported file type for {file}.")
        
    return NewFileTagger()