
import os
from dacite import from_dict

from common_ml.tagging.running import *
from common_ml.tagging.messages import *
from common_ml.tagging.model_types import *


def test_default_tag(video_model: VideoModel, test_videos: List[str], test_folder: str):
    output_path = os.path.join(test_folder, "out.jsonl")
    default_tag(video_model, files=test_videos, output_path=output_path)

    with open(output_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 4
        
        tag1 = from_dict(Tag, json.loads(lines[0]))
        assert tag1.tag == "action"
        assert tag1.start_time == 0 and tag1.end_time == 1000
        assert tag1.source_media == test_videos[0]
        
        tag2 = from_dict(Tag, json.loads(lines[1]))
        assert tag2.tag == "dialog"