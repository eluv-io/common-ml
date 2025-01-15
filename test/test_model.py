import argparse
from typing import Any, List
import random
from quick_test_py import Tester
import os
from dataclasses import asdict
import json

from common_ml.model import FrameModel, VideoModel
from common_ml.tags import FrameTag, VideoTag
from common_ml.model import default_tag
from common_ml.utils import get_file_type

media_path = os.path.join(os.path.dirname(__file__), 'test_media')

test_videos = []
test_images = []
for f in os.listdir(media_path):
    if get_file_type(f) == 'image':
        test_images.append(os.path.join(media_path, f))
    elif get_file_type(f) == 'video':
        test_videos.append(os.path.join(media_path, f))

class FakeFrameModel(FrameModel):
    def __init__(self):
        self.config = {}

    def tag(self, img: Any) -> List[FrameTag]:
        return self.random_tag()

    def random_tag(self) -> List[FrameTag]:
        num_tags = random.randint(1, 3)
        return [FrameTag.from_dict({"text": f"fake{random.randint(1, 10)}", "box": {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95}}) for _ in range(num_tags)]
    
    def set_config(self, config: dict) -> None:
        self.config = config

    def get_config(self) -> dict:
        return self.config
    
class FakeVideoModel(VideoModel):
    def __init__(self):
        self.config = {}

    def tag(self, fpath: str) -> List[VideoTag]:
        return self.random_tag()

    def random_tag(self) -> List[VideoTag]:
        return [VideoTag.from_dict({"text": f"fake{random.randint(1, 10)}", "start_time": i, "end_time": i+1}) for i in range(30)]
    
    def set_config(self, config: dict) -> None:
        self.config = config

    def get_config(self) -> dict:
        return self.config

def test_tag():
    random.seed(42)
    model = FakeFrameModel()
    test_video = test_videos[0]
    def t1():
        model.set_config({"fps": 1, "allow_single_frame": False})
        ftags, vtags = model.tag_video(test_video)
        return [{str(i): [asdict(ftag) for ftag in ft] for i, ft in ftags.items()}, [asdict(tag) for tag in vtags]]
    def t2():
        model.set_config({"fps": 1, "allow_single_frame": True})
        ftags, vtags = model.tag_video(test_video)
        return [{str(i): [asdict(ftag) for ftag in ft] for i, ft in ftags.items()}, [asdict(tag) for tag in vtags]]
    def t3():
        model.set_config({"fps": 2, "allow_single_frame": False})
        ftags, vtags = model.tag_video(test_video)
        return [{str(i): [asdict(ftag) for ftag in ft] for i, ft in ftags.items()}, [asdict(tag) for tag in vtags]]
    def t4():
        model.set_config({"fps": 2, "allow_single_frame": True})
        ftags, vtags = model.tag_video(test_video)
        return [{str(i): [asdict(ftag) for ftag in ft] for i, ft in ftags.items()}, [asdict(tag) for tag in vtags]]
    return [t1, t2, t3, t4]

def test_video_run():
    def video_model_test():
        random.seed(42)
        model = FakeVideoModel()
        outpath = os.path.join(os.path.dirname(__file__), 'tags/video')
        default_tag(model, test_videos, outpath)
        with open(os.path.join(outpath, 'test_tags.json')) as f:
            tags = json.load(f)
        with open(os.path.join(outpath, 'test2_tags.json')) as f:
            tags2 = json.load(f)
        return [tags, tags2]
    def frame_model_test():
        random.seed(42)
        model = FakeFrameModel()
        outpath = os.path.join(os.path.dirname(__file__), 'tags/frame')
        model.set_config({"fps": 1, "allow_single_frame": True})
        default_tag(model, test_videos, outpath)
        with open(os.path.join(outpath, 'test_tags.json')) as f:
            tags = json.load(f)
        with open(os.path.join(outpath, 'test2_tags.json')) as f:
            tags2 = json.load(f)
        with open(os.path.join(outpath, 'test_frametags.json')) as f:
            ftags = json.load(f)
        with open(os.path.join(outpath, 'test2_frametags.json')) as f:
            ftags2 = json.load(f)
        return [tags, ftags, tags2, ftags2]
    return [video_model_test, frame_model_test]

def test_image_run():
    def test():
        random.seed(42)
        model = FakeFrameModel()
        outpath = os.path.join(os.path.dirname(__file__), 'tags/frame')
        default_tag(model, test_images, outpath)
        tags = []
        for f in test_images:
            with open(os.path.join(outpath, f'{os.path.basename(f).split(".")[0]}_imagetags.json')) as f:
                tags.append(json.load(f))
        return tags
    return [test]

def test_run():
    tester = Tester(os.path.dirname(__file__) + '/test_data')
    tester.register('test_tag', test_tag())
    tester.validate()
    
def main():
    tester = Tester(os.path.dirname(__file__) + '/test_data')
    tester.register('test_tag', test_tag())
    tester.register('test_video_run', test_video_run())
    tester.register('test_image_run', test_image_run())
    if args.record:
        tester.record(args.tests)
    else:
        tester.validate(args.tests)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests', nargs='+')
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    main()