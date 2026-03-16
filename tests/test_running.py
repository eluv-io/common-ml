
import os
from dacite import from_dict
import multiprocessing

from common_ml.tagging.running import *
from common_ml.tagging.messages import *
from common_ml.tagging.model_types import *


def test_default_tag(video_model: VideoModel, test_videos: List[str], test_folder: str):
    output_path = os.path.join(test_folder, "out.jsonl")
    default_tag(video_model, files=test_videos, output_path=output_path)

    with open(output_path, "r") as f:
        lines = f.readlines()

    # limit to tag lines
    lines = [l for l in lines if "tag" in l]

    assert len(lines) == 4
    
    tag1 = from_dict(Tag, json.loads(lines[0]))
    assert tag1.tag == "action"
    assert tag1.start_time == 0 and tag1.end_time == 1000
    assert tag1.source_media == test_videos[0]
    
    tag2 = from_dict(Tag, json.loads(lines[1]))
    assert tag2.tag == "dialog"

def test_default_tag_frame_model_images(frame_model: FrameModel, test_images: List[str], test_folder: str):
    output_path = os.path.join(test_folder, "out.jsonl")
    default_tag_frame_model(frame_model, test_images, output_path)

    with open(output_path, "r") as f:
        lines = f.readlines()

    # limit to tag lines
    lines = [l for l in lines if "tag" in l]
    
    assert len(lines) == 3

    tag1 = from_dict(Tag, json.loads(lines[0]))
    assert tag1.tag == "a"
    assert tag1.frame_info
    assert len(tag1.frame_info.box) == 4
    assert tag1.source_media == test_images[0]

    tag2 = from_dict(Tag, json.loads(lines[1]))
    assert tag2.tag == "b"
    assert tag2.source_media == test_images[0]

    tag3 = from_dict(Tag, json.loads(lines[2]))
    assert tag3.tag == "a"
    assert tag3.source_media == test_images[1]

def test_default_tag_frame_model_videos(frame_model: FrameModel, test_videos: List[str], test_folder: str):
    output_path = os.path.join(test_folder, "out.jsonl")
    default_tag_frame_model(frame_model, test_videos, output_path, fps=1, allow_single_frame=True)

    with open(output_path, "r") as f:
        lines = f.readlines()

    # limit to tag lines
    lines = [l for l in lines if "tag" in l]
    
    assert len(lines) == 122
    frame_lines = [l for l in lines if "frame_idx" in l]
    video_lines = [l for l in lines if "frame_idx" not in l]

    for line in lines:
        tag = from_dict(Tag, json.loads(line))
        assert tag.source_media in test_videos
        assert tag.tag in ["a", "b"]

    for line in frame_lines:
        tag = from_dict(Tag, json.loads(line))
        assert tag.frame_info
        assert len(tag.frame_info.box) == 4
        assert tag.start_time == tag.end_time

    for line in video_lines:
        tag = from_dict(Tag, json.loads(line))
        assert tag.end_time > tag.start_time

def test_default_tag_frame_model_videos_single_false(frame_model: FrameModel, test_videos: List[str], test_folder: str):
    output_path = os.path.join(test_folder, "out.jsonl")
    default_tag_frame_model(frame_model, test_videos, output_path, fps=1, allow_single_frame=False)

    with open(output_path, "r") as f:
        lines = f.readlines()

    # limit to tag lines
    lines = [l for l in lines if "tag" in l]

    video_lines = [l for l in lines if "frame_idx" not in l]

    for line in video_lines:
        tag = from_dict(Tag, json.loads(line))
        # we shouldn't have single frame video tags. 
        assert tag.end_time > tag.start_time + 1000

def _run_tag_loop(model, output_path, read_fd, write_fd):
    # clever claude fix for the test not terminating, we need the ref count for the FD to hit 0 to trigger EOF
    # and there is a reference in both parent and subprocess. 
    os.close(write_fd)
    sys.stdin = os.fdopen(read_fd, 'r')
    start_tag_loop(model, output_path)


def test_loop(frame_model: FrameModel, test_videos: List[str], test_images: List[str], test_folder: str):
    output_path = os.path.join(test_folder, "out.jsonl")

    read_fd, write_fd = os.pipe()
    proc = multiprocessing.Process(target=_run_tag_loop, args=(frame_model, output_path, read_fd, write_fd))
    proc.start()
    os.close(read_fd)
    write_pipe = os.fdopen(write_fd, 'w')

    write_pipe.write("\n".join(test_videos) + "\n")
    write_pipe.flush()

    time.sleep(1)

    with open(output_path, "r") as f:
        lines = f.readlines()
    lines = [l for l in lines if "tag" in l]
    num_lines = len(lines)
    assert len(lines) > 100

    write_pipe.write("\n".join(test_images) + "\n")
    write_pipe.flush()

    time.sleep(0.5)

    with open(output_path, "r") as f:
        lines = f.readlines()
    lines = [l for l in lines if "tag" in l]
    assert len(lines) == num_lines + 3

    write_pipe.close()
    proc.join(timeout=5)