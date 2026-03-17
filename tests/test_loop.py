import os
import sys
import time
import multiprocessing

from common_ml.tagging.run_helpers import start_loop_from_frame_model
from common_ml.tagging.models.frame_based import *

def _run_tag_loop(model, output_path, read_fd, write_fd):
    # clever claude fix for the test not terminating, we need the ref count for the FD to hit 0 to trigger EOF
    # and there is a reference in both parent and subprocess. 
    os.close(write_fd)
    sys.stdin = os.fdopen(read_fd, 'r')
    try:
        start_loop_from_frame_model(model, output_path)
    except Exception as e:
        print(e)

def test_loop(frame_model: FrameModel, test_videos: List[str], test_images: List[str], test_folder: str):
    output_path = os.path.join(test_folder, "out.jsonl")

    read_fd, write_fd = os.pipe()
    proc = multiprocessing.Process(target=_run_tag_loop, args=(frame_model, output_path, read_fd, write_fd))
    proc.start()
    os.close(read_fd)
    write_pipe = os.fdopen(write_fd, 'w')

    write_pipe.write("\n".join(test_videos) + "\n")
    write_pipe.flush()

    try:
        time.sleep(1)

        with open(output_path, "r") as f:
            lines = f.readlines()
        tag_lines = [l for l in lines if "tag" in l]
        num_tag_lines = len(tag_lines)
        assert len(tag_lines) > 100

        status_lines = [l for l in lines if "progress" in l]
        assert len(status_lines) == 2
        assert test_videos[0] in status_lines[0]
        assert test_videos[1] in status_lines[1]

        write_pipe.write("\n".join(test_images) + "\n")
        write_pipe.flush()

        time.sleep(2)

        with open(output_path, "r") as f:
            lines = f.readlines()

        tag_lines = [l for l in lines if "tag" in l]
        # we should get 3 more from tagging the images (2 from the first 1 from second)
        assert len(tag_lines) == num_tag_lines + 3

        status_lines = [l for l in lines if "progress" in l]
        assert len(status_lines) == 4
    finally:
        write_pipe.close()
        proc.join(timeout=5)