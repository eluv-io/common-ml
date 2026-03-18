import os
import shutil
import sys
import time
import multiprocessing

import pytest

from common_ml.tagging.producer import TagMessageProducer
from common_ml.tagging.run_helpers import start_loop_from_frame_model, start_loop_from_producer
from common_ml.tagging.models.frame_based import *
from common_ml.tagging.messages import *

def _run_tag_loop(model, output_path, read_fd, write_fd):
    # clever claude fix for the test not terminating, we need the ref count for the FD to hit 0 to trigger EOF
    # and there is a reference in both parent and subprocess. 
    os.close(write_fd)
    sys.stdin = os.fdopen(read_fd, 'r')
    try:
        start_loop_from_frame_model(model, output_path)
    except Exception as e:
        print(e)

def _run_producer_loop(producer, output_path, read_fd, write_fd, continue_on_error, max_batch):
    # clever claude fix for the test not terminating, we need the ref count for the FD to hit 0 to trigger EOF
    # and there is a reference in both parent and subprocess. 
    os.close(write_fd)
    sys.stdin = os.fdopen(read_fd, 'r')
    try:
        start_loop_from_producer(producer, output_path, continue_on_error=continue_on_error, batch_limit=max_batch)
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

def test_loop_with_exception(frame_model: FrameModel, test_videos: List[str], test_folder: str):
    output_path = os.path.join(test_folder, "out.jsonl")

    producer = TagMessageProducer.from_model(frame_model)

    class ExceptionProducer(TagMessageProducer):
        def produce(self, files: List[str]) -> List[Message]:
            if test_videos[-1] in files:
                raise RuntimeError("test exception")
            return producer.produce(files)

    read_fd, write_fd = os.pipe()
    proc = multiprocessing.Process(target=_run_producer_loop, args=(ExceptionProducer(), output_path, read_fd, write_fd, False, None))
    proc.start()
    os.close(read_fd)
    write_pipe = os.fdopen(write_fd, 'w')

    write_pipe.write("\n".join(test_videos) + "\n")
    write_pipe.flush()

    time.sleep(2)

    with open(output_path, "r") as f:
        lines = f.readlines()

    try:
        assert len(lines) == 1
        assert "error" in lines[0]
        assert "test exception" in lines[0]

        # process should exit
        assert proc.exitcode is not None
    finally:
        write_pipe.close()
        proc.join(timeout=5)

    os.remove(output_path)
    

    # try again but with continue_on_error=True and max_batch=1
    read_fd, write_fd = os.pipe()
    proc = multiprocessing.Process(target=_run_producer_loop, args=(ExceptionProducer(), output_path, read_fd, write_fd, True, 1))
    proc.start()
    os.close(read_fd)
    write_pipe = os.fdopen(write_fd, 'w')

    write_pipe.write("\n".join(test_videos) + "\n")
    write_pipe.flush()

    time.sleep(2)

    with open(output_path, "r") as f:
        lines = f.readlines()

    try:
        tag_lines = [l for l in lines if "tag" in l]
        num_tag_lines = len(lines)
        assert len(tag_lines) > 0
        error_lines = [l for l in lines if "error" in l]
        assert len(error_lines) == 1

        # write something else
        write_pipe.write("\n".join([test_videos[0]]) + "\n")
        write_pipe.flush()

        time.sleep(2)

        with open(output_path, "r") as f:
            lines = f.readlines()

        tag_lines = [l for l in lines if "tag" in l]
        assert len(tag_lines) > num_tag_lines # should keep tagging
    finally:
        write_pipe.close()
        proc.join(timeout=5)

def test_loop_with_error_message(frame_model: FrameModel, test_videos: List[str], test_folder: str):
    output_path = os.path.join(test_folder, "out.jsonl")

    producer = TagMessageProducer.from_model(frame_model)

    class ExceptionProducer(TagMessageProducer):
        def produce(self, files: List[str]) -> List[Message]:
            messages = producer.produce(files)
            messages.append(ErrorMessage(type="error", data=Error(message="test error message", source_media=files[-1])))
            return messages

    read_fd, write_fd = os.pipe()
    proc = multiprocessing.Process(target=_run_producer_loop, args=(ExceptionProducer(), output_path, read_fd, write_fd, False, None))
    proc.start()
    os.close(read_fd)
    write_pipe = os.fdopen(write_fd, 'w')

    write_pipe.write("\n".join(test_videos) + "\n")
    write_pipe.flush()

    time.sleep(2)

    with open(output_path, "r") as f:
        lines = f.readlines()

    try:
        assert len(lines) > 1
        assert "error" in lines[-1]
        assert "test error message" in lines[-1]

        # process should exit
        assert proc.exitcode is not None
    finally:
        write_pipe.close()
        proc.join(timeout=5)