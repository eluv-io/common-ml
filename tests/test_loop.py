import json
import os
import sys
import time
import multiprocessing
from typing import Iterator, List, Optional
from unittest.mock import patch

from common_ml.tagging.producer import TagMessageProducer
from common_ml.tagging.models.processor import TagProcessor
from common_ml.tagging.run_helpers import start_loop_from_frame_model, start_loop_from_producer, run_default
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

def test_loop_for_processor(tag_processor: TagProcessor, test_timestamp_files: List[str], test_folder: str):    

    output_path = os.path.join(test_folder, "out.jsonl")

    os.environ['ELV_CONTENT'] = "iq__pytest"
    os.environ['ELV_TOKEN'] = "apy_auth"

    read_fd, write_fd = os.pipe()
    proc = multiprocessing.Process(target=_run_producer_loop, args=(TagMessageProducer.from_tag_processor(tag_processor), output_path, read_fd, write_fd, False, None))
    proc.start()
    
    os.close(read_fd) ## close the read side of the pipe, we are the writer
    
    closed = False
    write_pipe = os.fdopen(write_fd, 'w')

    try:
        ## file 0 -- 1 part of a 2 part group
        write_pipe.write("\n".join(test_timestamp_files[0:1]) + "\n")
        write_pipe.flush()

        time.sleep(2.1)

        with open(output_path, "r") as f:
            lines = f.readlines()
        print(str(len(lines)) + " LINES:", "".join(lines))
        tag_lines = [l for l in lines if "tag" in l]
        num_tag_lines = len(tag_lines)
        skip = len(lines)
        print("number of tag lines (first set):", len(tag_lines))

        assert len(tag_lines) == 100

        status_lines = [l for l in lines if "progress" in l]
        assert len(status_lines) == 1

        assert test_timestamp_files[0] in status_lines[0]

        ## files 1-3 -- the rest of the first group, and a whole group
        write_pipe.write("\n".join(test_timestamp_files[1:4]) + "\n")
        write_pipe.flush()

        time.sleep(2.1)
        
        skip = len(lines)
        with open(output_path, "r") as f:
            lines = f.readlines()
            
        lines = lines[skip:]
        print(str(len(lines)) + " LINES:", "".join(lines))

        tag_lines = [l for l in lines if "tag" in l]
        print("number of tag lines (middle set):", len(tag_lines))
        assert len(tag_lines) == 300

        status_lines = [l for l in lines if "progress" in l]
        assert len(status_lines) == 3

        assert test_timestamp_files[1] in status_lines[0]
        assert test_timestamp_files[2] in status_lines[1] 
        assert test_timestamp_files[3] in status_lines[2]

        ## file 4 should not generate any tags (test processor cutoff range)
        write_pipe.write("\n".join(test_timestamp_files[4:5]) + "\n")
        write_pipe.flush()

        time.sleep(2.1)

        skip += len(lines)
        with open(output_path, "r") as f:
            lines = f.readlines()
            
        lines = lines[skip:]
        print(str(len(lines)) + " LINES:", "".join(lines))

        tag_lines = [l for l in lines if "tag" in l]
        print("number of tag lines (final set):", len(tag_lines))
        assert len(tag_lines) == 0

        status_lines = [l for l in lines if "progress" in l]
        assert len(status_lines) == 1

        assert test_timestamp_files[4] in status_lines[0]

        write_pipe.close()
        closed = True

        ## check on completion tags
        time.sleep(.5)

        skip += len(lines)
        with open(output_path, "r") as f:
            lines = f.readlines()
            
        lines = lines[skip:]
        print(str(len(lines)) + " LINES:", "".join(lines))


    finally:
        if not closed: write_pipe.close()
        proc.join(timeout=5)

def test_loop_with_exception(frame_model: FrameModel, test_videos: List[str], test_folder: str):
    output_path = os.path.join(test_folder, "out.jsonl")

    producer = TagMessageProducer.from_model(frame_model)

    class ExceptionProducer(TagMessageProducer):
        def produce(self, files: List[str]) -> Iterator[Message]:
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
        def produce(self, files: List[str]) -> Iterator[Message]:
            messages = producer.produce(files)
            yield from messages
            yield Error(message="test error message", source_media=files[-1])


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

def test_run_default(frame_model: FrameModel, test_videos: List[str]):
    sys.argv = [
        "prog",
        "--params",
        '{"fps":2, "allow_single_frame": false, "continue_on_error": true}',
        "--output-path",
        "custom.jsonl",
    ]

    with patch("common_ml.tagging.run_helpers.start_loop_from_frame_model") as mock:
        run_default(frame_model)
        args, kwargs = mock.call_args
        assert kwargs["fps"] == 2
        assert kwargs["allow_single_frame"] == False
        assert kwargs["continue_on_error"] == True

def test_loop_with_completion(frame_model: FrameModel, test_videos: List[str], test_images: List[str], test_folder: str):
    output_path = os.path.join(test_folder, "out.jsonl")

    producer = TagMessageProducer.from_model(frame_model)

    class AllAtOnceWrapper(TagMessageProducer):
        def __init__(self, producer: TagMessageProducer):
            self.files = []
            self.producer = producer

        def produce(self, files: List[str]) -> Iterator[Message]:
            self.files += files
            yield from ()

        def on_completion(self) -> Iterator[Message]:
            yield Tag(start_time=0, end_time=0, tag="on_completion started", source_media="who cares")
            yield from self.producer.produce(self.files)

    my_favorite_producer = AllAtOnceWrapper(producer)

    read_fd, write_fd = os.pipe()
    proc = multiprocessing.Process(target=_run_producer_loop, args=(my_favorite_producer, output_path, read_fd, write_fd, False, None))
    proc.start()
    os.close(read_fd)
    write_pipe = os.fdopen(write_fd, 'w')

    write_pipe.write("\n".join(test_videos) + "\n")
    write_pipe.flush()

    time.sleep(1)

    try:
        print("Output path:", output_path)
        with open(output_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 0

        write_pipe.write("\n".join(test_images) + "\n")
        write_pipe.flush()

        time.sleep(1)

        with open(output_path, "r") as f:
            lines = f.readlines()
        # still nothing
        assert len(lines) == 0

        # close to trigger eof and release tag messages
        write_pipe.close()

        time.sleep(1)

        with open(output_path, "r") as f:
            lines = f.readlines()

        # now after 
        assert len(lines) > 100
        # ensure that messages were generated from on_completion
        assert "on_completion started" in lines[0]

    finally:
        write_pipe.close()
        proc.join(timeout=5)

def test_loop_with_progress(frame_model: FrameModel, test_videos: List[str], test_images: List[str], test_folder: str):
    output_path = os.path.join(test_folder, "out.jsonl")

    producer = TagMessageProducer.from_model(frame_model)

    class PartialProgressWrapper(TagMessageProducer):
        def __init__(self, producer: TagMessageProducer):
            self.files = []
            self.producer = producer

        def produce(self, files: List[str]) -> Iterator[Message]:
            self.files += files
            yield from ()

        def on_completion(self) -> Iterator[Message]:
            for i, f in enumerate(self.files):
                yield ProgressRatio(progress=(i+1)/len(self.files))
            yield from self.producer.produce(self.files)

    my_favorite_producer = PartialProgressWrapper(producer)

    read_fd, write_fd = os.pipe()
    proc = multiprocessing.Process(target=_run_producer_loop, args=(my_favorite_producer, output_path, read_fd, write_fd, False, None))
    proc.start()
    os.close(read_fd)
    write_pipe = os.fdopen(write_fd, 'w')

    write_pipe.write("\n".join(test_videos) + "\n")
    write_pipe.flush()

    time.sleep(1)

    try:
        with open(output_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 0

        write_pipe.write("\n".join(test_images) + "\n")
        write_pipe.flush()

        write_pipe.close()

        time.sleep(1)

        with open(output_path, "r") as f:
            lines = f.readlines()

        # now after 
        assert len(lines) > 100

        progress_messages = [json.loads(l) for l in lines if "progress_ratio" in l]
        assert len(progress_messages) == len(test_videos) + len(test_images)
        assert 0 < progress_messages[0]["data"]["progress"] < 1
        assert progress_messages[-1]["data"]["progress"] == 1.0

    finally:
        write_pipe.close()
        proc.join(timeout=5)
