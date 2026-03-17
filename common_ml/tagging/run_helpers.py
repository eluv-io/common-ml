
import argparse
from typing import Union
import json
from queue import Queue
from dataclasses import asdict
import threading
import time
import sys

from common_ml.tagging.producer import TagMessageProducer
from common_ml.tagging.models.frame_based import FrameModel, BatchFrameModel
from common_ml.tagging.models.av import AVModel
from common_ml.tagging.file_tagger import *
from common_ml.tagging.producer import *
from common_ml.tagging.messages import *

class AbortTaggingException(Exception):
    pass

def write_message(msg: Message, fout):
    if isinstance(msg, TagMessage):
        fout.write(json.dumps({"type": msg.type, "data": asdict(msg.data)}) + "\n")
    elif isinstance(msg, ProgressMessage):
        fout.write(json.dumps({"type": msg.type, "data": asdict(msg.data)}) + "\n")
    elif isinstance(msg, ErrorMessage):
        fout.write(json.dumps({"type": msg.type, "data": asdict(msg.data)}) + "\n")
    else:
        raise ValueError(f"Unnexpected message type: {msg}")
    fout.flush()

def start_loop_from_av_model(
    model: AVModel, 
    output_path: str,
    continue_on_error: bool=False,
    batch_timeout: float=0.2,
) -> None:
    producer = TagMessageProducer.from_model(model, continue_on_error=continue_on_error)
    start_loop_from_producer(
        producer=producer,
        output_path=output_path,
        continue_on_error=continue_on_error,
        batch_timeout=batch_timeout,
    )

def start_loop_from_frame_model(
    model: Union[FrameModel, BatchFrameModel],
    output_path: str,
    continue_on_error: bool=False,
    batch_timeout: float=0.2,
    fps: float=1,
    allow_single_frame: bool=True,
) -> None:
    producer = TagMessageProducer.from_model(model, fps=fps, allow_single_frame=allow_single_frame, continue_on_error=continue_on_error)
    start_loop_from_producer(
        producer=producer,
        output_path=output_path,
        continue_on_error=continue_on_error,
        batch_timeout=batch_timeout,
    )

def start_loop_from_producer(
    producer: TagMessageProducer,
    output_path: str,
    continue_on_error: bool=False,
    batch_timeout: float=0.2
) -> None:
    """
    Live mode: reads file paths from stdin and processes them in batches
    
    Args:
        model: The model to use for tagging, can be AVModel, FrameModel, or BatchFrameModel
        output_path: The file path to write the output tags (.jsonl format)
        batch_timeout: Timeout for batching files
        fps: Frames per second, only relevant or FrameModel or BatchFrameModel when processing videos
        allow_single_frame: Whether to allow processing of single-frame videos, only relevant for FrameModel or BatchFrameModel
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
            print("Stopping stdin reader", file=sys.stderr)
            file_queue.put(None)
    
    def process_batch(files, fd):
        print(f"Processing batch of {len(files)} files...", file=sys.stderr)
        try:
            messages = producer.produce(files)
            for msg in messages:
                write_message(msg, fd)
                if isinstance(msg, ErrorMessage):
                    raise AbortTaggingException("Received an error response from the producer")
        except AbortTaggingException:
            if not continue_on_error:
                # we already wrote the error
                raise
        except Exception as e:
            write_message(ErrorMessage(type="error", data=Error(message=str(e))), fd)
            if not continue_on_error:
                raise
        print(f"Completed batch of {len(files)} files", file=sys.stderr)
    
    reader_thread = threading.Thread(target=stdin_reader, daemon=True)
    reader_thread.start()
    
    current_batch = []

    fdout = open(output_path, 'a')
    
    while True:
        try:
            while not file_queue.empty():
                try:
                    file_path = file_queue.get_nowait()
                    
                    # last batch
                    if file_path is None:
                        if current_batch:
                            process_batch(current_batch, fdout)
                        fdout.close()
                        return
                    
                    # add to current batch to process
                    current_batch.append(file_path)
                except:
                    break
            
            if current_batch:
                process_batch(current_batch, fdout)
                current_batch = []
            
            if not reader_thread.is_alive() and file_queue.empty():
                break
            
            time.sleep(batch_timeout)
        except (KeyboardInterrupt, SystemExit):
            break

    fdout.close()

def run_default(
    model: Union[AVModel, FrameModel, BatchFrameModel],
    continue_on_error: bool=False,
    batch_timeout: float=0.2,
    fps: float=1,
    allow_single_frame: bool=True,
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', required=True, help='Path to write output tags (.jsonl)')
    args = parser.parse_args()

    if isinstance(model, AVModel):
        start_loop_from_av_model(model, output_path=args.output_path, continue_on_error=continue_on_error, batch_timeout=batch_timeout)
    else:
        start_loop_from_frame_model(model, output_path=args.output_path, continue_on_error=continue_on_error, batch_timeout=batch_timeout, fps=fps, allow_single_frame=allow_single_frame)