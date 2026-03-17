
import argparse
from typing import List, Union
import os
from queue import Queue
import threading
import time
import sys

from common_ml.tagging.abstract import MessageProducer
from common_ml.tagging.models.abstract import *
from common_ml.tagging.file_tagger_adapt import *
from common_ml.tagging.producer_adapt import *
from common_ml.tagging.messages import *

class AbortTaggingException(Exception):
    pass

def write_message(msg: Message, fout):
    if isinstance(msg, TagMessage):
        fout.writeline({"type": "tag", "data": msg.data})
    elif isinstance(msg, ProgressMessage):
        fout.writeline({"type": "progress", "data": msg.data})
    elif isinstance(msg, ErrorMessage):
        fout.writeline({"type": "error", "data": msg.data})

def start_tag_loop(
    model: Union[VideoModel, FrameModel, BatchFrameModel],
    output_path: str,
    continue_on_error: bool=False,
    batch_timeout: float=0.2,
    fps: float=1,
    allow_single_frame: bool=True,
) -> None:
    """
    Live mode: reads file paths from stdin and processes them in batches
    
    Args:
        model: The model to use for tagging, can be VideoModel, FrameModel, or BatchFrameModel
        output_path: The file path to write the output tags (.jsonl format)
        batch_timeout: Timeout for batching files
        fps: Frames per second, only relevant or FrameModel or BatchFrameModel when processing videos
        allow_single_frame: Whether to allow processing of single-frame videos, only relevant for FrameModel or BatchFrameModel
    """

    if isinstance(model, VideoModel):
        file_tagger = get_file_tagger_from_video_model(model)
    elif isinstance(model, (FrameModel, BatchFrameModel)):
        file_tagger = get_file_tagger_from_frame_model(model, fps, allow_single_frame)
    else:
        raise ValueError("Model must be either VideoModel, FrameModel, or BatchFrameModel")

    producer = get_message_producer_from_file_tagger(file_tagger)
    
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
            messages = producer.produce_messages(files)
            for msg in messages:
                write_message(msg, fd)
                if isinstance(msg, ErrorMessage) and not continue_on_error:
                    raise AbortTaggingException("Received an error response from the producer")
        except AbortTaggingException:
            raise
        except Exception as e:
            write_message(ErrorMessage(type="error", data=Error(message=str(e))), fd)
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

def serve_model(
    model: Union[VideoModel, FrameModel, BatchFrameModel],
    continue_on_error: bool=False,
    batch_timeout: float=0.2,
    fps: float=1,
    allow_single_frame: bool=True,
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', required=True, help='Path to write output tags (.jsonl)')
    args = parser.parse_args()

    start_tag_loop(model, output_path=args.output_path, continue_on_error=continue_on_error, batch_timeout=batch_timeout, fps=fps, allow_single_frame=allow_single_frame)