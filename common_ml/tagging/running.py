
import argparse
from typing import List, Union
import os
from queue import Queue
import threading
import time
import sys
from loguru import logger

from common_ml.tagging.abstract import MessageProducer
from common_ml.tagging.model_types import *
from common_ml.tagging.tag_types import *
from common_ml.tagging.messages import *
from common_ml.tagging.conversion import *

def get_message_producer_from_file_tagger(file_tagger: FileTagger, continue_on_error: bool = False) -> MessageProducer:
    
    class NewMessageProducer(MessageProducer):
        def produce_messages(self, files: List[str]) -> List[Message]:
            res = []

            for fname in files:
                try:
                    file_tagger.tag(fname)
                except Exception as e:
                    logger.opt(exception=e).error(f"Error processing file {fname}")
                    res.append(ErrorMessage(type='error', data=Error(message=str(e), source_media=fname)))
                    if not continue_on_error:
                        return res
                # finished fname, add progress
                res.append(ProgressMessage(type='progress', data=Progress(source_media=fname)))

            return res

    return NewMessageProducer()

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
        file_tagger = get_file_tagger_from_frame_model(model, fps)
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
            messages = producer.produce_messages(valid_files)
            for msg in messages:
                write_message(msg, sys.stdout)
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

def serve_model(
    model: Union[VideoModel, FrameModel, BatchFrameModel],
    batch_timeout: float=0.2,
    fps: float=1,
    allow_single_frame: bool=True,
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', required=True, help='Path to write output tags (.jsonl)')
    args = parser.parse_args()

    start_tag_loop(model, output_path=args.output_path, batch_timeout=batch_timeout, fps=fps, allow_single_frame=allow_single_frame)