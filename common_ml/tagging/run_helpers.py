
import argparse
import traceback
from typing import Union, Any, Dict
import json
from queue import Queue
from dataclasses import asdict
import threading
import time
import sys

from loguru import logger

from common_ml.tagging.producer import TagMessageProducer
from common_ml.tagging.models.frame_based import FrameModel, BatchFrameModel
from common_ml.tagging.models.av import AVModel
from common_ml.tagging.file_tagger import *
from common_ml.tagging.producer import *
from common_ml.tagging.messages import *

def run_default(
    model: Union[
        AVModel, 
        FrameModel,
        BatchFrameModel,
        TagMessageProducer
    ],
    batch_timeout: float=0.2,
    batch_limit: Optional[int]=None,
):
    """
    This is the default entry point for running a tagging model. It supports four different interfaces: AVModel, FrameModel, BatchFrameModel, and TagMessageProducer. 
    
    This function will run indefinitely as a tagging daemon: receiving input files over stdin and outputting to a .jsonl file for the Eluvio Tagging runtime to process.

    Args:
        model: The tagging model to run. Can be an instance of AVModel, FrameModel, BatchFrameModel, or TagMessageProducer.
        batch_timeout: Time in seconds to wait before processing a batch of files.
        batch_limit: Maximum number of files to process in a single batch.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', required=True, help='Path to write output tags (.jsonl)')
    parser.add_argument('--params', required=False, help='Runtime parameters as JSON string, e.g. \'{"foo": "bar"}\'')
    args, _ = parser.parse_known_args()

    params = {}
    if args.params:
        params = json.loads(args.params)

    # support the following params by default

    continue_on_error = params.get("continue_on_error", False)
    ## for frame models only
    fps = params.get("fps", 1) # rate at which to tag the source media in the case of video
    allow_single_frame = params.get("allow_single_frame", True) # configure whether two consecutive identical frames must exist to generate a tag
    

    if isinstance(model, TagMessageProducer):
        start_loop_from_producer(model, output_path=args.output_path, continue_on_error=continue_on_error, batch_timeout=batch_timeout, batch_limit=batch_limit)
    elif isinstance(model, AVModel):
        start_loop_from_av_model(model, output_path=args.output_path, continue_on_error=continue_on_error, batch_timeout=batch_timeout, batch_limit=batch_limit)
    elif isinstance(model, (FrameModel, BatchFrameModel)):
        start_loop_from_frame_model(model, output_path=args.output_path, continue_on_error=continue_on_error, batch_timeout=batch_timeout, fps=fps, allow_single_frame=allow_single_frame, batch_limit=batch_limit)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

def catch_errors():
    """
    Call this function once at the beginning of your programs so that any raised exception will be recorded for the tagger runtime.

    This is important so that unexpected errors in your program will be propagated to the tagger and a reasonable error message can be
    delivered to the caller.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', required=True, help='Path to write output tags (.jsonl)')
    args, _ = parser.parse_known_args()
    output_path = args.output_path
    def handler(exc_type, exc_value, exc_tb):
        print("Caught unhandled exception:")
        traceback.print_exception(exc_type, exc_value, exc_tb)
        with open(output_path, 'a') as fout:
            write_message(Error(message=f"{exc_type.__name__}: {exc_value}"), fout)
    sys.excepthook = handler

def get_params() -> Dict[str, Any]:
    """
    Receive a dictionary containing user submitted params - injected via the Tagger runtime. As an implementer of a tagger model, you can define
    any json structured parameters that you want to expose to end users.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, required=False, help='Runtime parameters as JSON')
    args, _ = parser.parse_known_args()
    params_str = args.params
    if not params_str:
        return {}
    config_dict = json.loads(params_str)
    return config_dict

class AbortTaggingException(Exception):
    pass

def write_message(msg: Message, fout):
    if isinstance(msg, Tag):
        fout.write(json.dumps({"type": "tag", "data": asdict(msg)}) + "\n")
    elif isinstance(msg, Progress):
        fout.write(json.dumps({"type": "progress", "data": asdict(msg)}) + "\n")
    elif isinstance(msg, Error):
        fout.write(json.dumps({"type": "error", "data": asdict(msg)}) + "\n")
    else:
        raise ValueError(f"Unnexpected message type: {msg}")
    fout.flush()

def start_loop_from_av_model(
    model: AVModel, 
    output_path: str,
    continue_on_error: bool=False,
    batch_timeout: float=0.2,
    batch_limit: Optional[int]=None,
) -> None:
    producer = TagMessageProducer.from_model(model)
    start_loop_from_producer(
        producer=producer,
        output_path=output_path,
        continue_on_error=continue_on_error,
        batch_timeout=batch_timeout,
        batch_limit=batch_limit,
    )

def start_loop_from_frame_model(
    model: Union[FrameModel, BatchFrameModel],
    output_path: str,
    continue_on_error: bool=False,
    batch_timeout: float=0.2,
    fps: float=1,
    allow_single_frame: bool=True,
    batch_limit: Optional[int]=None,
) -> None:
    producer = TagMessageProducer.from_model(model, fps=fps, allow_single_frame=allow_single_frame)
    start_loop_from_producer(
        producer=producer,
        output_path=output_path,
        continue_on_error=continue_on_error,
        batch_timeout=batch_timeout,
        batch_limit=batch_limit,
    )

def start_loop_from_producer(
    producer: TagMessageProducer,
    output_path: str,
    continue_on_error: bool=False,
    batch_timeout: float=0.2,
    batch_limit: Optional[int]=None,
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
        for fname in files:
            print(f"Got {fname}")
        try:
            messages = producer.produce(files)
            for msg in messages:
                write_message(msg, fd)
                if isinstance(msg, Error):
                    raise AbortTaggingException("Received an error response from the producer")
        except AbortTaggingException:
            if not continue_on_error:
                # we already wrote the error
                raise
        except Exception as e:
            write_message(Error(message=str(e)), fd)
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
                file_path = file_queue.get_nowait()
                
                # last batch
                if file_path is None:
                    if current_batch:
                        process_batch(current_batch, fdout)
                    fdout.close()
                    return
                
                # add to current batch to process
                current_batch.append(file_path)
                if batch_limit is not None and len(current_batch) >= batch_limit:
                    process_batch(current_batch, fdout)
                    current_batch = []
            
            if current_batch:
                process_batch(current_batch, fdout)
                current_batch = []
            
            if not reader_thread.is_alive() and file_queue.empty():
                break
            
            time.sleep(batch_timeout)
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            logger.opt(exception=e).error("Error in main loop")
            fdout.close()
            return

    fdout.close()