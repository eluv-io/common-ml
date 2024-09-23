
import numpy as np
from typing import Tuple, List
from fractions import Fraction
import subprocess
import json
from loguru import logger

def get_fps(video_file: str) -> float:
    cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v", "-show_frames",
            "-show_entries", "stream=r_frame_rate,avg_frame_rate",
            "-print_format", "json", video_file]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise Exception(e.output.decode("utf-8"))
    
    output = json.loads(output)
    
    if len(output["streams"]) > 1:
        logger.warning(f"Found multiple streams in {video_file}... using the first one")

    stream_info = output["streams"][0]

    fps = float(Fraction(stream_info["avg_frame_rate"]))
    if stream_info["r_frame_rate"] != stream_info["avg_frame_rate"]:
        logger.error(f"{video_file} has variable frame rate, defaulting to average fps.")

    return fps

# input can be either downloadUrl or filename
def get_key_frames(video_file: str) -> Tuple[np.ndarray, List[int], List[float]]:
    cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v", "-show_frames",
            "-show_entries", "frame=width,height,pict_type,pkt_pts_time",
            "-print_format", "json", video_file]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise Exception(e.output.decode("utf-8"))
    except FileNotFoundError as e:
        raise FileNotFoundError("ffprobe not found in PATH. Make sure ffmpeg is installed.")
    
    output = json.loads(output)
    w, h = output["frames"][0]["width"], output["frames"][0]["height"]

    timestamps = [float(f["pkt_pts_time"]) for f in output["frames"] if f["pict_type"] == 'I']
    f_pos = [i for i, f in enumerate(output["frames"]) if f["pict_type"] == 'I']

    cmd = ["ffmpeg", "-nostdin", "-i", video_file,
            "-vf", "select='eq(pict_type,I)'", "-vsync", "2",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "pipe:"]

    process = subprocess.Popen(cmd, stderr=-1, stdout=-1)
    out, err = process.communicate()
    retcode = process.poll()
    if retcode:
        raise Exception(f"ffmpeg error: {err.decode('utf-8')}")

    frames = np.frombuffer(out, np.uint8)
    frames = frames.reshape((-1, h, w, 3))

    assert len(timestamps) == frames.shape[0], "Key frames returned and key frames extracted from file metadata do not match"

    sorted_frames = sorted(((frame, pos, ts) for frame, pos, ts in zip(frames, f_pos, timestamps)), key=lambda x: x[1])
    frames, f_pos, timestamps = zip(*sorted_frames)
    return frames, f_pos, timestamps