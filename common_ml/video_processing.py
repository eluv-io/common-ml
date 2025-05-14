
import numpy as np
from typing import Tuple, List
from fractions import Fraction
import subprocess
import json
import os
from loguru import logger
import re
import cv2
import random

def get_fps(video_file: str) -> float:
    cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v",
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

def get_key_frames(video_path: str) -> Tuple[List[np.ndarray], List[int], List[float]]:
    freq = 0
    video_part_size = 30.03
    logger.info(f'[frame] get_video_frames video part is {video_part_size}')

    if not os.path.exists(video_path):
        logger.info(f'Video file not exists, check path {video_path}')
        raise SystemExit

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info('Getting i frames')
    tmp_path = f'{os.path.basename(video_path)}_iframe.txt'
    os.system(
        f'ffprobe -hide_banner -select_streams v -show_frames -show_entries frame=pict_type '
        f'-of csv "{video_path}" 2>> ffprobe.stderr.log | grep frame | grep -n I | '
        f'cut -d : -f 1 > {tmp_path}'
    )

    def _get_fnum(freq, fps_min=4, fps_max=8, mezz_dur=video_part_size):
        with open(tmp_path, 'r') as f:
            f_num = [int(n.strip()) - 1 for n in f.readlines()]
        n = len(f_num)
        if n < mezz_dur * fps_min:
            freq = int((mezz_dur * fps_min - n) // (n - 1) + 1 + freq)
            logger.info(f"Sampling frequency modified to {freq}")
        else:
            random.seed(1)
            f_num = sorted(random.sample(f_num, min(n, int(mezz_dur * fps_max))))
        tmp = []
        for i in range(1, len(f_num)):
            tmp.extend([
                f_num[i - 1] + int((f_num[i] - f_num[i - 1]) * (j + 1) / (freq + 1))
                for j in range(freq)
            ])
        return sorted(set(tmp).union(set(f_num)))

    f_num = _get_fnum(freq)
    logger.info(f'Frame IDs to tag: {f_num}')
    os.remove(tmp_path)

    images = []
    timestamps = []
    n_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if n_frame in f_num:
            images.append(frame)
            timestamps.append(n_frame / fps)
        if n_frame % 1000 == 0:
            logger.info(f'Capturing frame # {n_frame}')
        n_frame += 1

    cap.release()
    assert len(images) == len(f_num)
    logger.info(f"Total # of frames {len(images)}")
    return images, f_num, timestamps

# video_file: path to video file
# fps: frames per second to sample
def get_frames(video_file: str, fps: int) -> Tuple[np.ndarray, List[int], List[float]]:
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {video_file}"
    try:
        output = subprocess.check_output(cmd.split(' '), stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise Exception(e.output.decode("utf-8"))
    except FileNotFoundError as e:
        raise FileNotFoundError("ffprobe not found in PATH. Make sure ffmpeg is installed.")
    
    try:
        output = output.decode("utf-8")
        w, h = str(output).strip().split(',')
        w, h = int(w), int(h)
    except ValueError as e:
        raise ValueError(f"Could not extract width and height from ffprobe output: {output}")

    # get frames 
    cmd = [
        "ffmpeg",
        "-i", video_file,        
        "-vf", f"fps={fps}, showinfo",        
        "-f", "image2pipe",       
         "-f", "rawvideo", 
         "-pix_fmt", "rgb24",        
        "pipe:1"                
    ]

    process = subprocess.Popen(cmd, stderr=-1, stdout=-1)
    out, err = process.communicate()
    retcode = process.poll()
    if retcode:
        raise Exception(f"ffmpeg error: {err.decode('utf-8')}")
    
    video_fps = get_fps(video_file)
    
    timestamps, frame_idx = [], []
    stderr_output = err.decode("utf-8")
    for line in stderr_output.splitlines():
        if "showinfo" in line and "n:" in line:  # Lines containing frame info
            ts = float(re.search(r"pts_time:([0-9.]+)", line).group(1))
            ts += 1 / (2 * fps)  
            fidx = video_fps * ts
            timestamps.append(ts)
            frame_idx.append(round(fidx))
    
    frames = np.frombuffer(out, np.uint8)
    frames = frames.reshape((-1, h, w, 3))

    return frames, frame_idx, timestamps

def unfrag_video(video_file: str, output_file: str):
    cmd = f"ffmpeg -y -i {video_file} -c copy {output_file}"
    out, err = _run_command(cmd)
    file_size = os.path.getsize(output_file) / 1024**2
    if file_size < 0.01:
        raise RuntimeError(f"Failed to unfrag video file {video_file}:\n file size={file_size}MiB\n cmd={cmd},\n stdout={out}\n stderr={err}") 

# Run a command and return it's stdout
# Throws error if command fails
def _run_command(cmd: str) -> Tuple[str, str]:
    logger.debug(f"Running command\n{cmd}\n")
    res = subprocess.run(cmd.split(), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to run command\n{cmd}\nstderr=...\n{res.stderr}\nstdout=...\n {res.stdout}")
    return res.stdout, res.stderr