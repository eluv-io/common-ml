from typing import List, Tuple
import os
import cv2
import random
import numpy as np
import tempfile
import subprocess
from loguru import logger

# Get video frames from a video path
# 
# Args:
#  video_path: path to the video
#  sample_rate: additional frames to sample in between key frames
#        e.g.
#        key_frames sampling with 'sample_rate' frames sampled in between
#        e.g.
#            key frames:{0, 23}, sample_rate:1 --> frames sampled: {0, 12, 24}
#            param: video_path, sample_rate
#            ret: f_num
#
# Returns:
#  1. key frame indices
#  2. key frames as list of numpy arrays
# TODO: implement caching
def get_video_frames(video_path: str, sample_rate: int=0) -> Tuple[List[int], List[np.ndarray]]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(video_path)

    logger.info('Getting i frames')
    tmp_path = _get_i_frames(video_path)

    def _get_fnum(sample_rate, fps_min=4, fps_max=8, mezz_dur=30.03):
        with open(tmp_path, 'r') as f:
            f_num = [int(n.strip())-1 for n in f.readlines()]
        n = len(f_num)
        if n <= 1:
            return set(f_num)
        if n < mezz_dur*fps_min:
            sample_rate = int((mezz_dur*fps_min-n)//(n-1) + 1 + sample_rate)
            logger.info(f"Sampling sample_rateuency modified to {sample_rate}")
        else:
            random.seed(1)
            f_num = sorted(random.sample(f_num, min(n, int(mezz_dur*fps_max))))
        tmp = []
        for i in range(1, len(f_num)):
            tmp.extend([
                f_num[i-1]+int((f_num[i]-f_num[i-1])*(j+1)/(sample_rate+1))
                for j in range(sample_rate)
            ])
        return set(tmp).union(set(f_num))

    f_num = _get_fnum(sample_rate)
    os.remove(tmp_path)

    if len(f_num) == 0:
        return [], []

    n_frame = 0
    images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            '''uniform frame sampling, e.g. 0,2,4,6,etc.
            if n_frame%sample_rate==0:
                images.append(frame)
            '''
            if n_frame in f_num:
                images.append(frame)
        else:
            break
        n_frame += 1

    cap.release()
    frame_num = len(images)

    assert frame_num == len(f_num)
    logger.info("Total # of frames %s" % frame_num)

    return f_num, images
    
def _get_i_frames(video_path: str) -> str:
    tmp_path = tempfile.mktemp(prefix=f'{os.path.basename(video_path)}_iframe')

     # Build the command as a list
    command = [
        'ffprobe',
        '-select_streams', 'v',
        '-show_frames',
        '-show_entries', 'frame=pict_type',
        '-of', 'csv',
        video_path
    ]

    # Execute the command with a pipeline, grep only 'I' frames and get the line numbers
    # Redirect stdout to a file and suppress stderr
    with open(tmp_path, 'w') as output_file:
        p1 = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p2 = subprocess.Popen(['grep', 'frame'], stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p3 = subprocess.Popen(['grep', '-n', 'I'], stdin=p2.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p4 = subprocess.Popen(['cut', '-d', ':', '-f', '1'], stdin=p3.stdout, stdout=output_file, stderr=subprocess.DEVNULL)

    # Wait for the process to complete
    p4.communicate()

    return tmp_path