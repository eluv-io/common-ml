
import pytest
import os

from common_ml.video_processing import get_frames

TEST_DATA = os.path.join(os.path.dirname(__file__), "test-data")

def test_get_frames():
    video_path = os.path.join(TEST_DATA, "1.mp4")
    frames, _, _ = get_frames(video_path, fps=1)
    
    assert len(frames) > 0