import argparse
import os
from quick_test_py import Tester
from common_ml.video_processing import get_fps, get_key_frames

test_file = os.path.join(os.path.dirname(__file__), 'test.mp4')

def test_get_fps():
    return [lambda: get_fps(test_file)]

def test_get_key_frames():
    def tc1():
        frames, f_pos, timestamps = get_key_frames(test_file)
        # converting to list for comparison
        frames = [f.tolist() for f in frames]
        return frames, f_pos, timestamps
    return [tc1]

def main():
    tester = Tester(os.path.join(os.path.dirname(__file__), 'test_data'))
    tester.register('test_fps', test_get_fps())
    tester.register('test_key_frames', test_get_key_frames())
    if args.record and args.log:
        raise ValueError("Cannot record and log at the same time")
    if args.record:
        tester.record()
    elif args.log:
        tester.log()
    else:
        tester.validate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()
    main()