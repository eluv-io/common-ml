import argparse
import os
from loguru import logger
from quick_test_py import Tester
from common_ml.video_processing import get_fps, get_key_frames, unfrag_video, _run_command

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

def test_unfrag_video():
    def tc1():
        filedir = os.path.dirname(__file__)
        if os.path.exists(os.path.join(filedir, 'unfrag.mp4')):
            os.remove(os.path.join(filedir, 'unfrag.mp4'))
        unfrag_video(os.path.join(filedir, 'fragmented.mp4'), os.path.join(filedir, 'unfrag.mp4'))
        probe_cmd = f"ffprobe -v quiet -show_format -print_format json {os.path.join(filedir, 'unfrag.mp4')}"
        output = _run_command(probe_cmd)
        logger.debug(output)
        return os.path.exists(os.path.join(filedir, 'unfrag.mp4')) and os.path.getsize(os.path.join(filedir, 'unfrag.mp4')) > 1e6
    return [tc1]

def main():
    tester = Tester(os.path.join(os.path.dirname(__file__), 'test_data'))
    tester.register('test_fps', test_get_fps())
    tester.register('test_key_frames', test_get_key_frames())
    tester.register('test_unfrag_video', test_unfrag_video())
    if args.record:
        tester.record(args.tests)
    else:
        tester.validate(args.tests)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--tests', nargs='+')
    args = parser.parse_args()
    main()