import unittest
from common_ml.video_processing import get_fps, get_key_frames

class TestVideo(unittest.TestCase):
    def test_get_fps(self):
        self.assertEqual(get_fps('test.mp4'), 23.9736290080911)

    def test_get_frames(self):
        print('Getting key frames...')
        print(get_key_frames('test.mp4'))

if __name__ == '__main__':
    unittest.main()