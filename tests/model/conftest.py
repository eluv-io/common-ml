import pytest

from common_ml.tags import Tag, FrameInfo

@pytest.fixture
def make_tag():
    def _make_tag(text: str, frame_idx: int, source_media: str = "test.mp4", track: str = "default") -> Tag:
        return Tag(
            text=text,
            start_time=0,
            end_time=0,
            source_media=source_media,
            track=track,
            frame_info=FrameInfo(frame_idx=frame_idx, box={}, confidence=None),
        )
    return _make_tag