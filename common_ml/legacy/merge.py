from typing import List
from loguru import logger

from .tags import FrameTags, VideoTag

# Merges two frame tags dictionaries, deferring to the more recent tags in case of overlap
def merge_frame_tags(old_tags: FrameTags, new_tags: FrameTags) -> FrameTags:
    old_tags, new_tags = old_tags.copy(), new_tags.copy()

    result = {}
    for frame in new_tags.keys():
        result[frame] = new_tags[frame]

    overlap = False
    for frame in old_tags.keys():
        if frame in result and len(result[frame].tags) > 0:
            overlap = True
        else:
            result[frame] = old_tags[frame]
    
    if overlap:
        logger.warning(f"Overlapping frame tags, deferring to the more recent ones")
            
    return result

# Merges two video tag lists, deferring to the more recent tags in case of overlap
def merge_video_tags(old_tags: List[VideoTag], new_tags: List[VideoTag]) -> List[VideoTag]:
    old_tags, new_tags = old_tags[:], new_tags[:]

    if len(old_tags) == 0:
        return new_tags
    elif len(new_tags) == 0:
        return old_tags

    old_tags = sorted(old_tags, key=lambda x: x.start_time)
    new_tags = sorted(new_tags, key=lambda x: x.start_time)

    result = []
    overlap = False
    i, j = 0, 0
    while i < len(old_tags) and j < len(new_tags):
        if _overlaps(old_tags[i], new_tags[j]):
            overlap = True
            result.append(new_tags[j])
            i, j = i+1, j+1
        elif old_tags[i].start_time < new_tags[j].start_time:
            result.append(old_tags[i])
            i += 1
        else:
            result.append(new_tags[j])
            j += 1
            
    if overlap:
        logger.warning(f"Overlapping video tags, deferring to the more recent ones")

    # add remaining tags
    for i in range(i, len(old_tags)):
        result.append(old_tags[i])

    for j in range(j, len(new_tags)):
        result.append(new_tags[j])

    return result

# Returns true if tag1 and tag2 overlap
def _overlaps(tag1: VideoTag, tag2: VideoTag) -> bool:
    if not _valid(tag1) or not _valid(tag2):
        raise ValueError(f"Invalid tag, start time must be less than end time")
    
    if tag1.start_time > tag2.end_time:
        return False
    elif tag2.start_time > tag1.end_time:
        return False
    
    return True

def _valid(tag: VideoTag) -> bool:
    return tag.end_time >= tag.start_time