from typing import List, Tuple, Dict
from copy import deepcopy
from .tags import *

# Args:
#   intervals: a list of (start, end) intervals
#   label_to_tags: a dictionary mapping a label to a list of tags
#
# Returns:
#   A list of aggregated tags for each interval
def aggregate(intervals: List[Tuple[int, int]], label_to_tags: Dict[str, List[VideoTag]]) -> List[AggTag]:
    for label, tags in label_to_tags.items():
        label_to_tags[label] = sorted(tags, key=lambda x: x.start_time)

    # merged tags into their appropriate intervals
    result = []
    for left, right in intervals:
        agg_tags = AggTagSchema().unmarshal({"start_time": left, "end_time": right, "tags": {}})
        for label, tags in label_to_tags.items():
            for tag in tags:
                if tag.start_time >= left and tag.start_time < right:
                    #print('here')
                    if label not in agg_tags.tags:
                        agg_tags.tags[label] = []
                    agg_tags.tags[label].append(tag)    
        result.append(agg_tags)

    for i in range(len(result)): 
        agg_tag = result[i]
        if "speech_to_text" in agg_tag.tags:
            agg_tag.tags["speech_to_text"] = _coalesce(agg_tag.tags["speech_to_text"])
    
    return result

# Args:
#   tags: a list of tags
#
# Returns:
#   A list of intervals based on start/end of each tag... to be used for aggregation (see above)
def get_tag_intervals(tags: List[VideoTag]) -> List[Tuple[int, int]]:
    intervals = []
    for tag in tags:
        intervals.append((tag.start_time, tag.end_time))
    return intervals

# Args:
#   tags: A list of tags which include text NOTE: these are probably STT tags. 
# TODO: I think text should be made non-optional in video-tag and just put "" if not present
#
# Returns:
#  A list of intervals representing the start/end times of each sentence
def get_sentence_intervals(tags: List[VideoTag]) -> List[Tuple[int, int]]:
    sentence_delimiters = ['.', '?', '!']
    intervals = []
    if len(tags) == 0:
        return []
    quiet = True
    curr_int = [0]
    for i, tag in enumerate(tags):
        assert tag.text is not None 
        if quiet and tag.start_time > curr_int[0]:
            # commit the silent interval
            curr_int.append(tag.start_time)
            intervals.append((curr_int[0], curr_int[-1]))
            curr_int.clear()
            # start a new speaking interval
            curr_int.append(tag.start_time)
            quiet = False
        if tag.text[-1] in sentence_delimiters or i == len(tags)-1:
            # end and commit the speaking interval
            curr_int.append(tag.end_time)
            intervals.append((curr_int[0], curr_int[-1]))
            curr_int.clear()
            # start a new silent interval
            curr_int.append(tag.end_time)
            quiet = True
    return intervals

# Args:
#   tags: a list of tags
#
# Returns:
#   A single tag result with the texts merged together and the start/end times of the first/last tag respectively
# 
# NOTE: this function is only used for the ASR service, when we need this to happen for other services we should make it more general
# TODO: remove this function 
def _coalesce(tags: List[VideoTag]) -> List[VideoTag]:
    if len(tags) <= 1:
        return tags
    result = []
    curr_sentence = []
    for tag in tags:
        if len(curr_sentence) == 0:
            start_time = tag.start_time
        curr_sentence.append(tag.text)
        if tag.text[-1] in ['.', '?', '!']:
            result.append(VideoTagSchema().unmarshal({"start_time": start_time, "end_time": tag.end_time, "text": " ".join(curr_sentence)}))
            curr_sentence.clear()
    if len(curr_sentence) > 0:
        result.append(VideoTagSchema().unmarshal({"start_time": start_time, "end_time": tag.end_time, "text": " ".join(curr_sentence)}))

    return result

def _do_coalesce(tag: VideoTag) -> bool:
    return tag.coalesce 
    
# Merges a list of frame tags into a single larger frame tag with the 
# frame indices based on the offset and video lengths
# 
# Args:
#   frame_tags: a list of frame tags from a single source
#   video_lengths: a list of video lengths in milliseconds
#   fps: frames per second of the video
#   offset: offset in milliseconds
#
# Returns:
#   A frame tag combining all the input frame_tags. Correcting for the provided offset.
def merge_frame_tags(frame_tags: List[FrameTags], video_lengths: List[int], fps: float, offset: int) -> FrameTags:
    res = {}
    frame_offset = round((offset/1000)*fps) # convert to frames, TODO: could rounding be a problem here?
    for vid_length, ftag in zip(video_lengths, frame_tags):
        for frame_idx, tags in ftag.items():
            res[frame_idx + frame_offset] = deepcopy(tags)
 
        vid_length = round((vid_length/1000)*fps) # convert to frames
        frame_offset += vid_length

    return res