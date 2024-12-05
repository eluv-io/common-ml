
"""
The functions in this file are used to convert the tag structure used in this project (see in src/common/tags.py) with 
the format usual format of tags stored in the fabric objects (video-tag-tracks-XXXX.json and video-tags-overlay-XXXX.json)

This format is used to store the tags in the fabric objects, and is used to display the tags in the video player.
"""

from typing import Any, Dict, List, Optional
from collections import defaultdict

from .tags import FrameTagSchema, FrameTags, VideoTagSchema, VideoTag, AggTag
from .utils import track_to_label, label_to_track

# This function is responsible for converting from  
#
# Args:
#   overlay_tags: a dictionary mapping (feature name) -> frame tag
#   interval: formatted overlay tags are broken into buckets of this size (in minutes). If None, no bucketing is done.
#  
# Returns:
#    An overlay tag following the usual format for the video-tags-overlay files. Each element in the output list corresponds to one of the "video-tags-overlay-XXXX.json" files
def format_overlay(frame_tags: Dict[str, FrameTags], interval: Optional[int]=None) -> List[Dict[str, object]]:
    if len(frame_tags) == 0:
        return []
    buckets = {}
    interval = interval*1000*60 if interval is not None else None
    for feature, ftags in frame_tags.items():
        label = feature.replace("_", " ").title()
        for frame_idx, tag in ftags.items():
            assert tag.timestamp_sec is not None, "timestamp_sec is required for overlay tags"
            bucket_idx = int(tag.timestamp_sec/interval) if interval is not None else 0
            if bucket_idx not in buckets:
                buckets[bucket_idx] = {"version": 1, "overlay_tags": {"frame_level_tags": {}}}
            if frame_idx not in buckets[bucket_idx]["overlay_tags"]["frame_level_tags"]:
                buckets[bucket_idx]["overlay_tags"]["frame_level_tags"][frame_idx] = {"timestamp_sec": tag.timestamp_sec}
            t = tag.to_dict()
            # remove timestamp_sec from the individual tag since we only need one per frame
            del t["timestamp_sec"]
            buckets[bucket_idx]["overlay_tags"]["frame_level_tags"][frame_idx][label_to_track(label)] = t
    buckets = [buckets[i] if i in buckets else {"version": 1, "overlay_tags": {"frame_level_tags": {}}} for i in range(max(buckets.keys())+1)]
    return buckets

# Args:
#   agg_tags: a dictionary mapping (label) -> list of aggregated tags (i.e shot_tags -> list of aggregated shot tags)
#   tracks: a dictionary mapping (feature name) -> list of tags (i.e the direct output of a 'service')
#   interval: formatted tracks are broken into buckets of this size (in minutes). If None, no bucketing is done.
#   
# Returns:
#    A tracks tag following the usual format for the video-tags-tracks files. Each element in the output list corresponds to one of the "video-tags-tracks-XXXX.json" files
def format_tracks(agg_tags: Dict[str, List[AggTag]], tracks: Dict[str, List[VideoTag]], interval: Optional[int]=None) -> List[Dict[str, object]]:
    result = {}
    # convert to milliseconds
    interval = interval*1000*60 if interval is not None else None
    # add aggregated tags
    for label, tags in agg_tags.items():
        key = label_to_track(label)
        for agg_tag in tags:
            entry: Any = {
                "start_time": agg_tag.start_time,
                "end_time": agg_tag.end_time,
                "text": {} 
            }
            bucket_idx = int(agg_tag.start_time/interval) if interval is not None else 0
            if bucket_idx not in result:
                result[bucket_idx] = {"version": 1, "metadata_tags": {}}
            if key not in result[bucket_idx]["metadata_tags"]:
                result[bucket_idx]["metadata_tags"][key] = {"label": label, "tags": []}
            
            for track, video_tags in agg_tag.tags.items():
                track_label = track_to_label(track)
                entry["text"][track_label] = []
                for vtag in video_tags:
                    as_dict = vtag.to_dict()
                    if "text" in as_dict and as_dict["text"]:
                        # NOTE: this is just a tag file convention, probably should just be a string value
                        as_dict["text"] = [as_dict["text"]]
                    entry["text"][track_label].append(as_dict) 

            result[bucket_idx]["metadata_tags"][key]["tags"].append(entry)

    # add standalone tracks
    for feature, video_tags in tracks.items():
        key = feature
        label = track_to_label(feature)
        for vtag in video_tags:
            entry = {
                "start_time": vtag.start_time,
                "end_time": vtag.end_time,
            }
            if vtag.text is not None:
                entry["text"] = vtag.text
            if vtag.data is not None:
                entry["data"] = vtag.data
            bucket_idx = int(vtag.start_time/interval) if interval is not None else 0
            if bucket_idx not in result:
                result[bucket_idx] = {"version": 1, "metadata_tags": {}}
            if key not in result[bucket_idx]["metadata_tags"]:
                result[bucket_idx]["metadata_tags"][key] = {"label": label, "tags": []}
            result[bucket_idx]["metadata_tags"][key]["tags"].append(entry)

    # convert to list
    return [result[i] if i in result else {"version": 1, "metadata_tags": {}} for i in range(max(result.keys())+1)]

# Extract individual tracks from formatted tags track file
def extract_frames(overlay: Dict[str, object]) -> Dict[str, FrameTags]:
    result = defaultdict(dict)
    for frame_idx, ftags in overlay["overlay_tags"]["frame_level_tags"].items():
        for key, tag in ftags.items():
            if key == "timestamp_sec":
                continue
            if "tags" not in tag:
                continue
            if key not in result:
                result[key] = {}
            new_tag = {}
            new_tag["timestamp_sec"] = ftags["timestamp_sec"]
            new_tag["tags"] = tag["tags"]
            result[key][int(frame_idx)] = FrameTagSchema().unmarshal(new_tag)
    return dict(result)

# Extract individual tracks from formatted tags track file
def extract_tracks(tracks: Dict[str, object]) -> Dict[str, List[VideoTag]]:
    result = {}
    for key, track in tracks["metadata_tags"].items():
        label = track["label"]
        if label in ["Shot Tags", "Sentence Tags", "STT Sentences"]:
            # skip aggregated tags
            continue
        track_name = label_to_track(label)

        old_stt = label == "Speech to Text" and len(track["tags"]) > 0 and "wordpiece_timestamps" in track["tags"][0]

        if old_stt:
            formatted_tags = sum(([VideoTagSchema().unmarshal({"start_time": int(wp[1]), "end_time": int(wp[1])+1, "text": wp[0], "coalesce": True}) for wp in tags["wordpiece_timestamps"]] for tags in track["tags"]), [])
        else:
            # NOTE: sometimes tag text is stored as string instead of single element list. we should stick to one or the other. 
            try:
                formatted_tags = []
                for v in track["tags"]:
                    if "text" not in v or len(v["text"]) > 0 and v["text"][0] == None:
                        formatted_tags.append(VideoTagSchema().unmarshal({"start_time": v["start_time"], "end_time": v["end_time"]}))
                    else:
                        formatted_tags.append(VideoTagSchema().unmarshal({"start_time": v["start_time"], "end_time": v["end_time"], "text": ' '.join(v["text"]) if type(v["text"])==list else v["text"]}))
            except Exception as e:
                print(1)
                raise e
   
        result[track_name] = formatted_tags
    return result