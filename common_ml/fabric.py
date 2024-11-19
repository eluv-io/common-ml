from elv_client_python import ElvClient
from typing import List, Dict, Any
from collections import defaultdict
from requests.exceptions import HTTPError
from loguru import logger

def get_tags(qhot: str, client: ElvClient, start_time: int, end_time: int, padding: int, include_tracks: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    qhash = _resolve_hash(client, qhot)
    try:
        tag_files = client.content_object_metadata(
            version_hash=qhash, metadata_subtree='video_tags/metadata_tags').keys()
    except HTTPError as e:
        raise ValueError(
            f"Failed to get tags from content hash: {qhash} with error {e}")

    relevant_files = [file for file in tag_files if len(file) == 4 and file.isdigit()]
    relevant_files.sort()

    tag_duration = 10 * 60 * 1000 # NOTE: assumption that each tag file is 10 minutes long

    left_file, right_file = start_time // tag_duration, end_time // tag_duration

    # Pad because tags are placed in files according to start time. So we might miss overlapping tags on the ends. 
    left_file = max(0, left_file-1)
    right_file = min(len(relevant_files)-1, right_file+1)

    all_tags = []
    for file in relevant_files[left_file:right_file+1]:
        try:
            shot_tags = client.content_object_metadata(
                version_hash=qhash, metadata_subtree=f'video_tags/metadata_tags/{file}/metadata_tags/shot_tags', resolve_links=True)
            all_tags.append(shot_tags)
        except HTTPError as e:
            logger.warn(
                f"Failed to get tags from file: {file} with error {e}")
        
    res = defaultdict(list)
    ref_times = [max(start_time-padding, 0), end_time+padding]
    for chunk in all_tags:
        if chunk['label'] != 'Shot Tags':
            logger.warn(
                f"Chunk label is {chunk['label']}, expected 'Shot Tags'.")
        for tag in chunk['tags']:
            # check if shot overlap with the requested time range
            if _does_overlap(tag["start_time"], tag['end_time'], ref_times=ref_times):
                tag_fields = tag["text"]
                for track, track_data in tag_fields.items():
                    if track not in include_tracks:
                        continue
                    for td in track_data:
                        # is the tag within the requested time range
                        if _is_within(td["start_time"], td["end_time"], ref_times=ref_times):
                            res[track].append(td)
    return res

def _resolve_hash(client: ElvClient, q: str) -> str:
    if q.startswith("hq__"):
        qhash = q
    elif q.startswith("iq__"):
        qhash = client.content_object(q)["hash"]
    elif q.startswith("tq__"):
        raise ValueError("Write tokens are not supported.")
    else:
        raise ValueError("Invalid content id or hash.")
    return qhash

def _is_within(st_time, end_time, ref_times=[0, 1e9]):
    return ref_times[0] <= st_time <= ref_times[1] and ref_times[0] <= end_time <= ref_times[1]

def _does_overlap(st_time, end_time, ref_times=[0, 1e9]):
    if ref_times[0] <= st_time <= ref_times[1] or ref_times[0] <= end_time <= ref_times[1]:
        return True
    elif st_time <= ref_times[0] <= end_time or st_time <= ref_times[1] <= end_time:
        return True
    return False