import argparse
import os
import json
from typing import Dict, List, Any
from collections import defaultdict

from common_ml.legacy.formatting import extract_tracks, extract_frames

# go through the tags directory and find all tracks of the form video-tags-tracks-....json,
# extract the tracks and save them in the same directory as the tags file based on their feature name 
# which can be inferred by their label
def extract(qid: str, libid: str, save_path: str) -> None:
    content_dir = os.path.join(save_path, libid, qid)
    tracks_path = os.path.join(content_dir, 'tracks')
    frames_path = os.path.join(content_dir, 'frames')
    if not os.path.exists(tracks_path):
        os.makedirs(tracks_path)
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    tracks = defaultdict(list)
    frames: Dict[str, Dict[int, Any]] = {}
    for tags_file in sorted(os.listdir(content_dir)):
        if is_track_file(tags_file):
            with open(os.path.join(content_dir, tags_file), 'r') as f:
                data = json.load(f)
            vtracks = extract_tracks(data)
            for feature, video_tags in vtracks.items():
                tracks[feature].extend([video_tag.to_dict() for video_tag in video_tags])
        if is_overlay_file(tags_file):
            with open(os.path.join(content_dir, tags_file), 'r') as f:
                data = json.load(f)
            vframes = extract_frames(data)
            for feature, frame_tags in vframes.items():
                if feature not in frames:
                    frames[feature] = {}
                frames[feature].update({frame_idx: frame_tag.to_dict() for frame_idx, frame_tag in frame_tags.items()})

    for feature, frame_tags in frames.items():
        save_path = os.path.join(frames_path, f"{feature}_frames.json")
        with open(save_path, 'w') as f:
            json.dump(frame_tags, f)

    for feature, video_tags in tracks.items():
        save_path = os.path.join(tracks_path, f"{feature}.json")
        with open(save_path, 'w') as f:
            json.dump(video_tags, f)

def is_track_file(tags_file: str) -> bool:
    return tags_file.startswith("video-tags-tracks-0") and tags_file.endswith(".json")

def is_overlay_file(tags_file: str) -> bool:
    return tags_file.startswith("video-tags-overlay-0") and tags_file.endswith(".json")

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--qids",
        type=str,
        nargs='+',
        default=""
    )
    parser.add_argument(
        "--libid",
        type=str,
        help="library id",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    qids = args.qids
    if not qids:
        qids = os.listdir(os.path.join(args.save_path, args.libid))
    for qid in qids:
        extract(qid, args.libid, args.save_path)

if __name__ == "__main__":
    main()