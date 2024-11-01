
import json
import argparse
import os
from typing import List, Tuple

from common_ml.legacy import agg
from common_ml.legacy import utils
from common_ml.legacy.formatting import format_tracks, format_overlay
from common_ml.legacy.tags import AggTag
from common_ml.legacy.tags import *

"""
Driver script for the tag aggregation

Usage:
    python bin/agg.py --id <content id> --shot
    python bin/agg.py --id <content id> --sentence
"""

def main():
    if args.contents and args.qids:
        raise ValueError("Cannot specify both --contents and --qids")
    save_path = args.save_path
    qids = []
    lib_path = os.path.join(save_path, args.library)
    if args.contents:
        with open(args.contents, 'r') as f:
            qids = list(map(lambda x: x.strip(), f))
    elif args.qids:
        qids = args.qids
    else:
        # load all contents from library
        qids = os.listdir(lib_path)

    for qid in qids:
        content_dir = os.path.join(lib_path, qid)
        agg_tags = {}
        if args.shot:
            # add shot aggregated tags
            agg_tags["Shot Tags"] = agg_shot(qid, args.library, args.save_path)
        if args.sentences:
            # add sentence aggregated tags
            agg_tags["Sentence Tags"] = agg_sentences(qid, args.library, args.save_path)
        tracks = load_tracks(os.path.join(content_dir, 'tracks'))
        agg_tracks = format_tracks(agg_tags, tracks, args.interval)
        for i, track in enumerate(agg_tracks):
            with open(os.path.join(content_dir, f"video-tags-tracks-{i:04d}.json"), 'w') as f:
                json.dump(track, f)
        overlays = create_overlays(os.path.join(content_dir, 'frames'), args.interval)
        for i, overlay in enumerate(overlays):
            with open(os.path.join(content_dir, f"video-tags-overlay-{i:04d}.json"), 'w') as f:
                json.dump(overlay, f)

def agg_sentences(qid: str, libid: str, save_path: str) -> List[AggTag]:
    content_dir = os.path.join(save_path, libid, qid)
    tracks_dir = os.path.join(content_dir, 'tracks')
    if not os.path.exists(os.path.join(tracks_dir, 'speech_to_text.json')):
        raise ValueError("No asr tags found")
    tags = load_tracks(tracks_dir)
    tags = {"speech_to_text": tags["speech_to_text"]}
    intervals = agg.get_sentence_intervals(tags['speech_to_text'])
    agg_tags = agg.aggregate(intervals, tags)

    return agg_tags

def create_overlays(content_dir: str, interval: int) -> List[dict]:
    frame_tags = {}
    for path in os.listdir(content_dir):
        if not path.endswith("_frames.json"):
            continue
        feature = path.split("_")[0]
        # load frame level tags (based on frame idx)
        with open(os.path.join(content_dir, path), 'r') as f:
            f_tags = json.load(f)
        frame_tags[feature] = f_tags

    # convert to int keys
    frame_tags = {feature: _unpack_frame_tags(f_tags) for feature, f_tags in frame_tags.items()}
    overlay_tags = format_overlay(frame_tags, interval)
    return overlay_tags

def _unpack_frame_tags(frame_tags: Any) -> FrameTags:
    res = {}
    for idx, f_tag in frame_tags.items():
        res[int(idx)] = FrameTagSchema().unmarshal(f_tag)
    return res
    
def get_shot_intervals(content_dir: str) -> List[Tuple[int, int]]:
    if not os.path.exists(os.path.join(content_dir, 'shot_detection.json')):
        raise ValueError(f"No shot tags found path={content_dir}")

    with open(os.path.join(content_dir, f"shot_detection.json"), 'r') as f:
        shot_tags = json.load(f)
    shot_tags = [VideoTagSchema().unmarshal(t) for t in shot_tags]

    return agg.get_tag_intervals(shot_tags)

def load_tracks(content_dir: str) -> Dict[str, List[VideoTag]]:
    tracks = {}
    for feature in os.listdir(content_dir):
        if feature.endswith(".json"):
            with open(os.path.join(content_dir, feature), 'r') as f:
                tags = json.load(f)
            tracks[feature.split(".")[0]] = [VideoTagSchema().unmarshal(t) for t in tags]
    return tracks

def agg_shot(id: str, libid: str, save_path: str) -> List[AggTag]:
    content_dir = os.path.join(save_path, libid, id)
    tracks_dir = os.path.join(content_dir, 'tracks')
    intervals = get_shot_intervals(tracks_dir)   
    tracks = load_tracks(tracks_dir)
    del tracks['shot_detection']

    return agg.aggregate(intervals, tracks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate tags")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--contents", type=str, default="")
    parser.add_argument("--qids", type=str, nargs='+', default="")
    parser.add_argument("--library", type=str)
    parser.add_argument("--sentences", help="aggregate on stt sentence boundaries", action='store_true')
    parser.add_argument("--shot", help="aggregate on shot boundaries", action='store_true')
    parser.add_argument("--interval", help="split into multiple files based on this time range (in seconds)", default=None, type=int)
    args = parser.parse_args()
    main()