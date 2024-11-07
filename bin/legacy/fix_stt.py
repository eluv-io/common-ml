import argparse
import json
import os
from loguru import logger

from config import config

def fix_asr(qid: str, libid: str, filename: str) -> None:
    content_dir = os.path.join(config["storage"]["tags"], libid, qid)
    tracks_path = os.path.join(content_dir, config["tags"]["tracks"])
    with open(os.path.join(tracks_path, filename), 'r') as f:
        data = json.load(f)

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
        "--fps",
        type=int,
        help="fps of the content, defaults to what's given in the overlay files if not provided",
    )
    args = parser.parse_args()
    for qid in args.qids:
        fix_timestamps(qid, args.libid, args.fps)

if __name__ == "__main__":
    main()