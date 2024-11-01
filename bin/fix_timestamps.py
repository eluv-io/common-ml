import argparse
import json
import os
from loguru import logger

from config import config

def fix_timestamps(qid: str, libid: str, fps: int=None) -> None:
    content_dir = os.path.join(config["storage"]["tags"], libid, qid)
    for tags_file in sorted(os.listdir(content_dir)):
        if is_overlay_file(tags_file):
            with open(os.path.join(content_dir, tags_file), 'r') as f:
                data = json.load(f)
            if not fps:
                logger.info(f"fps not provided, using fps from overlay file {tags_file}")
                fps = data["overlay_tags"]["frames_per_sec"]
            for f_idx, tag in data["overlay_tags"]["frame_level_tags"].items():
                tag["timestamp_sec"] = (int(f_idx) / fps) * 1000

            with open(os.path.join(content_dir, tags_file), 'w') as f:
                json.dump(data, f)

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
    qids = args.qids
    if not qids:
        qids = os.listdir(os.path.join(config["storage"]["tags"], args.libid))
    for qid in qids:
        fix_timestamps(qid, args.libid, args.fps)

if __name__ == "__main__":
    main()