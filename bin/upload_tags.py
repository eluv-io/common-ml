import os
import argparse
import json
from loguru import logger
from json.decoder import JSONDecodeError
from typing import Optional
from datetime import datetime

def upload_file(src_path: str, dest_folder: str, qwt: str, libid: str, cli_config: str, use_elv: bool=False) -> None:
    client = 'elv' if use_elv else 'qfab_cli'
    cmd = [client, 'files', 'upload', src_path, '--qwt', qwt, '--library', libid, '--dest-path', dest_folder, '--config', cli_config, "--force-no-encrypt"]
    result = os.popen(' '.join(cmd)).read()
    logger.info(result)

def dict_to_str(d: dict) -> str:
    return '{' + ', '.join(f'"{k}": "{v}"' for k, v in d.items()) + '}'

def add_link(filename: str, qwt: str, cli_config: str, use_elv: bool=False):
    client = 'elv' if use_elv else 'qfab_cli'
    if 'video-tags-tracks' in filename:
        tag_type = 'metadata_tags'
    elif 'video-tags-overlay' in filename:
        tag_type = 'overlay_tags'
    else:
        return
    idx = ''.join([char for char in filename if char.isdigit()])

    data = "'" + dict_to_str({"/": f"./files/video_tags/{filename}"}) + "'"
    cmd = [client, 'content', 'meta', 'merge', qwt, data, f'video_tags/{tag_type}/{idx}', '--config', cli_config]
    result = os.popen(' '.join(cmd)).read()
    logger.info(result)

def merge(qwt: str, data: dict, path: str, cli_config: str, use_elv: bool=False):
    client = 'elv' if use_elv else 'qfab_cli'
    data = "'" + dict_to_str(data) + "'"
    cmd = [client, 'content', 'meta', 'merge', qwt, data, path, '--config', cli_config]
    result = os.popen(' '.join(cmd)).read()
    logger.info(result)

def edit(client: str, id: str, cli_config: str) -> str:
    cmd = [client, 'content', 'edit', id, '--config', cli_config]
    stream = os.popen(' '.join(cmd))
    output = stream.read()
    try:
        tok = json.loads(output)['q']['write_token']
    except (JSONDecodeError, KeyError):
        raise RuntimeError(f"error getting write token for {id}, output: {output}")
    return tok

def finalize(client: str, tok: str, cli_config: str, message: Optional[str]) -> str:
    if message is None:
        message = "Uploaded tags"
    merge(tok, {"message": message, "timestamp": datetime.now().isoformat(timespec='microseconds') + 'Z'}, 'commit', cli_config)
    cmd = [client, 'content', 'finalize', tok, '--config', cli_config]
    return os.popen(' '.join(cmd)).read()

def upload_directory(id: str, libid: str, dirname: str, cli_config: str, dest_path="/video_tags", use_elv: bool=False, do_finalize: bool=False) -> None:
    client = 'elv' if use_elv else 'qfab_cli'
    write_token = edit(client, id, cli_config)
    for tag_file in os.listdir(dirname):
        upload_file(os.path.join(dirname, tag_file), dest_path, write_token, libid, cli_config, use_elv)
        logger.info(f"done uploading files for {id}")

    if do_finalize:
        finalize(client, write_token, cli_config)
        logger.info(f"published tags to fabric")
    else:
        logger.info(f"done uploading files, ready to finalize token: {write_token}")

def upload_tags(id: str, libid: str, cli_config: str, do_finalize: bool, message: Optional[str], tags_path: str, use_elv: bool=False) -> None:
    tags_path = os.path.join(tags_path, libid, id)
    client = 'elv' if use_elv else 'qfab_cli'
    write_token = edit(client, id, cli_config)
    for tag_file in os.listdir(tags_path):
        if is_track_file(tag_file):
            upload_file(os.path.join(tags_path, tag_file), 'video_tags', write_token, libid, cli_config, use_elv)
            add_link(tag_file, write_token, cli_config, use_elv)
            logger.info(f"done uploading files for {id}")
    
    if do_finalize:
        finalize(client, write_token, cli_config, message)
        logger.info(f"published tags to fabric")
    else:
        logger.info(f"done uploading files, ready to finalize token: {write_token}")

def is_track_file(tags_file: str) -> bool:
    return tags_file.startswith("video-tags-tracks-0") and tags_file.endswith(".json")

def is_overlay_file(tags_file: str) -> bool:
    return tags_file.startswith("video-tags-overlay-0") and tags_file.endswith(".json")

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--qids",
        type=str,
        nargs='+',
        default=""
    )
    parser.add_argument(
        "--contents",
        type=str,
        default=""
    )
    parser.add_argument(
        "--libid",
        type=str,
        help="library id",
    )
    parser.add_argument(
        "--finalize",
        action='store_true',
    )
    parser.add_argument(
        "--message",
        type=str,
        help="commit message",
    )
    parser.add_argument(
        "--use_elv",
        action='store_true',
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True
    )
    args = parser.parse_args()
    if args.contents:
        with open(args.contents, 'r') as f:
            ids = f.readlines()
    else:
        ids = args.qids
    if not ids:
        # iterate through all qids in the library
        ids = list(os.listdir(os.path.join(args.save_path, args.libid)))
    for id in ids:
        logger.info(f"uploading tags for {id}")
        upload_tags(id, args.libid, args.config, args.finalize, args.message, args.save_path, args.use_elv)

if __name__ == "__main__":
    main()