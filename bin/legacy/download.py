from loguru import logger
import os
import shutil
import argparse

def download_tags(id: str, libid: str, cli_config: str, save_path: str, use_elv: bool=False) -> None:
    client = 'elv' if use_elv else 'qfab_cli'
    save_path = os.path.join(save_path, libid, id)
    os.makedirs(save_path, exist_ok=True)
    if id.startswith("iq__"):
        cmd = [client, 'files', 'download', 'video_tags', save_path, '--qid', id, '--config', cli_config, '--library', libid, '--decryption-mode', 'none']
    else:
        cmd = [client, 'files', 'download', 'video_tags', save_path, '--qhash', id, '--config', cli_config, '--library', libid, '--decryption-mode', 'none']
    result = os.popen(' '.join(cmd)).read()
    logger.info(result)
    # files are saved under video_tags directory, move them up one level and delete video_tags
    for entry in os.listdir(os.path.join(save_path, 'video_tags')):
        if os.path.isdir(os.path.join(save_path, 'video_tags', entry)):
            shutil.rmtree(os.path.join(save_path, entry), ignore_errors=True)
        os.rename(os.path.join(save_path, 'video_tags', entry), os.path.join(save_path, entry))
    os.rmdir(os.path.join(save_path, 'video_tags'))

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
        required=False
    )
    parser.add_argument(
        "--libid",
        type=str,
        help="library id",
    )
    parser.add_argument(
        "--use_elv",
        action='store_true',
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    if args.qids and args.contents:
        raise ValueError("Cannot specify both qids and contents")
    if args.qids:
        qids = args.qids
    if args.contents:
        with open(args.contents, 'r') as f:
            qids = [line.strip() for line in f]
    for qid in qids:
        download_tags(qid, args.libid, args.config, args.save_path, args.use_elv)

if __name__ == "__main__":
    main()