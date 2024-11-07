
import os
import argparse
import json
from elv_client_py import ElvClient
from loguru import logger

from common_ml.tag_formatting import format_fabric_tags

def get_auth_token(qid: str, elv_config: str) -> str:
    cmd = f"elv content token create {qid} --config {elv_config} --update"
    return json.loads(os.popen(cmd).read())["bearer"]

def get_write_token(qid: str, elv_config: str) -> str:
    cmd = f"elv content edit {qid} --config {elv_config}"
    res = json.loads(os.popen(cmd).read())
    return res["q"]["write_token"]

def finalize(qid: str, elv_config: str):
    cmd = f"elv content finalize {qid} --config {elv_config}"
    return os.popen(cmd).read()

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--qids",
        type=str,
        nargs='+',
        default=""
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--streams",
        type=str,
        nargs='+',
        default=["audio", "video"]
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        required=False
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
    )
    args = parser.parse_args()
    fabric_config = json.load(open(args.config, 'r'))["api"]["url"] + "/config?self&qspace=main"
    for qid in args.qids:
        auth_token = get_auth_token(qid, args.config)
        write_token = get_write_token(qid, args.config)
        logger.info(f"Created write token for {qid}: {write_token}")
        client = ElvClient.from_configuration_url(fabric_config, auth_token)
        format_fabric_tags(client, write_token, args.streams, args.interval)
        if args.finalize:
            finalize(qid, args.config)
        else:
            logger.info(f"Please finalize {qid} manually.\n write token={write_token}")

if __name__ == "__main__":
    main()
