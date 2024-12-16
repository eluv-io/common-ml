import argparse
from typing import Any, List
import random
from quick_test_py import Tester
import os
from dataclasses import asdict
from elv_client_py import ElvClient

from common_ml.fabric import get_tags

test_file = os.path.join(os.path.dirname(__file__), 'test.mp4')
auth_env = "TEST_AUTH"
qid = 'iq__42WgpoYgLTyyn4MSTejY3Y4uj81o'

def test_get_tags(auth: str): 
    client = ElvClient.from_configuration_url("https://main.net955305.contentfabric.io/config", auth)
    def t1():
        tags = get_tags(qid, client, 0, 100000, 0)
        return tags
    return [t1]
    
def main():
    auth = os.environ.get(auth_env)
    if auth is None:
        raise ValueError(f"Please set the {auth_env} environment variable")
    tester = Tester(os.path.dirname(__file__) + '/test_data')
    tester.register('test_get_tags', test_get_tags(auth))
    if args.record:
        tester.record()
    else:
        tester.validate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    main()