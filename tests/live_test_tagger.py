from typing import List
import os

from common_ml.model import run_live_mode

def test_tag_fn(file_paths: List[str]) -> None:
    batch_size = len(file_paths)
    for fpath in file_paths:
        if os.path.exists(fpath):
            print(f"Tagging file: {fpath}")
            with open(f"{fpath}.txt", 'w') as f:
                f.write(str(batch_size))

if __name__ == '__main__':
    run_live_mode(test_tag_fn)