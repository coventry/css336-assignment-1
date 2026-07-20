import pathlib
import sys

ROOT_PATH = pathlib.Path("/workspace/repo")

if (spath := str(ROOT_PATH)) not in sys.path:
    sys.path.append(spath)

TRAIN_PATH = ROOT_PATH / "data/TinyStoriesV2-GPT4-train.txt"
VALIDATION_PATH = ROOT_PATH / "data/TinyStoriesV2-GPT4-valid.txt"

EOT_TOKEN = "<|endoftext|>"
