import pathlib
import sys

ROOT_PATH = pathlib.Path("/workspace/repo")

if (spath := str(ROOT_PATH)) not in sys.path:
    sys.path.append(spath)

DATA_PATH = ROOT_PATH / "data"

TRAIN_PATH = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
VALIDATION_PATH = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"
TOKENIZER_VOCAB_PATH = DATA_PATH / "vocab.txt"
TOKENIZER_MERGES_PATH = DATA_PATH / "merges.txt"

OWT_TRAIN_PATH = DATA_PATH / "owt_train.txt"
OWT_VALIDATION_PATH = DATA_PATH / "owt_valid.txt"
OWT_TOKENIZER_VOCAB_PATH = DATA_PATH / "owt-vocab.txt"
OWT_TOKENIZER_MERGES_PATH = DATA_PATH / "owt-merges.txt"


EOT_TOKEN = "<|endoftext|>"
