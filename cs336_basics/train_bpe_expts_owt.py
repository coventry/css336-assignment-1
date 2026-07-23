from cs336_basics.train_tokenizer import train_bpe
from cs336_basics.common import (
    EOT_TOKEN,
    ROOT_PATH,
)

VOCAB_SIZE = 32_000
if __name__ == "__main__":
    TRAIN_PATH = ROOT_PATH / "data/owt_train.txt"
    vocab, merges = train_bpe(str(TRAIN_PATH), VOCAB_SIZE, [EOT_TOKEN])
    vocab_file = open(ROOT_PATH / "data/owt-vocab.txt", "w")
    for v in range(len(vocab)):
        print(vocab[v], file=vocab_file)
    vocab_file.close()
    merges_file = open(ROOT_PATH / "data/owt-merges.txt", "w")
    for m in merges:  # Store as tuple representation, for ease of parsing.
        print(repr((m[0], m[1])), file=merges_file)
    merges_file.close()
