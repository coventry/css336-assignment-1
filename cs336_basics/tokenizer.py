from ast import literal_eval
from collections.abc import Iterable, Iterator
import functools
from itertools import chain
from os import PathLike
from typing import Self

from cs336_basics.common import EOT_TOKEN

from importlib import reload

import cs336_basics.pretokenizer

reload(cs336_basics.pretokenizer)
from cs336_basics.pretokenizer import pretokenizer_split


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],  # vocab idx -> utf-8 bytes
        merges: list[tuple[bytes, bytes]],  # merges from tokenizer training
        special_tokens: list[str] | None = None,  # Special BPE tokens
    ):
        """Construct a tokenizer from a given vocabulary, list of merges, and
        (optionally) a list of special tokens.

        """
        vi = {v: i for i, v in vocab.items()}  # vocab -> integer repr.
        mi = {(vi[p0], vi[p1]): i for i, (p0, p1) in enumerate(merges)}
        mr = {(vi[t0], vi[t1]): vi[t0 + t1] for (t0, t1) in merges}
        if special_tokens is None:
            special_tokens = [EOT_TOKEN]

        self.vocab = vocab
        self.vocab_idx: dict[bytes, int] = vi  # Reverse look-up
        # Token pair -> merge idx during BPE training
        self.merge_idx: dict[tuple[int, int], int] = mi
        # Pair of int repr's -> int repr of merge
        self.merge_results: dict[tuple[int, int], int] = mr
        # Special tokens which were used in BPE training
        self.special_tokens: list[str] = special_tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath: PathLike,
        merges_filepath: PathLike,
        special_tokens: list[str] | None = None,
    ) -> Self:
        """Constructs and returns a Tokenizer from a serialized vocabulary and
        list of merges (in the same format that your BPE training code output)
        and (optionally) a list of special tokens.

        """
        merges: list[tuple[bytes, bytes]] = []

        for line in open(merges_filepath, "rb"):
            merge: tuple[bytes, bytes] = literal_eval(line.decode())
            merge_types = list(map(type, merge))
            if not isinstance(merge, tuple) or merge_types != [bytes, bytes]:
                raise ValueError(
                    f"Not of type tuple[bytes, bytes]: {merge!r}. "
                    f"It's type {merge_types}"
                )
            merges.append(merge)

        vocab: dict[int, bytes] = {}
        for v_idx, line in enumerate(open(vocab_filepath, "rb")):
            vocab_entry = literal_eval(line.decode())
            if not isinstance(vocab_entry, bytes):
                raise ValueError(f"Not of type bytes: {vocab_entry: r}")
            vocab[v_idx] = vocab_entry
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        word_stream = pretokenizer_split(text, self.special_tokens, True)
        return list(chain(*map(self.tokenization, word_stream)))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a
        generator that lazily yields token IDs. This is required for
        memory-efficient tokenization of large files that we cannot directly
        load into memory.

        """
        return chain(*map(self.encode, iterable))

    def decode(self, ids: list[int]) -> str:
        "Decode a sequence of token IDs into text."
        kwargs = dict(errors="replace")
        return b"".join(self.vocab[t] for t in ids).decode("utf-8", **kwargs)

    @functools.lru_cache(maxsize=1_000_000_000)
    def tokenization(self, word: str) -> list[int]:
        "Return the tokenization of word (which should be a utf-8 string)"
        if word in self.special_tokens:
            # This works because special tokens are forced to be ASCII in
            # train_tokenizer.train_bpe
            return [self.vocab_idx[word.encode("ascii")]]
        # Get bytes of the word as utf-8, the most elementary BPE encoding
        w_tokens = [self.vocab_idx[bytes([c])] for c in word.encode("utf-8")]

        # Keep merging pairs of tokens in w_tokens until done
        while True:
            # Find token pair in the word first merged during BPE training.

            # TODO: Updating token_pairs and merge_idx_pairs in-place would be
            # more efficient than recalculating all pairs and merge indices on
            # each iteration.
            token_pairs = list(zip(w_tokens, w_tokens[1:]))
            merge_idx_pairs = [
                (self.merge_idx[p], p)
                for p in set(token_pairs)
                if p in self.merge_idx
            ]
            if len(merge_idx_pairs) == 0:
                break  # No more pairs to merge in self.merge_idx
            _, next_merge_pair = min(merge_idx_pairs)
            # Replace earliest-merged pair with token formed by merging them.
            merge_token = self.merge_results[next_merge_pair]
            wix = token_pairs.index(next_merge_pair)
            w_tokens[wix : wix + 2] = [merge_token]
            if len(w_tokens) == 1:
                break  # No more pairs to merge
        return w_tokens


if __name__ == "__main__":
    t = Tokenizer.from_files(
        "/workspace/repo/data/vocab.txt",
        "/workspace/repo/data/merges.txt",
        [EOT_TOKEN],
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    print([t.decode([x]) for x in t.encode(test_string)])
