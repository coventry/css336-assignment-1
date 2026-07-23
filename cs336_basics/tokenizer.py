from ast import literal_eval
from collections.abc import Iterable, Iterator
import functools
from typing import Self

from cs336_basics.common import EOT_TOKEN


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, str],  # vocab idx -> latin-1 str
        merges: list[tuple[str, str]],  # merges from tokenizer training
        special_tokens: list[str] | None = None,  # Special tokens used in BPE
    ):
        """Construct a tokenizer from a given vocabulary, list of merges, and
        (optionally) a list of special tokens.

        """
        mi = {p: i for i, p in enumerate(merges)}
        vi = {v: i for i, v in vocab.items()}  # vocab -> integer repr.
        mr = {(vi[t0], vi[t1]): vi[t0 + t1] for (t0, t1) in merges}
        self.merge_idx: dict[tuple[str, str], int] = mi  # Pair -> merge rank
        # Pair of int repr's -> int repr of merge
        self.merge_results: dict[tuple[int, int], int] = mr
        # Special tokens which were used in BPE training
        self.special_tokens: list[str] = special_tokens or [EOT_TOKEN]

    @classmethod
    def from_files(
        cls, vocab_filepath, merges_filepath, special_tokens=None
    ) -> Self:
        """Constructs and returns a Tokenizer from a serialized vocabulary and
        list of merges (in the same format that your BPE training code output)
        and (optionally) a list of special tokens.

        """
        merges: list[tuple[str, str]] = []

        def decode(s: bytes) -> str:
            return s.decode("latin-1")

        for line in open(merges_filepath, "rb"):
            merge: tuple[bytes, bytes] = literal_eval(str(line))
            merge_types = list(map(type, merge))
            if not isinstance(merge, tuple) or merge_types != [bytes, bytes]:
                raise ValueError(f"Not of type tuple[bytes, bytes]: {merge:r}")
            merges.append((decode(merge[0]), decode(merge[1])))

        vocab: dict[int, str] = {}
        for v_idx, line in enumerate(open(vocab_filepath, "rb")):
            vocab_entry = literal_eval(str(line))
            if not isinstance(vocab_entry, bytes):
                raise ValueError(f"Not of type bytes: {vocab_entry: r}")
            vocab[v_idx] = decode(vocab_entry)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a
        generator that lazily yields token IDs. This is required for
        memory-efficient tokenization of large files that we cannot directly
        load into memory.

        """
        return iter(())  # XXX:

    def decode(self, ids: list[int]) -> str:
        "Decode a sequence of token IDs into text."
        return ""  # XXX:

    @functools.lru_cache(maxsize=int(1e9))
    def tokenization(word: str) -> list[int]:
        "Return the tokenization of word (a latin-1 string)"
        return []  # XXX:
