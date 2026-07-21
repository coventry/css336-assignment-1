class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, str],  # vocab idx -> latin-1 str
        merges: list[tuple[str, str]],  # merges from tokenizer training
        special_tokens: list[str] = None,  # Spec. tokens used in BPE training
    ):
        """Construct a tokenizer from a given vocabulary, list of merges, and
        (optionally) a list of special tokens.

        """

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Constructs and returns a Tokenizer from a serialized vocabulary and
        list of merges (in the same format that your BPE training code output)
        and (optionally) a list of special tokens.

        """

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a
        generator that lazily yields token IDs. This is required for
        memory-efficient tokenization of large files that we cannot directly
        load into memory.

        """

    def decode(self, ids: list[int]) -> str:
        "Decode a sequence of token IDs into text."
