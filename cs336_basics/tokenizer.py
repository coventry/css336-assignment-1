from collections import Counter

from cs336_basics.pretokenizer import get_pretoken_counts

# In this code, for the sake of human interpretability of ASCII text, we use
# latin-1 strings to represent byte arrays. Its encoding has exactly 256
# elements, and the lower 128 is ASCII, making it relatively easy to interpret.


def train_bpe(
    input_path: str,  # Path to UTF-8 training dataset
    vocab_size: int,  # Number of tokens in final tokenizer
    special_tokens: list[str],  # List of special tokens to include & break on
) -> tuple[  # Return value types
    dict[int, str],  # Tokenizer map, token_num -> latin-1 token_contents
    list[tuple[str, str]],  # Pairs of latin-1 tokens merged during BPE process
]:
    if not set(map(ord, "".join(special_tokens))).issubset(range(128)):
        # This is for compatibility with the latin-1 encoding.
        raise ValueError("Special tokens must be ASCII")
    if len(special_tokens) + 256 > vocab_size:
        raise ValueError("vocab_size too small")
    # UTF-8 strings -> number of occurrences in training dataset
    pretoken_counts: Counter[str] = get_pretoken_counts(
        input_path, special_tokens
    )
    return bpe(pretoken_counts, vocab_size, special_tokens)


def bpe(
    pretoken_counts: Counter[str], vocab_size: int, special_tokens: list[str]
) -> tuple[
    dict[int, str],  # Tokenizer map, token_num -> latin-1 token_contents
    list[tuple[str, str]],  # Pairs of latin-1 tokens merged during BPE process
]:
    """Learn a byte-pair encoding vocabulary from weighted pretokens.

    Each Unicode pretoken is encoded as UTF-8. Individual bytes and learned
    byte sequences are represented as Latin-1 strings, with one Latin-1
    character per byte. Encoding one of these strings as Latin-1 therefore
    recovers the corresponding byte sequence.

    Training begins with ``special_tokens`` in the order supplied, followed
    by the 256 singleton byte tokens in byte-value order. On each iteration,
    the most frequent adjacent token pair is selected, all non-overlapping
    occurrences of that pair are merged from left to right, and the resulting
    token is appended to the vocabulary.

    Pair frequencies are weighted by the values in ``pretoken_counts``.
    Equal-frequency pairs are resolved by selecting the lexicographically
    greatest pair.

    This function does not itself identify or remove special tokens from the
    training data. It assumes that ``pretoken_counts`` was produced using
    special tokens as hard boundaries and that the special tokens themselves
    do not contribute to the counts.

    Args:
        pretoken_counts: A mapping from Unicode pretokens to their occurrence
            counts in the training corpus. Counts are used as frequency
            weights and the mapping is not modified.
        vocab_size: Desired total vocabulary size, including singleton byte
            tokens and special tokens. This must be at least
            ``256 + len(special_tokens)``.
        special_tokens: Tokens to place at the beginning of the vocabulary.
            They are inserted verbatim and do not participate in BPE merges.

    Returns:
        A pair ``(vocab, merges)`` where:

        * ``vocab`` maps token IDs to special tokens or Latin-1 representations
          of byte sequences.
        * ``merges`` contains the adjacent token pairs that were merged, in
          training order.

    Raises:
        ValueError: If another merge is needed to reach ``vocab_size`` but
            none of the remaining pretokens contains an adjacent token pair.

    Examples:
        Counts weight the pair frequencies. Here ``("a", "a")`` is merged
        before ``("a", "b")``:

        >>> from collections import Counter
        >>> vocab, merges = bpe(
        ...     Counter({"aa": 2, "ab": 1}),
        ...     vocab_size=258,
        ...     special_tokens=[],
        ... )
        >>> merges
        [('a', 'a'), ('a', 'b')]
        >>> vocab[97], vocab[256], vocab[257]
        ('a', 'aa', 'ab')

        Special tokens precede the singleton byte tokens and shift their IDs:

        >>> vocab, merges = bpe(
        ...     Counter({"ab": 1}),
        ...     vocab_size=258,
        ...     special_tokens=["<|endoftext|>"],
        ... )
        >>> vocab[0]
        '<|endoftext|>'
        >>> vocab[98], vocab[99], vocab[257]
        ('a', 'b', 'ab')
        >>> merges
        [('a', 'b')]

        Non-ASCII text is learned from its UTF-8 bytes. The Latin-1 encoding
        of the learned token recovers those bytes:

        Ties select the lexicographically greatest pair:

        >>> _, merges = bpe(
        ...     Counter({"ab": 1, "xy": 1}),
        ...     vocab_size=257,
        ...     special_tokens=[],
        ... )
        >>> merges
        [('x', 'y')]

        Merging must preserve the original byte sequence when statistics are
        recomputed for subsequent iterations:

        >>> _, merges = bpe(
        ...     Counter({"aba": 1}),
        ...     vocab_size=258,
        ...     special_tokens=[],
        ... )
        >>> merges
        [('b', 'a'), ('a', 'ba')]
    """
    # Tuple of latin-1 strings -> number of occurrences in training dataset.
    bytewise_counts: dict[tuple[str, ...], int] = {
        # Initially, each tuple of the single latin-1 characters in the UTF-8
        # representation of the string from pretoken_counts
        tuple(w.encode("utf-8").decode("latin-1")): c
        for w, c in pretoken_counts.items()
    }
    # Initial vocab is the special tokens, plus the singleton bytes
    vocab: list[str] = special_tokens + [
        bytes([c]).decode("latin-1") for c in range(256)
    ]  # singleton bytes
    merges: list[tuple[str, str]] = []  # Max-freq pairs which got merged
    while len(vocab) < vocab_size:  # Loop until vocab size reaches vocab_size
        tok_pair = most_common_pair(bytewise_counts)
        new_token = "".join(tok_pair)  # Merge the pair
        vocab.append(new_token)
        merges.append(tok_pair)
        # Update bytewise_counts with the new, merged token TODO: Could be
        # optimized by updating pair_counts on the fly as merges are observed,
        # instead of fully recomputing?
        bytewise_counts = {
            merge_word(w, tok_pair): c for w, c in bytewise_counts.items()
        }
    return dict(enumerate(vocab)), merges


def most_common_pair(
    bytewise_counts: dict[tuple[str, ...], int],
) -> tuple[str, str]:
    """Return the most frequent adjacent token pair in a weighted vocabulary.

    For each tokenized word in ``bytewise_counts``, counts every adjacent
    token pair and weights each occurrence by the word's frequency. Multiple
    occurrences of the same pair within one word are counted separately.

    If multiple pairs have the same maximum frequency, the lexicographically
    greatest pair is returned, matching the tie-breaking behavior of
    ``max((count, pair), ...)``.

    Args:
        bytewise_counts: A mapping from tokenized words to their frequencies.

    Returns:
        The adjacent token pair with the greatest total weighted frequency.

    Raises:
        ValueError: If none of the words contains an adjacent token pair.

    Examples:
        Frequencies are accumulated across words:

        >>> most_common_pair({
        ...     ("a", "b", "c"): 4,
        ...     ("a", "b"): 3,
        ...     ("b", "c"): 1,
        ... })
        ('a', 'b')

        Repeated occurrences within a word are counted separately:

        >>> most_common_pair({("a", "b", "a", "b"): 2})
        ('a', 'b')

        Ties are resolved by choosing the lexicographically greatest pair:

        >>> most_common_pair({
        ...     ("a", "b"): 1,
        ...     ("x", "y"): 1,
        ... })
        ('x', 'y')

        At least one word must contain two or more tokens:

        >>> most_common_pair({("a",): 4, (): 2})
        Traceback (most recent call last):
        ...
        ValueError: At least one word must contain at least two tokens
    """
    pair_counts: Counter[tuple[str, str]] = Counter()
    for w, c in bytewise_counts.items():
        for tok1, tok2 in zip(w, w[1:]):
            pair_counts[tok1, tok2] += c
    if not pair_counts:
        raise ValueError("At least one word must contain at least two tokens")
    _, rv = max(  # Most frequent token pair
        (c, tok_pair) for tok_pair, c in pair_counts.items()
    )
    return rv


def merge_word(
    word: tuple[str, ...], tok_pair: tuple[str, str]
) -> tuple[str, ...]:
    """Merge adjacent occurrences of a token pair within a tokenized word.

    Scans ``word`` from left to right and replaces each non-overlapping
    occurrence of ``tok_pair`` with the concatenation of its two tokens.
    Tokens not participating in a merge are preserved unchanged.

    When occurrences overlap, the leftmost occurrence is merged. For
    example, merging ``("a", "a")`` in ``("a", "a", "a")`` produces
    ``("aa", "a")`` rather than ``("a", "aa")``.

    Args:
        word: The sequence of tokens making up one pretokenized word.
        tok_pair: The adjacent pair of tokens to merge.

    Returns:
        A tuple containing the tokens after all non-overlapping occurrences
        of ``tok_pair`` have been merged.

    Examples:
        Merge one occurrence:

        >>> merge_word(("l", "o", "w"), ("l", "o"))
        ('lo', 'w')

        Merge multiple occurrences:

        >>> merge_word(("a", "b", "a", "b"), ("a", "b"))
        ('ab', 'ab')

        Overlapping occurrences are resolved from left to right:

        >>> merge_word(("a", "a", "a"), ("a", "a"))
        ('aa', 'a')

        Return the word unchanged when the pair does not occur:

        >>> merge_word(("l", "o", "w"), ("x", "y"))
        ('l', 'o', 'w')

        Words containing fewer than two tokens cannot be merged:

        >>> merge_word(("word",), ("w", "o"))
        ('word',)
        >>> merge_word((), ("w", "o"))
        ()

        Regression test:

        >>> merge_word(("a", "b", "a", "b"), ("b", "a"))
        ('a', 'ba', 'b')
    """
    if len(word) < 2:
        # No merging is possible, and following code assumes at least length 2
        return word
    new_word: list[str] = []  # Builds up the merged word from `word`
    word_idx = 0  # Iterate over pairs, looking for merged pair
    while word_idx < len(word):
        if (
            word_idx < len(word) - 1
            and (word[word_idx], word[word_idx + 1]) == tok_pair
        ):
            new_word.append("".join(tok_pair))
            word_idx += 1  # Skip second token, since it's now merged
        else:
            new_word.append(word[word_idx])
        word_idx += 1
    assert "".join(new_word) == "".join(word), f"{new_word} != {word}"
    return tuple(new_word)


if __name__ == "__main__":
    merge_word(("a", "b", "a", "b"), ("b", "a"))
