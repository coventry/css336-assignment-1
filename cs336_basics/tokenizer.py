from collections import Counter, defaultdict as ddict

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
    # Initial vocab is the special tokens, plus the singleton bytes
    vocab: list[str] = special_tokens + [
        bytes([c]).decode("latin-1") for c in range(256)
    ]  # singleton bytes
    merges: list[tuple[str, str]] = []  # Max-freq pairs which got merged
    # Tuple of latin-1 strings -> number of occurrences in training dataset.
    pair_counts: Counter[tuple[str, str]] = Counter()  # Pair -> pair count
    # Pair -> words which contain pair
    pair_words: dict[tuple[str, str], set[tuple[str, ...]]] = ddict(set)
    word_counts: dict[tuple[str, ...], int] = {}
    for w, c in pretoken_counts.items():
        w = tuple(w.encode("utf-8").decode("latin-1"))  # Tuple of "bytes"
        if len(w) < 2:
            continue  # Irrelevant to BPE, if there are no pairs.
        word_counts[w] = c
        for tok1, tok2 in zip(w, w[1:]):
            pair_counts[tok1, tok2] += c
            pair_words[tok1, tok2].add(w)
    # Loop until vocab size reaches vocab_size or all pairs merged
    while len(vocab) < vocab_size and pair_counts:
        # Find the most common pair, tie-break counts lexicographically
        _, tok_pair = max((c, p) for p, c in pair_counts.items())
        new_token = "".join(tok_pair)  # Merge the pair
        vocab.append(new_token)
        merges.append(tok_pair)
        for word in pair_words[tok_pair]:
            new_word, new_pairs, removed_pairs = merge_word(word, tok_pair)
            word_count = word_counts[word]
            del word_counts[word]  # Forget old word
            for p, c in removed_pairs.items():
                pair_counts[p] -= c * word_count
                assert pair_counts[p] >= 0
                if pair_counts[p] == 0:
                    del pair_counts[p]
                pair_words[p].remove(word)
                if not pair_words[p]:
                    del pair_words[p]
            if len(new_word) <= 1:  # Only relevant to BPE if at least a pair
                continue
            word_counts[new_word] = word_count
            for p, c in new_pairs.items():
                pair_counts[p] += c * word_count
                pair_words[p].add(new_word)
        del pair_words[tok_pair]
        del pair_counts[tok_pair]
    return dict(enumerate(vocab)), merges


def merge_word(word: tuple[str, ...], tok_pair: tuple[str, str]) -> tuple[
    tuple[str, ...],  # Merged word
    dict[tuple[str, str], int],  # New pairs from merged word
    dict[tuple[str, str], int],  # Removed pairs from old word
]:
    if len(word) < 2:
        # No merging is possible, and following code assumes at least length 2
        return word, {}, {}
    new_token = "".join(tok_pair)  # Merge the pair
    new_word: list[str] = []
    # Counts of new pairs resulting from the merge
    new_pairs: dict[tuple[str, str], int] = Counter()
    # Counts of old pairs which must be removed, given the merge
    removed_pairs: dict[tuple[str, str], int] = Counter()
    word_idx = 0
    while word_idx < len(word) - 1:
        if (word[word_idx], word[word_idx + 1]) == tok_pair:
            # We found the token pair; we'll want to remove that since
            # now it's going to be merged.
            removed_pairs[tok_pair] += 1
            new_word.append(new_token)
            if word_idx > 0:  # If there's a token prior to tok_pair
                pre_tok = word[word_idx - 1]
                new_tok_pair = (pre_tok, new_token)
                new_pairs[new_tok_pair] += 1
                old_tok_pair = (pre_tok, tok_pair[0])
                removed_pairs[old_tok_pair] += 1
            if word_idx < len(word) - 2:  # If there's a token after...
                post_tok = word[word_idx + 1]
                new_tok_pair = (new_token, post_tok)
                new_pairs[new_tok_pair] += 1
                old_tok_pair = (tok_pair[1], post_tok)
                removed_pairs[old_tok_pair] += 1
                removed_pairs[(tok_pair[1], post_tok)] += 1
            word_idx += 2  # Skip past the pair
        else:
            new_word.append(word[word_idx])
            word_idx += 1  # Move to next token in words
    assert "".join(new_word) == "".join(word), f"{new_word} != {word}"
    return tuple(new_word), new_pairs, removed_pairs


if __name__ == "__main__":
    merge_word(("a", "b", "a", "b"), ("b", "a"))
