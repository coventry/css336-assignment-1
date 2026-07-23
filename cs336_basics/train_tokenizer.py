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
    # Initial vocab is the special tokens, plus the singleton bytes
    vocab: list[str] = special_tokens + [
        bytes([c]).decode("latin-1") for c in range(256)
    ]  # singleton bytes
    merges: list[tuple[str, str]] = []  # Max-freq pairs which got merged
    # Tuple of latin-1 strings -> number of occurrences in training dataset.
    pair_counts: Counter[tuple[str, str]] = Counter()  # Pair -> pair count
    # Pair -> words which contain pair, in latin-1 encoding
    pair_words: dict[tuple[str, str], set[str]] = ddict(set)
    # latin-1 string -> current tokenization for that string.
    current_tokenizations: dict[str, tuple[str, ...]] = {}
    word_counts: dict[str, int] = Counter()  # Latin-1 string -> word count
    for w, c in pretoken_counts.items():
        latin1_string = w.encode("utf-8").decode("latin-1")
        tokenization = tuple(latin1_string)  # "bytes"
        if len(tokenization) < 2:
            continue  # Irrelevant to BPE, if there are no pairs.
        word_counts[latin1_string] = c
        current_tokenizations[latin1_string] = tokenization
        for tok1, tok2 in zip(latin1_string, latin1_string[1:]):
            pair_counts[tok1, tok2] += c
            pair_words[tok1, tok2].add(latin1_string)
    # Loop until vocab size reaches vocab_size or all pairs merged
    while len(vocab) < vocab_size and pair_counts:
        # Find the most common pair, tie-break counts lexicographically
        _, tok_pair = max((c, p) for p, c in pair_counts.items())
        new_token = "".join(tok_pair)  # Merge the pair
        vocab.append(new_token)
        merges.append(tok_pair)
        words = pair_words[tok_pair].copy()  # XXX: Debugging facility
        for word in pair_words[tok_pair].copy():
            tokenization = current_tokenizations[word]
            new_word, new_pairs, removed_pairs = merge_word(
                tokenization, tok_pair
            )
            word_count = word_counts[word]
            for p, c in removed_pairs.items():
                full_count = c * word_count
                # Must check before subtraction, because Counter won't go -ve.
                assert pair_counts[p] >= full_count
                pair_counts[p] -= full_count
                if pair_counts[p] == 0:
                    del pair_counts[p]
                if p not in zip(new_word, new_word[1:]):
                    pair_words[p].discard(word)
                if not pair_words[p]:
                    del pair_words[p]
            if len(new_word) > 1:  # Only relevant to BPE if at least a pair
                for p, c in new_pairs.items():
                    pair_counts[p] += c * word_count
                    pair_words[p].add(word)
            current_tokenizations[word] = new_word
        assert tok_pair not in pair_words  # Should have been removed by update
        assert tok_pair not in pair_counts
    byte_vocab = [v.encode("latin-1") for v in vocab]
    byte_merges = [tuple(t.encode("latin-1") for t in m) for m in merges]
    return dict(enumerate(byte_vocab)), byte_merges


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
    word_idx = 0
    while word_idx < len(word) - 1:
        if (word[word_idx], word[word_idx + 1]) == tok_pair:
            new_word.append(new_token)
            word_idx += 2  # Skip past the pair
        else:
            new_word.append(word[word_idx])
            word_idx += 1  # Move to next token in words
    if word_idx == len(word) - 1:  # Above will usually not add the last token
        new_word.append(word[-1])
    new_word_pairs = Counter(zip(new_word, new_word[1:]))
    old_pairs = Counter(zip(word, word[1:]))
    # Counter automatically removes items with negative counts
    new_pairs = new_word_pairs - old_pairs
    removed_pairs = old_pairs - new_word_pairs
    assert tok_pair not in new_pairs
    assert "".join(new_word) == "".join(word), f"{new_word} != {word}"
    return tuple(new_word), new_pairs, removed_pairs


if __name__ == "__main__":
    merge_word(("a", "b", "a", "b"), ("b", "a"))
