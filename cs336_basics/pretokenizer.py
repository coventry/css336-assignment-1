from collections import Counter
from collections.abc import Iterator
import math
import multiprocessing as mp
import os
from pathlib import Path
import regex
import shutil
from tempfile import mkdtemp
from typing import TextIO

EOT_TOKEN = "<|endoftext|>"
TEXT_PRETOKENIZER = regex.compile(
    r"""
      '(?:[sdmt]|ll|ve|re)          # contractions: 's, 'd, 'm, 't, 'll, 've, 're
    | [ ]?\p{L}+                    # optional leading space + letters
    | [ ]?\p{N}+                    # optional leading space + numbers
    | [ ]?[^\s\p{L}\p{N}]+          # optional leading space + punctuation/symbols/etc.
    | \s+(?!\S)                     # whitespace not followed by non-whitespace
    | \s+                           # remaining whitespace
      """,
    regex.VERBOSE,
)


def pretokenize_chunk(
    data: str,
    special_tokens: list[str],
) -> Counter[str]:
    """Return the counts of the pretokenized words in `data`.

    Special tokens do not contribute to merge statistics, so we don't count
    those here, we just treat them as hard boundaries.

    >>> import json
    >>> text = "I'm testing 123!!!\\n\\nnext<|endoftext|>"
    >>> dumps_args = dict(indent=2, sort_keys=True)
    >>> print(json.dumps(pretokenize_chunk(text, [EOT_TOKEN]), **dumps_args))
    {
      "\\n": 2,
      " 123": 1,
      " testing": 1,
      "!!!": 1,
      "'m": 1,
      "I": 1,
      "next": 1
    }
    """
    counts = Counter()
    splits = [data]
    # Split on longer special tokens first, in case any shorter ones are
    # substrings of longer ones.
    longer_to_shorter = sorted(special_tokens, key=len, reverse=True)
    for tok in longer_to_shorter:
        new_splits = []
        for s in splits:
            ns = s.split(tok)
            new_splits.extend(ns)
        splits = new_splits
    for s in splits:  # Now count the words in those splits
        for word in TEXT_PRETOKENIZER.finditer(s):
            counts[word.group()] += 1
    return counts


def iter_file_split(
    filename: Path,
    separator: str,
    *,
    chunk_size: int = 64 * 1024,
    encoding: str = "utf-8",
) -> Iterator[str]:
    """Yield the same strings as open(filename).read().split(separator)

    Without keeping the whole file in memory."""
    if not separator:
        raise ValueError("empty separator")

    with open(filename, encoding=encoding) as file:
        buffer = ""
        while chunk := file.read(chunk_size):
            buffer += chunk
            parts = buffer.split(separator)
            yield from parts[:-1]
            # This may contain either:
            #   * the unfinished current element, or
            #   * the beginning of a separator spanning two chunks.
            buffer = parts[-1]
        # Required even when empty, matching str.split().
        yield buffer


def shard_stories(
    data_file: Path,  # Path to file containing stories to split
    eot_token: str,  # end-of-text token to split stories on
    num_workers: int,  # Number of workers to split the stories among
    shard_dir: Path,  # Directory in which to put splits
    shard_fn_prefix: str,  # Filename prefix for shards
) -> list[Path]:
    """Shard stories in data_file to files for workers to process separately.

    Count the number of stories in `data_file`, as delimited by `eot_token`,
    divide that by `num_workers`, and create that many shards under
    `shard_dir`, with the stories divided approximately equally between them.

    Caller is responsible for cleaning up the shards.

    Shard filenames will be of the form {shard_fn_prefix}-n.txt

    """
    os.makedirs(shard_dir, exist_ok=True)
    num_stories = sum(1 for _ in iter_file_split(data_file, eot_token)) - 1
    stories_per_worker = math.ceil(num_stories / num_workers)
    output: TextIO = open("/dev/null")  # Dummy fileobj; fails on write
    shard_paths: list[Path] = []
    for story_idx, story in enumerate(iter_file_split(data_file, eot_token)):
        if story_idx % stories_per_worker == 0:  # Moving to a new shard
            new_fn = f"{shard_fn_prefix}-{story_idx // stories_per_worker}.txt"
            new_path = shard_dir / new_fn
            output.close()  # Ensure old shard is written to disk
            output = open(new_path, "w")
            shard_paths.append(Path(new_path))
        output.write(story + eot_token)  # Add back eot_token, removed by split
    output.close()  # Ensure final shard is written to disk prior to return
    return shard_paths


def shard_worker(args: tuple[Path, list[str]]) -> Counter[str]:
    shard_path, special_tokens = args
    return pretokenize_chunk(
        shard_path.read_text(encoding="utf-8"),
        special_tokens,
    )


def get_pretoken_counts(
    input_path: str,
    special_tokens: list[str] = [EOT_TOKEN],
) -> Counter[str]:
    output_dir = Path(mkdtemp())
    try:
        shard_paths = shard_stories(
            Path(input_path), EOT_TOKEN, mp.cpu_count(), output_dir, "shard"
        )
        aggregate_counter = Counter()
        with mp.Pool(processes=len(shard_paths)) as pool:
            tasks = [(p, special_tokens) for p in shard_paths]
            for counts in pool.imap_unordered(shard_worker, tasks):
                aggregate_counter.update(counts)
    finally:
        shutil.rmtree(output_dir)
    return aggregate_counter


if __name__ == "__main__":
    from common import VALIDATION_PATH

    counts = get_pretoken_counts(VALIDATION_PATH, [EOT_TOKEN])
    breakpoint()
    # Compare `counts` with, e.g.
    # `grep -F -o ' Suddenly' data/TinyStoriesV2-GPT4-valid.txt | wc -l`
