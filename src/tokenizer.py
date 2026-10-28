"""Byte-level BPE tokenizer, CS336 assignment-1 style.

Training: GPT-2 pre-tokenization regex, then incremental pair counting with a
lazy max-heap (only words containing the merged pair are re-scored).
Encoding: cached per-word BPE, always merging the lowest-rank pair first.

ID layout: 0..255 raw bytes, then merges (256..), then special tokens last.
"""
from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)
_PRETOKEN_RE = re.compile(GPT2_SPLIT_PATTERN)


def _merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(
    text: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train BPE on `text` up to `vocab_size` (incl. 256 bytes + specials)."""
    special_tokens = special_tokens or []
    n_merges = vocab_size - 256 - len(special_tokens)
    if n_merges <= 0:
        raise ValueError("vocab_size must exceed 256 + len(special_tokens)")

    # Remove special tokens from the training text, then pre-tokenize.
    if special_tokens:
        pat = re.compile("|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True)))
        text = pat.sub("", text)
    word_counts: Counter[bytes] = Counter()
    for w in _PRETOKEN_RE.findall(text):
        word_counts[w.encode("utf-8")] += 1

    # words[i] = [ids, count]; index pairs -> counts and word membership.
    words: list[list] = []
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, (wb, count) in enumerate(word_counts.items()):
        ids = list(wb)
        words.append([ids, count])
        for pair in zip(ids, ids[1:]):
            pair_counts[pair] += count
            pair_to_words[pair].add(wi)

    heap: list[tuple[int, tuple[int, int]]] = [(-c, p) for p, c in pair_counts.items() if c > 0]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []
    while len(merges) < n_merges and heap:
        # Lazy pop: skip entries whose count no longer matches current state.
        negc, pair = heapq.heappop(heap)
        if pair_counts.get(pair, 0) != -negc:
            continue
        new_id = 256 + len(merges)
        a, b = pair
        merges.append((bytes([a]) if a < 256 else None, bytes([b]) if b < 256 else None))
        # Resolve byte representations lazily at the end; store numeric pair too.
        merges[-1] = pair  # numeric for now, converted below
        for wi in list(pair_to_words[pair]):
            ids, count = words[wi]
            if pair not in zip(ids, ids[1:]):
                continue  # stale membership
            for old in zip(ids, ids[1:]):
                pair_counts[old] -= count
            ids = _merge_ids(ids, pair, new_id)
            words[wi][0] = ids
            for new_pair in zip(ids, ids[1:]):
                pair_counts[new_pair] += count
                pair_to_words[new_pair].add(wi)
                heapq.heappush(heap, (-pair_counts[new_pair], new_pair))
        pair_counts[pair] = 0

    # Convert numeric pairs to byte pairs.
    byte_vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    byte_merges: list[tuple[bytes, bytes]] = []
    for i, (a, b) in enumerate(merges):
        byte_merges.append((byte_vocab[a], byte_vocab[b]))
        byte_vocab[256 + i] = byte_vocab[a] + byte_vocab[b]
    return byte_vocab, byte_merges


@dataclass
class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] = field(default_factory=list)
    _ranks: dict[tuple[int, int], int] = field(default_factory=dict, init=False, repr=False)
    _special_to_id: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _word_cache: dict[bytes, list[int]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._ranks = {(self._id_of(a), self._id_of(b)): i for i, (a, b) in enumerate(self.merges)}
        base = 256 + len(self.merges)
        self._special_to_id = {s: base + i for i, s in enumerate(self.special_tokens)}
        self.vocab_size = base + len(self.special_tokens)

    def _id_of(self, b: bytes) -> int:
        # Byte string -> id via vocab reverse lookup (cached per instance).
        if not hasattr(self, "_byte_to_id"):
            self._byte_to_id = {v: k for k, v in self.vocab.items()}
        return self._byte_to_id[b]

    def _bpe_word(self, wb: bytes) -> list[int]:
        cached = self._word_cache.get(wb)
        if cached is not None:
            return cached
        ids = list(wb)
        while len(ids) > 1:
            best_rank: int | None = None
            best_pair: tuple[int, int] | None = None
            for pair in zip(ids, ids[1:]):
                r = self._ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            ids = _merge_ids(ids, best_pair, 256 + best_rank)
        self._word_cache[wb] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        out: list[int] = []
        if allow_special and self.special_tokens:
            # capture group keeps the special-token text in split output
            pat = re.compile(
                "(" + "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            )
            parts = pat.split(text)
            specials = set(self.special_tokens)
            for part in parts:
                if not part:
                    continue
                if part in specials:
                    out.append(self._special_to_id[part])
                else:
                    out.extend(self._encode_plain(part))
        else:
            out.extend(self._encode_plain(text))
        return out

    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        for w in _PRETOKEN_RE.findall(text):
            out.extend(self._bpe_word(w.encode("utf-8")))
        return out

    def decode(self, ids: list[int]) -> str:
        id_to_special = {v: k for k, v in self._special_to_id.items()}
        pieces: list[bytes] = []
        for i in ids:
            if i in id_to_special:
                pieces.append(id_to_special[i].encode("utf-8"))
            else:
                pieces.append(self.vocab[i])
        return b"".join(pieces).decode("utf-8", errors="replace")

    def save(self, path: str):
        obj = {
            "merges": [[a.decode("latin-1"), b.decode("latin-1")] for a, b in self.merges],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in obj["merges"]]
        vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(merges):
            vocab[256 + i] = a + b
        specials = obj.get("special_tokens", [])
        base = 256 + len(merges)
        for j, s in enumerate(specials):
            vocab[base + j] = s.encode("utf-8")
        return cls(vocab=vocab, merges=merges, special_tokens=specials)
