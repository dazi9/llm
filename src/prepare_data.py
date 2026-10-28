"""Data preparation: train BPE on a sample of TinyStories, then pre-tokenize
the full corpus into uint16 memmap-able bins using multiprocessing.

  python src/prepare_data.py --data-dir ~/llm-lab/data
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from tokenizer import Tokenizer, train_bpe

EOS = "<|endoftext|>"

_TOK: Tokenizer | None = None


def _init_worker(tokenizer_path: str):
    global _TOK
    _TOK = Tokenizer.load(tokenizer_path)


def _encode_chunk(chunk: bytes) -> list[int]:
    text = chunk.decode("utf-8", errors="ignore")
    return _TOK.encode(text, allow_special=False)


def encode_corpus(txt_path: str, out_path: str, tokenizer_path: str, workers: int) -> int:
    t0 = time.time()
    _init_worker(tokenizer_path)  # also load in the main process (for EOS id)
    with open(txt_path, "rb") as f:
        raw = f.read()
    chunks = [c for c in raw.split(EOS.encode()) if c.strip()]
    del raw
    n_chunks = len(chunks)
    print(f"  {os.path.basename(txt_path)}: {n_chunks:,} stories", flush=True)

    total = 0
    with mp.Pool(workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
        with open(out_path, "wb") as out:
            for i, ids in enumerate(pool.imap(_encode_chunk, chunks, chunksize=256)):
                ids.append(_TOK._special_to_id[EOS])
                arr = np.asarray(ids, dtype=np.uint16)
                out.write(arr.tobytes())
                total += len(ids)
                if (i + 1) % 200_000 == 0:
                    done = (i + 1) / n_chunks
                    eta = (time.time() - t0) / done * (1 - done)
                    print(f"  {i+1:,}/{n_chunks:,} stories ({done*100:.0f}%, ~{eta:.0f}s left)", flush=True)
    print(f"  wrote {total:,} tokens -> {out_path} in {time.time()-t0:.0f}s", flush=True)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/llm-lab/data"))
    ap.add_argument("--vocab-size", type=int, default=10000)
    ap.add_argument("--bpe-sample-bytes", type=int, default=262_144_000)  # 250MB
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    args = ap.parse_args()

    train_txt = os.path.join(args.data_dir, "TinyStories-train.txt")
    valid_txt = os.path.join(args.data_dir, "TinyStories-valid.txt")
    tok_path = os.path.join(args.data_dir, "tokenizer.json")

    if not os.path.exists(tok_path):
        print(f"[bpe] training on first {args.bpe_sample_bytes/1e6:.0f}MB sample...", flush=True)
        with open(train_txt, "rb") as f:
            sample = f.read(args.bpe_sample_bytes).decode("utf-8", errors="ignore")
        t0 = time.time()
        vocab, merges = train_bpe(sample, args.vocab_size, [EOS])
        tok = Tokenizer(vocab=vocab, merges=merges, special_tokens=[EOS])
        tok.save(tok_path)
        print(f"[bpe] done: vocab {tok.vocab_size}, {len(merges)} merges, {time.time()-t0:.0f}s", flush=True)
        demo = "Once upon a time, there was a little girl named Lily."
        ids = tok.encode(demo)
        print(f"[bpe] demo: {demo!r} -> {len(ids)} tokens, roundtrip ok: {tok.decode(ids) == demo}", flush=True)

    out_train = os.path.join(args.data_dir, "tokens_train.bin")
    out_valid = os.path.join(args.data_dir, "tokens_valid.bin")
    if not os.path.exists(out_train):
        n_tr = encode_corpus(train_txt, out_train, tok_path, args.workers)
    else:
        n_tr = os.path.getsize(out_train) // 2
    if not os.path.exists(out_valid):
        n_va = encode_corpus(valid_txt, out_valid, tok_path, args.workers)
    else:
        n_va = os.path.getsize(out_valid) // 2

    meta = {"train_tokens": n_tr, "valid_tokens": n_va, "vocab_size": args.vocab_size}
    with open(os.path.join(args.data_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[meta] {meta}", flush=True)
    print("PREPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
