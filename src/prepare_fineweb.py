"""Fineweb-Edu preparation: train a 32K BPE on web text, pre-tokenize shards.

Memory-safe: processes one shard at a time (read -> split -> encode -> free),
never holds more than a single shard's text in RAM.
"""
from __future__ import annotations

import argparse
import glob
import gc
import json
import multiprocessing as mp
import os
import subprocess
import time

import numpy as np

from tokenizer import Tokenizer, train_bpe

EOS = "<|endoftext|>"

_TOK: Tokenizer | None = None


def _init_worker(tokenizer_path: str):
    global _TOK
    _TOK = Tokenizer.load(tokenizer_path)


def _encode_chunk(text: str) -> list[int]:
    return _TOK.encode(text, allow_special=False)


def encode_docs(chunks: list[str], out_file, tok_path: str, workers: int,
                window: int = 8000) -> int:
    """Windowed map: Pool.imap pickles the WHOLE iterable eagerly (~GBs for web
    text) — bounded windows keep the main process flat."""
    from itertools import islice
    total = 0
    with mp.Pool(workers, initializer=_init_worker, initargs=(tok_path,)) as pool:
        it = iter(chunks)
        while True:
            batch = list(islice(it, window))
            if not batch:
                break
            for ids in pool.map(_encode_chunk, batch, chunksize=32):
                ids.append(_TOK._special_to_id[EOS])
                out_file.write(np.asarray(ids, dtype=np.uint16).tobytes())
                total += len(ids)
    return total


def _training_active() -> bool:
    """Defer to GPU training: prep's memory balloon must not OOM the box."""
    try:
        out = subprocess.run(["pgrep", "-f", "trai[n].py"],
                             capture_output=True, text=True)
        return out.returncode == 0
    except Exception:
        return False


def main():
    if _training_active():
        print("[guard] GPU training active - deferring prep (run me alone)", flush=True)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--fineweb-dir", default=os.path.expanduser("~/llm-lab/data/fineweb"))
    ap.add_argument("--data-dir", default=os.path.expanduser("~/llm-lab/data"))
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--bpe-sample-bytes", type=int, default=335_544_320)  # 320MB
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    args = ap.parse_args()

    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(args.fineweb_dir, "*.parquet")))
    assert files, f"no parquet files in {args.fineweb_dir}"
    print(f"[fineweb] {len(files)} shards", flush=True)

    tok_path = os.path.join(args.data_dir, "tokenizer_web_32k.json")
    if not os.path.exists(tok_path):
        per_file = args.bpe_sample_bytes // len(files)
        sample_parts = []
        for fp in files:
            with open(fp, "rb") as f:
                sample_parts.append(f.read(per_file).decode("utf-8", errors="ignore"))
        t0 = time.time()
        vocab, merges = train_bpe("\n".join(sample_parts), args.vocab_size, [EOS])
        tok = Tokenizer(vocab=vocab, merges=merges, special_tokens=[EOS])
        tok.save(tok_path)
        del sample_parts, vocab, merges
        gc.collect()
        print(f"[bpe] done: {len(tok.merges)} merges in {time.time()-t0:.0f}s", flush=True)
        demo = "The transformer architecture revolutionized natural language processing."
        ids = tok.encode(demo)
        print(f"[bpe] demo: {len(demo)} chars -> {len(ids)} tokens, roundtrip {tok.decode(ids) == demo}", flush=True)
    _init_worker(tok_path)
    print("[bpe] tokenizer ready (cached)", flush=True)

    import hashlib

    totals = dict(train=0, valid=0)
    out_train = open(os.path.join(args.data_dir, "tokens_web_train.bin"), "wb")
    out_valid = open(os.path.join(args.data_dir, "tokens_web_valid.bin"), "wb")

    def split_doc(d: str) -> str:
        # deterministic ~1% valid split, hash-based so it works while streaming
        h = int(hashlib.md5(d[:4096].encode("utf-8", "ignore")).hexdigest(), 16)
        return "valid" if h % 100 == 0 else "train"

    for fp in files:
        t0 = time.time()
        pf = pq.ParquetFile(fp)
        n_docs = n_tr = n_va = 0
        for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
            docs = [d for d in batch.column("text").to_pylist() if 200 < len(d) < 200_000]
            for split in ("train", "valid"):
                part = [d for d in docs if split_doc(d) == split]
                if not part:
                    continue
                n = encode_docs(part, out_train if split == "train" else out_valid,
                                tok_path, args.workers)
                totals[split] += n
                if split == "train":
                    n_tr += n
                else:
                    n_va += n
            n_docs += len(docs)
            del docs, batch
        del pf
        gc.collect()
        print(f"  {os.path.basename(fp)}: {n_docs:,} docs -> +{n_tr:,} train / +{n_va:,} valid "
              f"({time.time()-t0:.0f}s; running {totals['train']:,}/{totals['valid']:,})", flush=True)

    out_train.close()
    out_valid.close()
    meta = dict(train_tokens=totals["train"], valid_tokens=totals["valid"],
                vocab_size=args.vocab_size, tokenizer="tokenizer_web_32k.json")
    with open(os.path.join(args.data_dir, "meta_web.json"), "w") as f:
        json.dump(meta, f)
    print(f"[meta] {meta}", flush=True)
    print("FINEWEB_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
