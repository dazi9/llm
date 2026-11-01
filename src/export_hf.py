"""Export a trained checkpoint to HuggingFace LLaMA-style format, ready for
llama.cpp conversion (convert_hf_to_llama.py) and GGUF quantization.

  python src/export_hf.py --ckpt ~/llm-lab/runs/main/model_final.pt --out ~/llm-lab/export/v1
Then (on any machine, Intel/AMD alike):
  pip install gguf torch  # converter deps
  python llama.cpp/convert_hf_to_llama.py <out> --outfile model-f16.gguf
  llama-quantize model-f16.gguf model-q4_0.gguf q4_0   # ~130MB
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from model import TransformerLM
from tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = dict(ckpt["config"])
    tok = Tokenizer.load(ckpt["tokenizer"])

    model = TransformerLM(
        vocab_size=config["vocab_size"], d_model=config["d_model"],
        n_layers=config["n_layers"], n_heads=config["n_heads"],
        ctx=config["ctx"], d_ff=config.get("d_ff"), attn="sdpa",
        qk_norm=config.get("qk_norm", False),
        n_kv_heads=config.get("n_kv_heads"),
        use_rope=config.get("use_rope", True))
    model.load_state_dict(ckpt["model"])
    model.eval()

    os.makedirs(args.out, exist_ok=True)
    # Save safetensors via torch.save→safetensors shim (torch>=2.x supports it natively
    # through safetensors package if present; fall back to .bin which convert_hf_to_llama
    # also accepts).
    try:
        from safetensors.torch import save_file
        sd = {k: v.contiguous() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(args.out, "model.safetensors"),
                  metadata={"format": "pt"})
    except ImportError:
        torch.save(model.state_dict(), os.path.join(args.out, "pytorch_model.bin"))

    # LLaMA-style config: our architecture is LLaMA-compatible when qk_norm is OFF
    # (RMSNorm + RoPE + SwiGLU + untied embeddings + no biases).
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": config["d_model"],
        "intermediate_size": config["d_ff"],
        "num_hidden_layers": config["n_layers"],
        "num_attention_heads": config["n_heads"],
        "num_key_value_heads": config.get("n_kv_heads") or config["n_heads"],
        "vocab_size": config["vocab_size"],
        "max_position_embeddings": config["ctx"],
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "tie_word_embeddings": bool(config.get("tie_embeddings")),
        "attention_bias": False,
        "bos_token_id": tok._special_to_id.get("<|endoftext|>"),
        "eos_token_id": tok._special_to_id.get("<|endoftext|>"),
    }
    if config.get("qk_norm"):
        print("[warn] qk_norm is enabled: LLaMA GGUF conversion will NOT reproduce it. "
              "Export the non-qk-norm checkpoint (v1) or wait for qk-norm arch mapping.",
              flush=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)

    # tokenizer: emit a minimal sentencepiece-style vocab file list that
    # convert_hf_to_llama.py's tokenizer fallback can use; the GGUF stores
    # tokens directly, so we also write a plain token list.
    with open(os.path.join(args.out, "tokens.json"), "w") as f:
        json.dump([tok.vocab[i].decode("utf-8", errors="replace")
                   for i in range(tok.vocab_size)], f)

    print(f"exported -> {args.out}")
    print("next: python convert_hf_to_llama.py <out> --outfile model-f16.gguf")


if __name__ == "__main__":
    main()
