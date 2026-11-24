#!/usr/bin/env python3
"""Exercise the real CLIs with tiny locally initialized models; no downloads.

The default tiny GPT-2 is a plumbing fixture, not a pretrained diffusion model.
Reviewed local LLaDA source can be selected with --llada-source-dir. Passing
either check establishes pipeline connectivity, not counterspeech quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_tiny_models(output: Path, llada_source: Path | None = None):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        BertConfig,
        BertModel,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
    )

    torch.manual_seed(42)
    torch.set_num_threads(1)
    special = ["[PAD]", "[UNK]", "[MASK]", "<BOS>", "<EOS>", "[CLS]", "[SEP]"]
    texts = " ".join((ROOT / "examples" / name).read_text(encoding="utf-8")
                     for name in ("train.csv", "test.csv"))
    texts += " Task: Generate counterspeech for the hatespeech. Hatespeech: Style: Counterspeech:"
    vocabulary = special + sorted(set(re.findall(r"\w+|[^\w\s]", texts)) - set(special))
    tokenizer_impl = Tokenizer(WordLevel({word: i for i, word in enumerate(vocabulary)},
                                         unk_token="[UNK]"))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl, pad_token="[PAD]", unk_token="[UNK]",
        mask_token="[MASK]", bos_token="<BOS>", eos_token="<EOS>",
        cls_token="[CLS]", sep_token="[SEP]", model_max_length=128,
    )
    base_path, style_path = output / "base_model", output / "style_model"
    if llada_source:
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(str(llada_source), trust_remote_code=True,
                                            local_files_only=True)
        config.update(dict(
            d_model=16, n_heads=2, n_kv_heads=2, n_layers=1, mlp_hidden_size=32,
            vocab_size=len(tokenizer), embedding_size=len(tokenizer),
            max_sequence_length=128, init_device="cpu", flash_attention=False,
            attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id,
        ))
        AutoModel.from_config(config, trust_remote_code=True).save_pretrained(base_path)
    else:
        config = GPT2Config(
            vocab_size=len(tokenizer), n_positions=128, n_ctx=128,
            n_embd=16, n_layer=1, n_head=2,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        GPT2LMHeadModel(config).save_pretrained(base_path)
    tokenizer.save_pretrained(base_path)
    BertModel(BertConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
    )).save_pretrained(style_path)
    tokenizer.save_pretrained(style_path)
    return base_path, style_path


def run_step(name, arguments, environment, records):
    command = [sys.executable, *map(str, arguments)]
    print(f"Running {name}...", flush=True)
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    records.append({"step": name, "command": command, "exit_code": result.returncode})
    if result.returncode:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    print(result.stdout.strip(), flush=True)


def latest_checkpoint(directory, stage):
    paths = [p.parent for p in directory.rglob("config.json")
             if json.loads(p.read_text()).get("stage") == stage]
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one {stage} checkpoint under {directory}")
    return paths[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/smoke")
    parser.add_argument("--llada-source-dir", type=Path,
                        help="Optional reviewed local official LLaDA config/Python source; no weights needed")
    args = parser.parse_args(argv)
    if args.llada_source_dir:
        args.llada_source_dir = args.llada_source_dir.resolve()
        required = ("config.json", "configuration_llada.py", "modeling_llada.py")
        if any(not (args.llada_source_dir / name).is_file() for name in required):
            parser.error("--llada-source-dir must contain config.json, configuration_llada.py, modeling_llada.py")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", HF_HOME=str(output / "hf_cache"))
    # These apply to fixture construction too, before importing Transformers.
    os.environ.update({key: environment[key] for key in (
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HOME")})
    records = []
    try:
        base_path, style_path = build_tiny_models(output, args.llada_source_dir)
        common = ["pulse_train.py", "--train-csv", ROOT / "examples/train.csv",
                  "--device", "cpu", "--quantization", "none", "--epochs", "1",
                  "--batch-size", "2", "--max-length", "128", "--style-max-length", "64",
                  "--max-steps", "1", "--codebook-dim", "8"]
        lora_options = [] if args.llada_source_dir else ["--lora-target-modules", "c_attn"]
        run_step("SSLCL training", [*common, "--stage", "sslcl", "--model-name", base_path,
                 "--style-model-name", style_path, *lora_options,
                 "--output-dir", output / "sslcl"], environment, records)
        sslcl = latest_checkpoint(output / "sslcl", "sslcl")
        run_step("SCCG training", [*common, "--stage", "sccg", "--sslcl-checkpoint", sslcl,
                 "--output-dir", output / "sccg"], environment, records)
        sccg = latest_checkpoint(output / "sccg", "sccg")
        predictions = output / "predictions.csv"
        run_step("Reload and generate", ["pulse_inference.py", "--input_file",
                 ROOT / "examples/test.csv", "--output_file", predictions,
                 "--checkpoint", sccg, "--device", "cpu", "--gen_length", "8",
                 "--steps", "4", "--block_length", "4"], environment, records)
        run_step("Evaluate generated CSV", ["metric.py", "--input_file", predictions,
                 "--ground_truth_file", ROOT / "examples/test.csv", "--metrics", "basic",
                 "--output_dir", output / "metrics"], environment, records)
        with predictions.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 2 or [row["id"] for row in rows] != ["test-001", "test-002"]:
            raise RuntimeError("Inference did not preserve all input IDs")
        fixture = "random tiny official LLaDA + BERT, CPU" if args.llada_source_dir else "random tiny GPT-2 + BERT, CPU"
        report = {"status": "passed", "fixture": fixture,
                  "scope": "Pipeline connectivity only; no model quality or GPU validation",
                  "steps": records, "predictions": str(predictions)}
        (output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS: both training stages, reload, inference, evaluation. Report: {output / 'smoke_report.json'}")
    except Exception as exc:
        (output / "smoke_report.json").write_text(json.dumps(
            {"status": "failed", "error": str(exc), "steps": records}, indent=2) + "\n")
        print(f"Smoke pipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
