"""Generate counterspeech from a standalone PULSE SCCG checkpoint."""

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import tempfile


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", "--input-file", required=True,
                        help="CSV with Hatespeech and Style; other columns are preserved")
    parser.add_argument("--output_file", "--output-file", required=True,
                        help="Destination CSV, with generated text in CS")
    parser.add_argument("--checkpoint", "--sccg_checkpoint_path", required=True,
                        help="Standalone SCCG checkpoint directory from pulse_train.py")
    parser.add_argument("--model_name", "--model-name", default=None,
                        help="Optional base model location override (same architecture/weights)")
    parser.add_argument("--style_model_name", "--style-model-name", default=None,
                        help="Optional BERT model location override")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--gen_length", "--gen-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_length", "--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", "--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", choices=["low_confidence", "random"],
                        default="low_confidence")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def read_inputs(path):
    """Validate all rows before any model is loaded; preserve IDs/references."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError("input CSV contains duplicate column names")
        missing = {"Hatespeech", "Style"} - set(columns)
        if missing:
            raise ValueError(f"input CSV is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("input CSV is empty")
    ids = set()
    for number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"input CSV row {number} has the wrong number of fields")
        if not row["Hatespeech"].strip():
            raise ValueError(f"input CSV row {number} has empty Hatespeech")
        row["Style"] = row["Style"].strip()
        if row["Style"] not in {"Gandhi", "Mandela"}:
            raise ValueError(f"input CSV row {number}: Style must be Gandhi or Mandela")
        if "id" in columns:
            if not row["id"].strip() or row["id"] in ids:
                raise ValueError(f"input CSV row {number}: id must be nonempty and unique")
            ids.add(row["id"])
    return rows, columns


def write_predictions(path, rows, columns):
    """Atomically write the complete result so failed runs leave no partial CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns)
    if "CS" not in fieldnames:
        fieldnames.append("CS")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                                         dir=output.parent, prefix=".predictions-",
                                         suffix=".csv", delete=False) as handle:
            temporary_path = handle.name
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, output)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def select_device(value):
    import torch
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise ValueError("device must be auto, cpu, cuda, or cuda:N") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, cuda, or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but no CUDA device is available")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is unavailable")
        torch.cuda.set_device(index)
    return device


def generate_model_response(model, llada_tokenizer, bert_tokenizer, hatespeech,
                            style, metadata, device, gen_length=256, steps=128,
                            block_length=32, temperature=0.0, cfg_scale=0.0,
                            remasking="low_confidence"):
    import torch
    from generate_vq import generate
    from pulse_core import STYLE_TO_ID, build_prompt, encode_prompt

    prompt_ids = encode_prompt(llada_tokenizer, hatespeech, style)
    if not prompt_ids:
        raise ValueError("tokenizer produced an empty prompt")
    max_length = metadata["max_length"]
    if len(prompt_ids) + gen_length > max_length:
        raise ValueError(f"prompt ({len(prompt_ids)} tokens) plus gen_length ({gen_length}) "
                         f"exceeds checkpoint max_length ({max_length}); shorten the input "
                         "or reduce gen_length")
    llada_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    # The same prompt-only BERT input is used for SCCG training and generation.
    bert_encoding = bert_tokenizer(
        build_prompt(hatespeech, style), return_tensors="pt", truncation=True,
        max_length=metadata["style_max_length"], padding=True,
    )
    bert_input_ids = bert_encoding["input_ids"].to(device)
    bert_attention_mask = bert_encoding.get("attention_mask")
    if bert_attention_mask is not None:
        bert_attention_mask = bert_attention_mask.to(device)
    labels = torch.tensor([STYLE_TO_ID[style]], dtype=torch.long, device=device)
    output = generate(
        model, llada_input_ids, bert_input_ids=bert_input_ids,
        bert_attention_mask=bert_attention_mask, style_label=labels,
        attention_mask=torch.ones_like(llada_input_ids), mask_id=metadata["mask_id"],
        steps=steps, gen_length=gen_length, block_length=block_length,
        temperature=temperature, cfg_scale=cfg_scale, remasking=remasking,
    )
    generated_ids = output[0, llada_input_ids.shape[1]:].tolist()
    # Diffusion fills a fixed block, so explicitly stop at the first EOS/EOT.
    stop_ids = set()
    for value in (llada_tokenizer.eos_token_id,
                  getattr(model.llada_model.config, "eos_token_id", None)):
        if isinstance(value, int):
            stop_ids.add(value)
        elif isinstance(value, (list, tuple)):
            stop_ids.update(value)
    for index, token in enumerate(generated_ids):
        if token in stop_ids:
            generated_ids = generated_ids[:index]
            break
    return llada_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main(args=None):
    if args is None:
        args = parse_arguments()
    if Path(args.input_file).resolve() == Path(args.output_file).resolve():
        raise ValueError("input_file and output_file must be different paths")
    rows, columns = read_inputs(args.input_file)
    # Validate before loading model weights. This copy keeps --help and CSV
    # validation usable even when optional ML dependencies are not installed.
    if min(args.steps, args.gen_length, args.block_length) <= 0:
        raise ValueError("steps, gen_length, and block_length must be positive")
    if args.gen_length % args.block_length:
        raise ValueError("gen_length must be divisible by block_length")
    blocks = args.gen_length // args.block_length
    if args.steps % blocks or args.steps < blocks or args.steps > args.gen_length:
        raise ValueError("steps must be divisible by the number of blocks, at least "
                         "that number, and no greater than gen_length")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(args.cfg_scale) or args.cfg_scale < 0:
        raise ValueError("cfg_scale must be finite and nonnegative")
    from pulse_core import load_checkpoint_model, read_checkpoint_metadata, set_seed
    device = select_device(args.device)
    if args.quantization == "4bit" and device.type != "cuda":
        raise ValueError("4bit quantization requires a CUDA device; use --quantization none on CPU")
    if read_checkpoint_metadata(args.checkpoint).get("stage") != "sccg":
        raise ValueError("inference requires a stage sccg checkpoint")
    set_seed(args.seed)
    model, llada_tokenizer, bert_tokenizer, metadata = load_checkpoint_model(
        args.checkpoint, device, quantization=args.quantization,
        model_name=args.model_name, style_model_name=args.style_model_name,
    )
    model.eval()
    for index, row in enumerate(rows, start=1):
        try:
            row["CS"] = generate_model_response(
                model, llada_tokenizer, bert_tokenizer, row["Hatespeech"],
                row["Style"], metadata, device, args.gen_length, args.steps,
                args.block_length, args.temperature, args.cfg_scale, args.remasking,
            )
        except Exception as error:
            raise RuntimeError(f"generation failed at input CSV row {index + 1}: {error}") from error
        print(f"Generated {index}/{len(rows)}", flush=True)
    write_predictions(args.output_file, rows, columns)
    print(f"Saved {len(rows)} predictions to {args.output_file}")
    return args.output_file


if __name__ == "__main__":
    try:
        main()
    except ImportError as error:
        print(f"Dependency error: {error}. Install requirements.txt in a clean environment.",
              file=sys.stderr)
        sys.exit(1)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
