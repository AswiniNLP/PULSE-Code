#!/usr/bin/env python3
"""Train the SSLCL style-learning stage or the SCCG generation stage of PULSE."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import math
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pulse_core import (
    DEFAULT_MODEL, DEFAULT_STYLE_MODEL, STYLE_TO_ID,
    build_prompt, encode_prompt, create_model_and_tokenizers, load_checkpoint_model,
    read_checkpoint_metadata, resolve_device, save_checkpoint, set_seed,
)


REQUIRED_COLUMNS = ("Hatespeech", "Style", "Target", "Counterspeech")


def read_training_rows(path, target_labels=None):
    """Validate data before loading expensive models and persist the class mapping."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError("Training CSV contains duplicate column names")
        if any(not header.strip() for header in headers):
            raise ValueError("Training CSV contains an empty column name")
        missing = set(REQUIRED_COLUMNS) - set(headers)
        if missing:
            raise ValueError(f"Training CSV is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Training CSV contains no rows")
    seen_ids = set()
    for line, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV row {line}: number of values differs from the header")
        if "id" in row:
            identifier = row["id"].strip()
            if not identifier:
                raise ValueError(f"CSV row {line}: id must be nonempty")
            if identifier in seen_ids:
                raise ValueError(f"CSV row {line}: duplicate id {identifier!r}")
            seen_ids.add(identifier)
        for column in REQUIRED_COLUMNS:
            if not row.get(column) or not row[column].strip():
                raise ValueError(f"CSV row {line}: {column} must be nonempty")
        if row["Style"] not in STYLE_TO_ID:
            raise ValueError(f"CSV row {line}: unsupported style {row['Style']!r}; expected Gandhi or Mandela")
    labels = list(target_labels) if target_labels is not None else sorted({r["Target"] for r in rows})
    label_map = {label: index for index, label in enumerate(labels)}
    for row in rows:
        if row["Target"] not in label_map:
            raise ValueError(f"Target {row['Target']!r} is absent from the SSLCL checkpoint label mapping")
        row["target_label"] = label_map[row["Target"]]
    return rows, labels


class HateSpeechDataset(Dataset):
    """Prepare response-only denoising masks and padding-aware BERT inputs."""
    def __init__(self, dataset, llada_tokenizer, bert_tokenizer, model_type,
                 max_length=512, style_max_length=512):
        if model_type not in {"sslcl", "sccg"}:
            raise ValueError("model_type must be sslcl or sccg")
        self.dataset = dataset
        self.llada_tokenizer = llada_tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.model_type = model_type
        self.max_length = max_length
        self.style_max_length = style_max_length
        if max_length < 2 or style_max_length < 2:
            raise ValueError("Token sequence lengths must be at least 2")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        tokenizer = self.llada_tokenizer
        if self.model_type == "sccg":
            prefix = encode_prompt(tokenizer, row["Hatespeech"], row["Style"])
            # The style encoder sees the same prompt-only text at inference time.
            style_text = build_prompt(row["Hatespeech"], row["Style"])
        else:
            prefix = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
            style_text = row["Counterspeech"]
        if len(prefix) >= self.max_length - 1:
            raise ValueError(f"Sample {index}: prompt leaves no response space; increase --max-length or shorten Hatespeech")
        response = tokenizer.encode(row["Counterspeech"], add_special_tokens=False)
        response = response[:self.max_length - len(prefix) - 1] + [tokenizer.eos_token_id]
        ids = prefix + response
        real_length = len(ids)
        padding = self.max_length - real_length
        ids += [tokenizer.pad_token_id] * padding
        attention_mask = [1] * real_length + [0] * padding
        loss_mask = [False] * len(prefix) + [True] * len(response) + [False] * padding
        bert = self.bert_tokenizer(style_text, truncation=True, padding="max_length",
                                   max_length=self.style_max_length, return_tensors="pt")
        return {
            "llada_input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
            "bert_input_ids": bert["input_ids"].squeeze(0),
            "bert_attention_mask": bert["attention_mask"].squeeze(0),
            "leader_label": torch.tensor(STYLE_TO_ID[row["Style"]], dtype=torch.long),
            "target_label": torch.tensor(row["target_label"], dtype=torch.long),
        }


def forward_process(input_ids, mask_id, eligible_mask, eps=1e-3):
    """Mask only response tokens; prompts and padding remain intact."""
    probability = ((1 - eps) * torch.rand(input_ids.shape[0], 1, device=input_ids.device) + eps)
    probability = probability.expand_as(input_ids)
    selected = (torch.rand(input_ids.shape, device=input_ids.device) < probability) & eligible_mask
    return torch.where(selected, mask_id, input_ids), selected, probability


def diffusion_loss(logits, targets, selected, probability, eligible_mask):
    """Inverse-probability denoising loss, normalized by each response length."""
    if not selected.any():
        return logits.sum() * 0
    token_loss = F.cross_entropy(logits[selected].float(), targets[selected], reduction="none")
    lengths = eligible_mask.sum(1, keepdim=True).clamp_min(1).expand_as(targets)
    return (token_loss / probability[selected] / lengths[selected]).sum() / targets.shape[0]


def train(model, tokenizer, style_tokenizer, rows, stage, output_dir, epochs=3,
          batch_size=1, max_length=512, style_max_length=512, learning_rate=2.5e-5,
          max_steps=None, target_loss_weight=1.0, vq_loss_weight=1.0, grad_clip=1.0):
    """Train and save every epoch; return the final checkpoint directory."""
    if stage not in {"sslcl", "sccg"}:
        raise ValueError("stage must be sslcl or sccg")
    device = next(model.style_projection.parameters()).device
    if stage == "sccg":
        model.freeze_style()
        model.target_classifier.requires_grad_(True)
    else:
        model.target_classifier.requires_grad_(False)
    model.train()
    dataset = HateSpeechDataset(rows, tokenizer, style_tokenizer, stage, max_length, style_max_length)
    # Check all prompts before beginning expensive optimization or writing checkpoints.
    for index in range(len(dataset)):
        dataset[index]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    updates = 0
    last_checkpoint = None
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        epoch_updates = 0
        progress = tqdm(loader, desc=f"{stage.upper()} epoch {epoch}/{epochs}")
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            noisy, selected, probability = forward_process(
                batch["llada_input_ids"], model.metadata["mask_id"], batch["loss_mask"])
            if not selected.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            context = (torch.autocast("cuda", dtype=torch.bfloat16)
                       if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext())
            with context:
                outputs = model(noisy, bert_input_ids=batch["bert_input_ids"],
                                style_label=batch["leader_label"],
                                bert_attention_mask=batch["bert_attention_mask"],
                                attention_mask=batch["attention_mask"])
                loss = diffusion_loss(outputs.logits, batch["llada_input_ids"], selected,
                                      probability, batch["loss_mask"])
                if stage == "sslcl":
                    loss = loss + vq_loss_weight * outputs.vq_loss
                else:
                    loss = loss + target_loss_weight * F.cross_entropy(
                        outputs.target_logits.float(), batch["target_label"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; inspect inputs and numerical precision")
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), grad_clip)
            optimizer.step()
            updates += 1
            epoch_updates += 1
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", steps=updates)
            if max_steps is not None and updates >= max_steps:
                break
        if epoch_updates == 0:
            raise RuntimeError("No masked-token updates occurred; increase the data/response length or change --seed")
        metadata = dict(stage=stage, epoch=epoch, updates=updates,
                        requested_epochs=epochs, max_steps=max_steps, batch_size=batch_size, grad_clip=grad_clip,
                        average_loss=total_loss / epoch_updates, max_length=max_length,
                        style_max_length=style_max_length, learning_rate=learning_rate,
                        target_loss_weight=target_loss_weight, vq_loss_weight=vq_loss_weight)
        last_checkpoint = save_checkpoint(model, tokenizer, style_tokenizer,
                                          output_dir / f"{stage}_epoch_{epoch}", metadata)
        print(f"Checkpoint saved: {last_checkpoint} (mean loss {metadata['average_loss']:.4f})")
        if max_steps is not None and updates >= max_steps:
            break
    return last_checkpoint


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sslcl", "sccg"], required=True)
    parser.add_argument("--train-csv", required=True, help="CSV: Hatespeech,Style,Target,Counterspeech")
    parser.add_argument("--model-name", default=None, help=f"HF ID/local weights; default {DEFAULT_MODEL}")
    parser.add_argument("--style-model-name", default=None, help=f"HF ID/local BERT; default {DEFAULT_STYLE_MODEL}")
    parser.add_argument("--revision", default=None, help="Base model/code revision; pin a commit for reproducible downloads")
    parser.add_argument("--sslcl-checkpoint", default=None, help="SSLCL checkpoint directory required for SCCG")
    parser.add_argument("--output-dir", default=None, help="Default checkpoints/STAGE/CODEBOOK_DIM")
    parser.add_argument("--device", default=None, help="cpu or cuda[:index]; default auto")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="4bit")
    parser.add_argument("--codebook-dim", type=int, default=None, help="Default 768 or SSLCL checkpoint dimension")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--style-max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--vq-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer updates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-target-modules", nargs="+", default=None, help="Override detected LLaDA attention modules")
    parser.add_argument("--validate-only", action="store_true", help="Validate CSV/mapping without loading models")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "max_length", "style_max_length", "lora_rank", "lora_alpha"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.codebook_dim is not None and args.codebook_dim <= 0:
        parser.error("--codebook-dim must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if (not math.isfinite(args.learning_rate) or not math.isfinite(args.grad_clip)
            or args.learning_rate <= 0 or args.grad_clip <= 0):
        parser.error("--learning-rate and --grad-clip must be finite and positive")
    if (not math.isfinite(args.target_loss_weight) or not math.isfinite(args.vq_loss_weight)
            or args.target_loss_weight < 0 or args.vq_loss_weight < 0):
        parser.error("Loss weights must be finite and nonnegative")
    if args.stage == "sccg" and not args.sslcl_checkpoint:
        parser.error("--sslcl-checkpoint is required for SCCG")
    if args.stage == "sslcl" and args.sslcl_checkpoint:
        parser.error("--sslcl-checkpoint applies only to SCCG")
    try:
        checkpoint_metadata = read_checkpoint_metadata(args.sslcl_checkpoint) if args.sslcl_checkpoint else None
        if checkpoint_metadata and checkpoint_metadata.get("stage") != "sslcl":
            raise ValueError("SCCG initialization requires a checkpoint from the SSLCL stage")
        rows, target_labels = read_training_rows(
            args.train_csv, checkpoint_metadata["target_labels"] if checkpoint_metadata else None)
        if args.validate_only:
            print(json.dumps({"rows": len(rows), "target_labels": target_labels, "styles": sorted({r['Style'] for r in rows})}, indent=2))
            return
        device = resolve_device(args.device)
        set_seed(args.seed)
        if checkpoint_metadata:
            if args.codebook_dim is not None and args.codebook_dim != checkpoint_metadata["codebook_dim"]:
                raise ValueError("--codebook-dim differs from the SSLCL checkpoint")
            inherited_revisions = {checkpoint_metadata.get("revision"),
                                   checkpoint_metadata.get("resolved_model_revision")} - {None}
            if args.revision is not None and args.revision not in inherited_revisions:
                raise ValueError("SCCG inherits the pinned base revision saved in its SSLCL checkpoint")
            model, tokenizer, style_tokenizer, metadata = load_checkpoint_model(
                args.sslcl_checkpoint, device, args.quantization,
                model_name=args.model_name, style_model_name=args.style_model_name, is_trainable=True)
        else:
            model, tokenizer, style_tokenizer = create_model_and_tokenizers(
                args.model_name or DEFAULT_MODEL, args.style_model_name or DEFAULT_STYLE_MODEL,
                device=device, quantization=args.quantization, codebook_dim=args.codebook_dim or 768,
                target_labels=target_labels, lora_target_modules=args.lora_target_modules,
                lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, revision=args.revision)
        if args.style_max_length > model.style_encoder.config.max_position_embeddings:
            raise ValueError("--style-max-length exceeds the style encoder's position limit")
        base_config = model.llada_model.config
        position_limit = (getattr(base_config, "max_sequence_length", None)
                          or getattr(base_config, "max_position_embeddings", None)
                          or getattr(base_config, "n_positions", None))
        if position_limit and args.max_length > position_limit:
            raise ValueError(f"--max-length exceeds the base model position limit ({position_limit})")
        model.metadata["seed"] = args.seed
        model.metadata["training_csv_sha256"] = hashlib.sha256(Path(args.train_csv).read_bytes()).hexdigest()
        model.metadata["package_versions"] = {
            package: importlib.metadata.version(package) for package in ("torch", "transformers", "peft")
        }
        model.metadata["quantization"] = args.quantization
        model.metadata["batch_size"] = args.batch_size
        model.metadata["grad_clip"] = args.grad_clip
        history = list(model.metadata.get("training_history", []))
        history.append(dict(stage=args.stage, csv_sha256=model.metadata["training_csv_sha256"],
                            seed=args.seed, batch_size=args.batch_size, quantization=args.quantization,
                            grad_clip=args.grad_clip, package_versions=model.metadata["package_versions"]))
        model.metadata["training_history"] = history
        output_dir = args.output_dir or f"checkpoints/{args.stage}/{model.metadata['codebook_dim']}"
        train(model, tokenizer, style_tokenizer, rows, args.stage, output_dir,
              epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length,
              style_max_length=args.style_max_length, learning_rate=args.learning_rate,
              max_steps=args.max_steps, target_loss_weight=args.target_loss_weight,
              vq_loss_weight=args.vq_loss_weight, grad_clip=args.grad_clip)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
