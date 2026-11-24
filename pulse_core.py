"""Shared PULSE architecture, prompting and portable checkpoint format.

Base model weights are referenced, never duplicated in checkpoints. Checkpoints
contain the LoRA adapter and every learned PULSE component, including BERT.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

STYLE_TO_ID = {"Gandhi": 0, "Mandela": 1}
CHECKPOINT_VERSION = 1
DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_STYLE_MODEL = "bert-base-uncased"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(hatespeech: str, style: str) -> str:
    if style not in STYLE_TO_ID:
        raise ValueError(f"Unsupported style {style!r}; expected one of {list(STYLE_TO_ID)}")
    return ("Task: Generate counterspeech for the hatespeech.\n\n"
            f"Hatespeech:\n{hatespeech}\n\nStyle: {style}\n\nCounterspeech:")


def encode_prompt(tokenizer, hatespeech: str, style: str) -> list[int]:
    """Use exactly the same prompt tokens in SCCG training and inference."""
    prompt = build_prompt(hatespeech, style)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True,
        )
    prefix = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    return prefix + tokenizer.encode(prompt, add_special_tokens=False)


def resolve_mask_id(tokenizer, model=None) -> int:
    """Preserve LLaDA's reserved mask ID instead of replacing its vocabulary."""
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        vocab = tokenizer.get_vocab()
        mask_id = vocab.get("[MASK]", vocab.get("<|mdm_mask|>"))
    if mask_id is None:
        raise ValueError("Tokenizer must provide its native mask token; no new tokens are added.")
    if model is not None and mask_id >= model.get_input_embeddings().num_embeddings:
        raise ValueError("Mask token is outside the base model vocabulary.")
    return int(mask_id)


def masked_mean(hidden, attention_mask=None):
    if attention_mask is None:
        return hidden.mean(dim=1)
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs, style_labels=None):
        if style_labels is None:
            distances = (inputs.square().sum(1, keepdim=True)
                         + self.embedding.weight.square().sum(1)
                         - 2 * inputs @ self.embedding.weight.t())
            style_labels = distances.argmin(dim=1)
        style_labels = torch.as_tensor(style_labels, device=inputs.device, dtype=torch.long).reshape(-1)
        quantized = self.embedding(style_labels)
        loss = (F.mse_loss(quantized, inputs.detach())
                + self.commitment_cost * F.mse_loss(quantized.detach(), inputs))
        return loss, inputs + (quantized - inputs).detach(), style_labels


class PerStyM(nn.Module):
    def __init__(self, hidden_size, num_layers=3):
        super().__init__()
        self.transform_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_size, 2 * hidden_size),
                          nn.GELU(), nn.LayerNorm(2 * hidden_size))
            for _ in range(num_layers)
        ])
        self.semantic_gate = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.Sigmoid())
        self.style_gate = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.Sigmoid())

    def forward(self, zs, ef):
        ef = ef.unsqueeze(1).expand_as(zs)
        combined = torch.cat([zs, ef], dim=-1)
        for layer in self.transform_layers:
            combined = combined + layer(combined)
        semantic_gate = self.semantic_gate(combined)
        style_gate = self.style_gate(combined)
        z_sem = (1 - semantic_gate) * zs + semantic_gate * combined[..., :zs.size(-1)]
        z_style = (1 - style_gate) * zs + style_gate * ef
        return z_sem + z_style


class PulseModel(nn.Module):
    """Shared train/inference graph; style labels select fixed codebook entries."""
    def __init__(self, llada_model, style_encoder, codebook_dim=768, num_targets=1):
        super().__init__()
        self.llada_model = llada_model
        self.style_encoder = style_encoder
        config = llada_model.config
        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
        if hidden_size is None:
            raise ValueError("Base model config must expose hidden_size or d_model")
        style_hidden_size = style_encoder.config.hidden_size
        self.pre_quant_projection = (nn.Linear(style_hidden_size, codebook_dim)
                                     if style_hidden_size != codebook_dim else nn.Identity())
        self.style_quantizer = VectorQuantizer(len(STYLE_TO_ID), codebook_dim)
        self.style_projection = nn.Linear(codebook_dim, hidden_size)
        self.fusion = PerStyM(hidden_size)
        vocab_size = llada_model.get_input_embeddings().num_embeddings
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.target_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size),
            nn.GELU(), nn.Linear(hidden_size, num_targets),
        )
        self.frozen_style = False
        self.metadata = {}

    def freeze_style(self):
        self.frozen_style = True
        for module in (self.style_encoder, self.pre_quant_projection, self.style_quantizer):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_style:
            for module in (self.style_encoder, self.pre_quant_projection, self.style_quantizer):
                module.eval()
        return self

    def forward(self, llada_input_ids, bert_input_ids=None, style_label=None,
                bert_attention_mask=None, attention_mask=None):
        outputs = self.llada_model(
            input_ids=llada_input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False,
        )
        states = outputs.hidden_states
        hidden = states if isinstance(states, torch.Tensor) else states[-1]
        del outputs, states
        if hidden.ndim != 3:
            raise ValueError("Expected base hidden states with shape [batch, sequence, hidden]")
        # Auxiliary modules are float32; cast explicitly outside CUDA autocast too.
        hidden = hidden.to(self.style_projection.weight.dtype)
        if bert_input_ids is None:
            if style_label is None:
                raise ValueError("Provide BERT inputs or explicit style labels")
            labels = torch.as_tensor(style_label, device=hidden.device, dtype=torch.long).reshape(-1)
            quantized = self.style_quantizer.embedding(labels)
            vq_loss = hidden.new_zeros(())
        else:
            encoded = self.style_encoder(input_ids=bert_input_ids, attention_mask=bert_attention_mask)
            style = self.pre_quant_projection(masked_mean(encoded.last_hidden_state, bert_attention_mask))
            vq_loss, quantized, _ = self.style_quantizer(style, style_label)
        style_code = self.style_projection(quantized)
        logits = self.output_layer(self.fusion(hidden, style_code))
        target_logits = self.target_classifier(masked_mean(hidden, attention_mask))
        return SimpleNamespace(logits=logits, vq_loss=vq_loss, target_logits=target_logits)


def resolve_device(device=None):
    try:
        device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Invalid device: {device!r}; expected cpu or cuda[:index]") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; use --device cpu --quantization none for small models")
        index = device.index if device.index is not None else 0
        count = torch.cuda.device_count()
        if index >= count:
            raise ValueError(f"CUDA device index {index} is unavailable; detected {count} device(s)")
        # Precision/capability queries and allocations must use the selected GPU.
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: cpu and cuda[:index]")
    return device


def load_base_model(model_name, device, quantization="none", revision=None):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
    device = resolve_device(device)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision=revision)
    # The upstream LLaDA registers its masked model under AutoModel.
    loader = AutoModel if "AutoModel" in getattr(config, "auto_map", {}) else AutoModelForCausalLM
    kwargs = {"trust_remote_code": True, "revision": revision}
    if quantization == "4bit":
        if device.type != "cuda":
            raise ValueError("4-bit loading requires CUDA; select --quantization none for CPU")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs.update(device_map={"": str(device)}, torch_dtype=compute_dtype,
                      quantization_config=BitsAndBytesConfig(
                          load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True))
    elif quantization == "none":
        kwargs["torch_dtype"] = torch.float32
    else:
        raise ValueError("quantization must be 'none' or '4bit'")
    model = loader.from_pretrained(model_name, **kwargs)
    if quantization == "none":
        model = model.to(device)
    return model


def _attach_modules(base, style, metadata, device):
    model = PulseModel(base, style, metadata["codebook_dim"], len(metadata["target_labels"]))
    # Do not call .to() on the full 4-bit composite.
    for name, module in model.named_children():
        if name != "llada_model":
            module.to(device)
    model.metadata = metadata
    return model


def create_model_and_tokenizers(model_name=DEFAULT_MODEL, style_model_name=DEFAULT_STYLE_MODEL,
                                device="cpu", quantization="none", codebook_dim=768,
                                target_labels=None, lora_target_modules=None,
                                lora_rank=16, lora_alpha=32, revision=None):
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    device = resolve_device(device)
    base = load_base_model(model_name, device, quantization, revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=revision)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Base tokenizer requires a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    mask_id = resolve_mask_id(tokenizer, base)
    if tokenizer.eos_token_id is None:
        raise ValueError("Base tokenizer requires an EOS token")
    if quantization == "4bit":
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    if lora_target_modules is None:
        available = {name.rsplit(".", 1)[-1] for name, _ in base.named_modules()}
        if {"q_proj", "k_proj", "v_proj"}.issubset(available):
            lora_target_modules = ["q_proj", "k_proj", "v_proj"]
            lora_target_modules += [name for name in ("attn_out", "o_proj") if name in available]
        elif {"att_proj", "attn_out"}.issubset(available):
            lora_target_modules = ["att_proj", "attn_out"]
        else:
            raise ValueError("Could not detect LLaDA attention modules; supply --lora-target-modules for other architectures")
    # Generic PEFT forward avoids assuming causal decoding in a masked language model.
    base = get_peft_model(base, LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, target_modules=lora_target_modules,
        lora_dropout=0.05, bias="none"))
    style = AutoModel.from_pretrained(style_model_name).to(device)
    style_tokenizer = AutoTokenizer.from_pretrained(style_model_name)
    metadata = dict(format_version=CHECKPOINT_VERSION, model_name=str(model_name),
                    style_model_name=str(style_model_name), revision=revision,
                    codebook_dim=codebook_dim, target_labels=list(target_labels or ["unspecified"]),
                    style_to_id=STYLE_TO_ID, mask_id=mask_id,
                    resolved_model_revision=getattr(base.config, "_commit_hash", None))
    return _attach_modules(base, style, metadata, device), tokenizer, style_tokenizer


_COMPONENTS = ("style_encoder", "pre_quant_projection", "style_quantizer", "style_projection",
               "fusion", "output_layer", "target_classifier")


def save_checkpoint(model, tokenizer, style_tokenizer, checkpoint_path, metadata):
    """Write a self-contained PULSE checkpoint (base weights remain external)."""
    path = Path(checkpoint_path)
    path.mkdir(parents=True, exist_ok=False)
    config = {**model.metadata, **metadata, "format_version": CHECKPOINT_VERSION}
    model.llada_model.save_pretrained(path / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(path / "tokenizer")
    style_tokenizer.save_pretrained(path / "style_tokenizer")
    # Persist BERT's architecture as well as weights, avoiding a second remote download on reload.
    model.style_encoder.config.save_pretrained(path / "style_encoder_config")
    torch.save({name: {key: value.detach().cpu() for key, value in getattr(model, name).state_dict().items()}
                for name in _COMPONENTS}, path / "components.pt")
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return path


def read_checkpoint_metadata(checkpoint_path):
    path = Path(checkpoint_path)
    if path.is_file():
        raise ValueError("Legacy .pth checkpoints omitted learned components and cannot be restored faithfully. Retrain using the versioned checkpoint directories.")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing checkpoint config: {config_path}")
    metadata = json.loads(config_path.read_text())
    if metadata.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported PULSE checkpoint format version")
    if metadata.get("style_to_id") != STYLE_TO_ID:
        raise ValueError("Checkpoint style mapping differs from Gandhi=0, Mandela=1")
    return metadata


def checkpoint_base_revision(metadata, model_name=None):
    """Pin the resolved commit; an explicit relocated base uses its own weights."""
    reference = str(model_name or metadata["model_name"])
    if Path(reference).is_dir() or (model_name and reference != metadata["model_name"]):
        return None
    return metadata.get("resolved_model_revision") or metadata.get("revision")


def load_checkpoint_model(checkpoint_path, device="cpu", quantization="none",
                          model_name=None, style_model_name=None, is_trainable=False):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from peft import PeftModel, prepare_model_for_kbit_training
    device = resolve_device(device)
    path = Path(checkpoint_path)
    metadata = read_checkpoint_metadata(path)
    metadata = dict(metadata)
    effective_revision = checkpoint_base_revision(metadata, model_name)
    source_name = metadata["model_name"]
    metadata.setdefault("base_model_provenance", {
        "model_name": source_name, "requested_revision": metadata.get("revision"),
        "resolved_revision": metadata.get("resolved_model_revision"),
    })
    if model_name:
        metadata["model_name"] = str(model_name)
    if style_model_name:
        metadata["style_model_name"] = style_model_name
    base = load_base_model(metadata["model_name"], device, quantization, effective_revision)
    if model_name and str(model_name) != source_name:
        # Preserve original provenance separately from the active relocated base.
        metadata["revision"] = None
        metadata["resolved_model_revision"] = getattr(base.config, "_commit_hash", None)
    if quantization == "4bit" and is_trainable:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    base = PeftModel.from_pretrained(base, path / "adapter", is_trainable=is_trainable)
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", trust_remote_code=True)
    style_tokenizer = AutoTokenizer.from_pretrained(path / "style_tokenizer")
    if style_model_name:
        style = AutoModel.from_pretrained(style_model_name).to(device)
    else:
        style_config = AutoConfig.from_pretrained(path / "style_encoder_config")
        style = AutoModel.from_config(style_config).to(device)
    model = _attach_modules(base, style, metadata, device)
    state = torch.load(path / "components.pt", map_location="cpu", weights_only=True)
    if set(state) != set(_COMPONENTS):
        raise ValueError("Checkpoint is missing learned PULSE components")
    for name in _COMPONENTS:
        getattr(model, name).load_state_dict(state[name], strict=True)
    if resolve_mask_id(tokenizer, base) != metadata["mask_id"]:
        raise ValueError("Checkpoint/tokenizer mask token mismatch")
    if metadata.get("stage") == "sccg":
        model.freeze_style()
    model.train(is_trainable)
    return model, tokenizer, style_tokenizer, metadata
