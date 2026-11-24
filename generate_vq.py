"""Blockwise masked-token generation for the style-conditioned PULSE model.

The transfer schedule follows the official LLaDA sampler:
https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
Gumbel sampling is performed in log space to avoid overflow and invalid powers.
"""

import math

import torch
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    """Return log-space Gumbel sampling scores (argmax gives sampled tokens)."""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite nonnegative number")
    if temperature == 0:
        return logits
    scores = logits.to(torch.float64)
    # Clamping also handles the exact zero that torch.rand_like may return.
    uniform = torch.rand_like(scores).clamp_(min=torch.finfo(scores.dtype).tiny)
    return scores - temperature * torch.log(-torch.log(uniform))


def get_num_transfer_tokens(mask_index, steps):
    """Distribute each row's masked tokens across an exact number of steps."""
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if mask_index.ndim != 2:
        raise ValueError("mask_index must have shape [batch, length]")
    counts = mask_index.sum(dim=1, keepdim=True)
    positions = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    return counts // steps + (positions < counts % steps).long()


def validate_generation_options(steps, gen_length, block_length, temperature=0.0,
                                cfg_scale=0.0, remasking="low_confidence"):
    """Validate the block schedule before allocating or loading a large model."""
    for name, value in (("steps", steps), ("gen_length", gen_length),
                        ("block_length", block_length)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks or steps < num_blocks or steps > gen_length:
        raise ValueError("steps must be divisible by the number of blocks, at least "
                         "that number, and no greater than gen_length")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite nonnegative number")
    if not math.isfinite(cfg_scale) or cfg_scale < 0:
        raise ValueError("cfg_scale must be a finite nonnegative number")
    if remasking not in {"low_confidence", "random"}:
        raise ValueError("remasking must be low_confidence or random")


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128,
             temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
             mask_id=126336, print_logits=False, tokenizer=None,
             bert_input_ids=None, style_label=None, bert_attention_mask=None,
             attention_mask=None):
    """Fill generated masks while keeping prompts and completed blocks fixed.

    ``prompt`` is a [batch, prompt_length] tensor. Padded batches must use left
    padding and provide ``attention_mask``. ``style_label`` has one ID per row.
    The returned tensor contains both the original prompt and generated tokens.
    """
    validate_generation_options(steps, gen_length, block_length, temperature,
                                cfg_scale, remasking)
    if prompt.ndim != 2 or prompt.shape[0] == 0 or prompt.shape[1] == 0:
        raise ValueError("prompt must have nonempty shape [batch, prompt_length]")
    if prompt.dtype != torch.long:
        raise ValueError("prompt must have dtype torch.long")
    if (prompt == mask_id).any():
        raise ValueError("prompt must not contain the generation mask token")
    if attention_mask is not None and attention_mask.shape != prompt.shape:
        raise ValueError("attention_mask must have the same shape as prompt")
    for name, tensor in (("bert_input_ids", bert_input_ids),
                         ("bert_attention_mask", bert_attention_mask),
                         ("style_label", style_label)):
        if tensor is not None and (tensor.ndim == 0 or tensor.shape[0] != prompt.shape[0]):
            raise ValueError(f"{name} must have one entry per prompt row")

    batch_size, prompt_length = prompt.shape
    x = torch.full((batch_size, prompt_length + gen_length), mask_id,
                   dtype=torch.long, device=prompt.device)
    x[:, :prompt_length] = prompt
    full_attention_mask = None
    if attention_mask is not None:
        full_attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, gen_length), dtype=attention_mask.dtype,
                       device=prompt.device),
        ], dim=1)
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    def duplicate(tensor):
        return None if tensor is None else torch.cat([tensor, tensor], dim=0)

    for block in range(num_blocks):
        start = prompt_length + block * block_length
        end = start + block_length
        schedule = get_num_transfer_tokens(x[:, start:end] == mask_id,
                                           steps_per_block)
        for step in range(steps_per_block):
            candidates = x == mask_id
            candidates[:, :start] = False
            candidates[:, end:] = False
            if not candidates.any():
                break
            kwargs = {"bert_input_ids": bert_input_ids,
                      "style_label": style_label,
                      "bert_attention_mask": bert_attention_mask,
                      "attention_mask": full_attention_mask}
            if cfg_scale:
                unconditional = x.clone()
                unconditional[:, :prompt_length] = mask_id
                kwargs = {name: duplicate(tensor) for name, tensor in kwargs.items()}
                outputs = model(torch.cat([x, unconditional], dim=0), **kwargs)
            else:
                outputs = model(x, **kwargs)
            expected_batch = batch_size * (2 if cfg_scale else 1)
            if (outputs.logits.ndim != 3
                    or outputs.logits.shape[:2] != (expected_batch, x.shape[1])
                    or not 0 <= mask_id < outputs.logits.shape[-1]):
                raise ValueError("model logits or mask_id do not match the token sequence")
            # A selected mask would consume its scheduled transfer without resolving
            # the position, leaving masks in the final response.
            # Sampling only the active block avoids allocating full-sequence
            # float64 noise and softmax tensors for the 126k-token vocabulary.
            logits = outputs.logits[:, start:end].clone()
            del outputs
            if cfg_scale:
                conditional_logits, unconditional_logits = logits.chunk(2, dim=0)
                logits = unconditional_logits + (cfg_scale + 1) * (
                    conditional_logits - unconditional_logits)
                del conditional_logits, unconditional_logits
            logits[..., mask_id] = -torch.inf
            block_candidates = candidates[:, start:end]
            if not torch.isfinite(logits[block_candidates]).any(dim=-1).all():
                raise ValueError("model has no finite non-mask candidate for a generated position")
            if torch.isnan(logits[block_candidates]).any() or torch.isposinf(logits[block_candidates]).any():
                raise ValueError("model returned invalid logits for generated positions")
            if print_logits:
                position = block_candidates[0].nonzero().flatten()
                if position.numel():
                    values, indices = logits[0, position[0]].topk(min(5, logits.shape[-1]))
                    tokens = [tokenizer.decode([index.item()]) if tokenizer else index.item()
                              for index in indices]
                    print(f"Top candidate logits: {list(zip(tokens, values.tolist()))}")
            predicted = add_gumbel_noise(logits, temperature).argmax(dim=-1)
            if remasking == "low_confidence":
                probabilities = F.softmax(logits.float(), dim=-1)
                confidence = probabilities.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
            else:
                confidence = torch.rand((batch_size, block_length), device=x.device)
            confidence = confidence.masked_fill(~block_candidates, -torch.inf)
            for row in range(batch_size):
                count = int(schedule[row, step].item())
                if count:
                    selected = confidence[row].topk(count).indices
                    x[row, start + selected] = predicted[row, selected]
    if (x[:, prompt_length:] == mask_id).any():
        raise RuntimeError("generation finished with unresolved masks")
    return x
