# Pipeline investigation and validation

Audit date: **24 November 2026**. Scope: the supplied source directory, its training → inference → evaluation integration, and preparation for a public GitHub repository.

## Conclusion

The original code was not runnable end to end as supplied. After repairs, both training stages, full checkpoint reload, diffusion generation, and basic evaluation pass on CPU with locally initialized tiny models. This was verified with both a generic transformer fixture and the **actual official LLaDA architecture** reduced to a tiny configuration.

Full-size 8B training, CUDA/4-bit execution, model quality, and actual model-based evaluation remain unverified. The provided directory had no real dataset, trained checkpoints, external style classifier, or verified quote corpus. The machine's NVIDIA driver was unavailable. Nothing in this report constitutes a reproduced research result.

## Failures found and repaired

| Area | Original failure | Current behavior |
| --- | --- | --- |
| Configuration | Hardcoded `/home/models/...` paths and a placeholder CSV path | Model/data/checkpoint locations supplied by CLI |
| SSLCL saving | Missing directory creation; omitted BERT, output head, and pre-quantization projection | Complete versioned checkpoint directories saved every epoch |
| SCCG transfer | Reloaded only some SSLCL state; randomly reset learned components | Strict restoration of adapter and every PULSE component |
| SCCG saving | Saved only after epoch index 5 | Every completed epoch saved, including a short smoke run |
| Optimizer | Pre-quantization projection and target classifier omitted | All enabled trainable components included; tested parameter updates |
| Targets | Fixed nine classes and 4096-dimensional assumptions | Dimensions inferred; label mapping saved and enforced |
| Tokenization | Added/replaced reserved tokens; training/inference mask disagreement | Native vocabulary retained; saved mask ID validated |
| Prompting | Different SCCG training and inference prompts | One shared prompt encoder and matching BERT prompt text |
| Masking | Padding affected corruption/pooling; loss ignored noise probabilities | Response-only corruption, attention-aware pooling, inverse-probability loss |
| Style freezing | Frozen modules could return to train mode | BERT, pre-projection, and codebook stay frozen/eval in SCCG |
| Device handling | Unconditional CUDA autocast and quantization despite CPU fallback | Explicit CPU/quantization rules and device-aware precision |
| Sampling | Fractional powers of negative Gumbel values produced NaNs; fragile schedules and CFG batching | Stable log-space sampling; validated schedules; conditioning duplicated consistently |
| Evaluation alignment | Row-order truncation could pair unrelated outputs/references | Strict ID or input/style matching; intentional row alignment is explicit |
| Evaluation data | Latin-1 decoding and removal of empty generations | UTF-8 and all generated rows retained |
| Evaluation validity | Custom penalties mixed into standard metrics; fake NSP=0; failures hidden as scores | Documented standard metrics; no NSP; failed evaluators stop the run |
| Dependencies | Unbounded install instructions and incompatible host Hub version | Coherent pinned core versions, separate GPU/evaluation extras |
| Documentation | Broken Markdown fences and stale claim that `generate_vq.py` was missing | Complete README, data/evaluation guides, GitHub instructions, CI, MIT license |

The PerStyM fusion and separate trainable output head remain part of the supplied design. The output head is initialized anew for SSLCL and carried forward thereafter. This audit repairs the implementation; it does not establish the scientific effectiveness of that architecture or reinterpret stage names as an independently verified method.

## Executed checks

All **34 regression tests passed**, with no failures or skips. The suite covers masking and loss normalization, actual optimizer updates in both stages, exact checkpoint state/logit restoration, style freezing, malformed inputs, batched sampling, CFG conditioning, numerical sampling stability, alignment, quote formats, revision pinning, CUDA device selection, and evaluation provenance. Full-evaluator orchestration is tested using real Torch with **mocked evaluator models**; these tests do not measure real BERTScore, toxicity, or style accuracy. Ruff and Python compilation checks also passed.

```bash
python -m ruff check .
python -m compileall -q .
python -m unittest discover -s tests -v
python scripts/smoke_pipeline.py --output-dir artifacts/smoke
```

The default smoke run uses a random one-layer GPT-2 (hidden size 16) and a random one-layer BERT (hidden size 16), with codebook dimension 8 so the pre-quantization projection is exercised. It performs real backward/optimizer steps through both CLIs, reloads a saved SCCG checkpoint in a new process, generates for two input IDs, and evaluates the resulting CSV with basic metrics. Every command returned zero.

An additional run used official LLaDA model/configuration code from [`GSAI-ML/LLaDA-8B-Instruct`, commit `08b83a6feb34df1a6011b80c3c00c7563e963b07`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/tree/08b83a6feb34df1a6011b80c3c00c7563e963b07). No upstream weights were downloaded. The random tiny model had `d_model=16`, one layer, two heads, vocabulary 93, context 128, and 5,584 base parameters. Two updates per training stage, checkpoint reload, two-row generation (`gen_length=8`, `steps=4`, `block_length=4`), and ID-aligned basic evaluation all passed. A second generation run with temperature 0.6, CFG scale 1.5, and random remasking also passed. Output quality is intentionally meaningless for this random fixture.

To reproduce the architecture check, download **only** the pinned model code/config and review it, then run the offline smoke script:

```bash
hf download GSAI-ML/LLaDA-8B-Instruct \
  config.json configuration_llada.py modeling_llada.py \
  --revision 08b83a6feb34df1a6011b80c3c00c7563e963b07 \
  --local-dir models/llada-source
python scripts/smoke_pipeline.py \
  --llada-source-dir models/llada-source \
  --output-dir artifacts/llada-smoke
```

The source is external custom Python and is executed through Transformers' `trust_remote_code=True`. The script shrinks its configuration and initializes fresh tiny weights; it does not load an 8B checkpoint.

Other checks included CLI help, training data validation, a local editable package build/install, dependency import checks, and a repository scan for hardcoded user paths or credential patterns. The scan found no credentials in the prepared source. Core package versions and upstream code hashes are recorded in [validation evidence](validation/evidence.json).

## Limits and compatibility

- **Production GPU path:** no functioning NVIDIA driver, so bitsandbytes loading, GPU memory use, and full GPU optimization have not been executed. Only the base model is quantized; the additional float32 layers are large.
- **Full evaluation:** no project-trained persona classifier, quote corpus, or cached evaluation models were supplied. Actual full scores were not computed. Those external resources must be supplied separately.
- **Real data and quality:** no real train/validation/test corpus was supplied. The trainer has no automatic validation loop, early stopping, or best-checkpoint selection.
- **Legacy weights:** original `.pth` checkpoints omitted learned state and cannot be faithfully migrated from those files alone. Retraining is required.
- **Resume:** checkpoints support stage transfer and inference, but do not save optimizer state for exact interrupted-training continuation.
- **Protocol change:** corrected scores use metric definition version 2 and are not directly comparable to results from the original penalized evaluator.
- **Dependency scope:** the core stack was checked on Python 3.10.6. CI is configured for Python 3.10 and 3.11, but hosted CI and a fully fresh dependency installation have not been executed here. Validation used an isolated environment reusing installed Torch dependencies, with the incompatible Hub version overridden locally.

## Primary implementation references

- [LLaDA model code](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/blob/08b83a6feb34df1a6011b80c3c00c7563e963b07/modeling_llada.py): AutoModel registration, module names, masks, hidden-state API.
- [LLaDA fine-tuning guidelines](https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md): masked diffusion objective and inverse-probability weighting.
- [LLaDA sampler](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py): block transfer schedule and sampling contract.
- [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization): preparation for adapter training on a quantized base.
- [BERTScore](https://github.com/Tiiiger/bert_score), [Detoxify](https://github.com/unitaryai/detoxify), [NLTK METEOR](https://www.nltk.org/api/nltk.translate.meteor_score.html), and [SentenceTransformer](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html): evaluator APIs.
