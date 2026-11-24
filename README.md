# PULSE: style-conditioned counterspeech

PULSE learns to generate counterspeech in response to hateful text, conditioned on the rhetorical styles **Gandhi** and **Mandela**. It combines a LLaDA diffusion language model, LoRA adapters, a BERT style encoder, a supervised vector quantizer, and the PerStyM fusion module.

The pipeline has two training stages:

1. **SSLCL:** learn style representations and denoise counterspeech tokens.
2. **SCCG:** initialize from SSLCL and train response generation, with an auxiliary classifier for the target category. The learned BERT encoder, pre-quantization projection, and style codebook stay frozen.

Inference loads a complete SCCG checkpoint and fills masked response tokens block by block. Evaluation matches predictions to reference rows before computing scores.

**Validation status:** the CPU pipeline has been exercised through both training stages, checkpoint reload, generation, and basic evaluation using tiny local models. Full-size LLaDA training, GPU quantization, and model-based evaluation require resources that are not included. Read [the validation report](docs/VALIDATION_REPORT.md) for evidence and limits. These checks do not establish counterspeech quality or reproduce research results.

## Repository contents

```text
.
├── pulse_core.py              # Shared architecture, prompts, checkpoint save/load
├── pulse_train.py             # SSLCL and SCCG command-line training
├── pulse_inference.py         # CSV → generated counterspeech CSV
├── generate_vq.py             # Blockwise diffusion sampling
├── metric.py                  # Basic and full evaluation
├── pyproject.toml             # Core dependencies and optional extras
├── examples/                  # Small synthetic CSVs and quote corpus schema
├── scripts/
│   ├── check_environment.py   # Local dependency, GPU, and CSV checks
│   └── smoke_pipeline.py      # Offline CPU integration run with tiny models
├── tests/                    # Regression and checkpoint integration tests
├── docs/                     # Data, evaluation, validation, and GitHub instructions
└── .github/workflows/ci.yml   # CPU checks on Python 3.10 and 3.11
```

The real datasets, trained checkpoints, style classifier, and verified quote corpus are **not supplied**. The example CSVs contain synthetic demonstration text. They are sufficient for exercising the commands, not for training a useful model.

## Install

Use Python 3.10 or 3.11 and a fresh virtual environment. Run every command below from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Core versions are pinned together in `pyproject.toml`. The original machine had an incompatible Transformers/Hugging Face Hub combination; use this environment instead of relying on unrelated installed packages.

For full-size training and inference with 4-bit loading:

```bash
python -m pip install -e '.[gpu]'
python scripts/check_environment.py --check-ml --require-cuda
```

Install the PyTorch build appropriate for your CUDA environment if the default wheel is unsuitable; the package targets PyTorch 2.8.0. A working NVIDIA driver is required for the GPU path. `--quantization 4bit` quantizes the base language model; BERT, the fusion layers, and the large output projection still consume substantial memory. GPU memory requirements have not been benchmarked. Start with batch size 1 and measure memory before a long run.

The default pretrained models are [`GSAI-ML/LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) and `bert-base-uncased`. Model IDs download through Hugging Face; local directories are also accepted. LLaDA uses custom model code loaded with `trust_remote_code=True`. Review that source and pass `--revision <commit>` when you need a pinned base-model/code revision.

## Verify the installation

```bash
python scripts/check_environment.py --check-ml --train-csv examples/train.csv
python -m unittest discover -s tests -v
python scripts/smoke_pipeline.py --output-dir artifacts/smoke
```

The smoke script creates random tiny GPT-2 and BERT models locally, runs the actual training/inference/evaluation CLIs on CPU, and writes `artifacts/smoke/smoke_report.json`. It makes no model downloads. Use a fresh output directory for each run. Its generated text and scores are diagnostic fixtures. For the actual LLaDA architecture, the script also accepts `--llada-source-dir` with reviewed local model code; see [the validation report](docs/VALIDATION_REPORT.md).

## Prepare data

Training CSVs must use UTF-8 and contain `Hatespeech`, `Style`, `Target`, and `Counterspeech`. Style values are exactly `Gandhi` or `Mandela`; their fixed IDs are 0 and 1. Add a unique `id` column to every split and keep those IDs in evaluation outputs. See [the data guide](docs/DATA.md).

Keep the real files under ignored `data/`, for example `data/train.csv` and `data/test.csv`. Split data before training, and select checkpoints on a separate validation split. There is no automatic split, validation scoring, or best-checkpoint selection in this trainer.

Validate the training schema before loading models:

```bash
python pulse_train.py --stage sslcl --train-csv data/train.csv --validate-only
```

## Train both stages

First train SSLCL:

```bash
python pulse_train.py \
  --stage sslcl \
  --train-csv data/train.csv \
  --model-name GSAI-ML/LLaDA-8B-Instruct \
  --style-model-name bert-base-uncased \
  --device cuda:0 --quantization 4bit \
  --codebook-dim 768 --epochs 3 --batch-size 1 \
  --max-length 512 --style-max-length 512 \
  --output-dir checkpoints/sslcl/768
```

Then train SCCG from a selected SSLCL checkpoint:

```bash
python pulse_train.py \
  --stage sccg \
  --train-csv data/train.csv \
  --sslcl-checkpoint checkpoints/sslcl/768/sslcl_epoch_3 \
  --device cuda:0 --quantization 4bit \
  --epochs 3 --batch-size 1 \
  --max-length 512 --style-max-length 512 \
  --output-dir checkpoints/sccg/768
```

Each epoch creates a new checkpoint directory, such as `sccg_epoch_3`. Existing epoch directories are not overwritten. SCCG inherits the base-model revision, style mapping, codebook dimension, and target-class mapping from SSLCL; its data must not introduce unseen target classes.

Checkpoints contain the LoRA adapter, all PULSE components including the learned BERT weights, both tokenizers, the BERT architecture, and configuration metadata. Base LLaDA weights stay external. If a local base-model path moves, supply `--model_name /new/model/path` at inference. Checkpoints support stage transfer and inference; optimizer state and exact interrupted-run resumption are not implemented.

The original `.pth` files omitted learned state. This version rejects them rather than silently substituting random components. Retrain using the directory format.

Useful training options include `--learning-rate`, `--seed`, `--max-steps`, `--target-loss-weight`, and `--vq-loss-weight`. `--max-steps 1` is useful for a resource check. Small CPU models need `--device cpu --quantization none`; the full 8B model is not a practical CPU smoke test.

## Generate counterspeech

The input needs `Hatespeech` and `Style`; other columns, including `id` and `Counterspeech`, are preserved in the output. Generated text is added as `CS`.

```bash
python pulse_inference.py \
  --input_file data/test.csv \
  --output_file outputs/pulse.csv \
  --checkpoint checkpoints/sccg/768/sccg_epoch_3 \
  --device cuda:0 --quantization 4bit \
  --gen_length 128 --steps 64 --block_length 32 \
  --temperature 0 --seed 42
```

`gen_length` must be divisible by `block_length`. The number of blocks is `gen_length / block_length`; `steps` must be divisible by that number, at least that number, and no greater than `gen_length`. Prompt tokens plus generated tokens must fit the checkpoint's `max_length`. Overlong inputs produce an error rather than silently truncating the prompt.

Training and inference use the same prompt tokens, native mask ID, and style mapping. Inference processes rows individually and writes the CSV only after all rows succeed.

## Evaluate

For a quick offline check with no evaluation-model dependencies:

```bash
python metric.py \
  --input_file outputs/pulse.csv \
  --ground_truth_file data/test.csv \
  --metrics basic \
  --output_dir results/basic
```

Basic mode reports normalized exact match, multiset word-overlap F1, generated word count, and the empty-output rate. These measure overlap and output integrity, not toxicity or rhetorical style.

Full evaluation adds ROUGE, METEOR, BERTScore, sentence similarity, Detoxify toxicity, persona probability, and quote matching. It requires separate evaluator resources:

```bash
python -m pip install -r requirements-eval.txt
python -m nltk.downloader wordnet omw-1.4
python metric.py \
  --input_file outputs/pulse.csv \
  --ground_truth_file data/test.csv \
  --metrics full \
  --sentence_model_path sentence-transformers/all-mpnet-base-v2 \
  --roberta_model_path roberta-large \
  --style_checkpoint models/style_classifier.pt \
  --quotes_file data/quotes.json \
  --device cuda:0 --output_dir results/full
```

The style classifier must be independently trained with **Gandhi=0, Mandela=1**; it is not the BERT encoder in a PULSE checkpoint. No classifier-training pipeline is supplied. Quote matching requires a verified corpus; `examples/quotes.schema.json` describes its compact format. Read [the evaluation protocol](docs/EVALUATION.md) for definitions, offline resource setup, and alignment rules.

Evaluation writes per-row scores, a summary CSV, and JSON provenance with input hashes and options. Empty generations count in all averages. Missing resources and evaluator failures stop the run. **Metric definitions were corrected; scores from the original script are not directly comparable.**

## Publish and contribute

See [GitHub preparation](docs/GITHUB.md) for the upload commands and [CONTRIBUTING.md](CONTRIBUTING.md) for checks. Models, private datasets, outputs, caches, and credentials are excluded by `.gitignore`.

Code is released under the [MIT license](LICENSE). Model weights, datasets, and external evaluator resources retain their own terms.
