# Contributing

Use Python 3.10 or 3.11 in a virtual environment and install `python -m pip install -e '.[dev]'` from the root. Full evaluation dependencies are optional; the basic metrics, unit tests, and CPU smoke run do not download model weights.

Before submitting a change:

```bash
python -m ruff check .
python -m unittest discover -s tests -v
python scripts/smoke_pipeline.py --output-dir artifacts/contribution-check
```

Choose a fresh output directory for each smoke run. CI runs these checks on CPU. Changes to GPU quantization or full evaluator behavior need an additional resource-specific run and a clear statement of what was exercised.

Keep model architecture, prompt formatting, style mappings, and checkpoint handling in `pulse_core.py` so training and inference remain consistent. Checkpoint format changes must version the format and either provide a faithful migration or reject incompatible files explicitly. Add tests around substantive changes to masking, checkpoint transfer, sampling, and data alignment.

Preserve empty generations during evaluation. Never replace missing resources or evaluator errors with valid-looking scores. Changes to metric definitions must update the protocol version and documentation.

Do not commit real datasets, credentials, downloaded weights, or run artifacts. Describe the behavior change, test evidence, and any unverified resource-specific behavior in the pull request.
