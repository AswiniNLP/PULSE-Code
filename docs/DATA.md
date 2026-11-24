# Data contracts

All CSVs use UTF-8, a header row, standard CSV quoting, and one sample per row. See `examples/train.csv` and `examples/test.csv` for synthetic examples.

| Column | Training | Inference input | Evaluation reference | Meaning |
| --- | --- | --- | --- | --- |
| `id` | Recommended | Recommended | Recommended | Stable, nonempty, unique sample ID |
| `Hatespeech` | Required | Required | Required | Input text |
| `Style` | Required | Required | Required | Exactly `Gandhi` or `Mandela` |
| `Target` | Required | Optional | Optional | Target category for auxiliary classification |
| `Counterspeech` | Required | Optional | Required | Reference response |
| `CS` | Not used | Added to output | Not used | Generated response |

The trainer requires nonempty input, style, target, and reference values. It derives a sorted target-label mapping from SSLCL training data and saves that mapping. SCCG can use a subset of these categories but cannot introduce a new category. There is no fixed assumption of nine target classes.

Use the same sample IDs in the test input and reference file. Give repeated input/style pairs different IDs when they represent different samples. Preserve IDs as strings, including leading zeros. All original inference columns are retained alongside `CS`.

Training truncates responses to fit `max_length`, reserving one EOS token. Prompts are kept intact; a prompt that leaves no response space is an error. Padding and prompt tokens are excluded from the denoising objective. BERT inputs are separately truncated to `style_max_length`, which must fit its positional limit.

Split training, validation, and test data before running either stage. Keep related or duplicate samples in the same split when appropriate to the experiment. The current trainer runs only on the supplied training CSV; checkpoint selection and held-out validation are separate tasks.

The bundled examples are synthetic plumbing fixtures, not the project's original experimental data. Real datasets stay under ignored `data/` unless their redistribution has been explicitly arranged. No data preparation, annotation, data download, or classifier training is silently performed by the scripts.

## Quotes

For full quote evaluation, provide both personas with nonempty lists of verified quotes:

```json
{
  "Gandhi": ["<replace with a verified Gandhi quote>"],
  "Mandela": ["<replace with a verified Mandela quote>"]
}
```

These placeholders are documentation, not a quote corpus. The format is specified by `examples/quotes.schema.json`. The original nested format is also supported:

```json
{
  "authors": {
    "Mahatma Gandhi": {"quotes": ["<verified quote>"]},
    "Nelson Mandela": {"quotes": ["<verified quote>"]}
  }
}
```

Keep provenance for the verified source texts with the dataset. Fuzzy text similarity measures overlap with this corpus; it does not authenticate a quote.
