# Evaluation protocol

`metric.py` has two modes. `--metrics basic` runs with the Python standard library alone. `--metrics full` also computes model-based metrics and requires external packages, a trained style classifier, NLTK WordNet, and a quote corpus. No missing evaluator is replaced with a zero score.

## Reference matching

Default `--alignment auto` uses shared `id` values if both files contain that column; otherwise it uses the pair `Hatespeech` and `Style`. Keys must be nonempty and unique on both sides. Reference rows are reordered to match the generated rows. The input text and style are checked for consistency after ID matching.

Every generated row needs a reference, and every reference needs a generation. An intentional subset may use `--allow_subset`; unused references are then counted in metadata. `--alignment row` explicitly selects row-order matching and requires equal counts. It cannot be combined with `--allow_subset`.

The evaluator accepts legacy lowercase `hatespeech`/`style` and uppercase `ID` aliases, but rejects ambiguous duplicate headers. It never truncates to the shorter file or drops an empty generation. CSVs are read as UTF-8; whitespace is normalized for matching and scoring.

## Metrics

| Name | Definition | Interpretation |
| --- | --- | --- |
| Exact Match | Whitespace-normalized, case-folded text equality | Higher is more reference overlap |
| Token F1 | Precision/recall F1 over case-folded Unicode word-token multisets | Higher is more reference overlap |
| Generated Tokens | Count of the same word tokens | Word count, not LM subword count |
| Empty Generation | 1 for an empty generated string, otherwise 0 | Dataset mean is empty-output rate |
| R-1 / R-2 / R-L | Standard `rouge-score` F1, with stemming | Higher is more reference overlap |
| M | NLTK METEOR, Treebank tokens and WordNet | Higher is more reference overlap |
| BS | BERTScore F1, no baseline rescaling; RoBERTa-large layer 17 by default | Higher is more embedding similarity |
| CoSim | Cosine between normalized SentenceTransformer embeddings | Higher is more semantic similarity; negatives retained |
| T | Detoxify `unbiased` model's `toxicity` probability | Lower is less model-estimated toxicity |
| Style Conformity | RoBERTa classifier probability of the requested style after removing quoted spans | Probability, not classification accuracy |
| Quote Conformity | Mean best RapidFuzz full-string ratio against requested persona's quote corpus, divided by 100 | Similarity, not verified attribution |
| Quote Match Rate | Fraction of extracted quotes whose best similarity meets `quote_threshold` | Threshold is 0–100; result is 0–1 |
| Quote Presence | 1 if a quoted span is present, otherwise 0 | Dataset mean is quote-presence rate |
| Reference Quote Match | Mean best normalized similarity to quoted spans in the reference | Zero if no generated/reference quote matches |

Straight and curly double quotes are recognized. Quotation-based metrics are zero when there are no generated quoted spans, so dataset averages include non-quoting outputs. All summary metrics are arithmetic means over the evaluated rows. These automatic scores do not replace human evaluation of safety, usefulness, or rhetorical style.

## Full evaluation prerequisites

Install `requirements-eval.txt`. Provide a `RobertaForSequenceClassification` checkpoint with two labels, **Gandhi=0** and **Mandela=1**, compatible with `--roberta_model_path`. Both a raw state dictionary and a dictionary containing `state_dict` are accepted. Loading is strict. A PULSE checkpoint's BERT style encoder is a different model and cannot serve as this classifier. Training the external classifier is outside this repository.

`--roberta_model_path` is used for both BERTScore and the style-classifier architecture. If changing it, set an appropriate `--bertscore_num_layers`. The default is RoBERTa-large with 17 layers for BERTScore. `--sentence_model_path` defaults to `sentence-transformers/all-mpnet-base-v2`. Detoxify downloads its `unbiased` model unless a local `--toxicity_checkpoint` is supplied. These evaluator models can require substantial memory and may download on the first run.

Install WordNet explicitly:

```bash
python -m nltk.downloader -d nltk_data wordnet omw-1.4
```

For an offline full run, cache all Hugging Face evaluator models and the Detoxify checkpoint in advance, then supply local paths:

```bash
python metric.py \
  --input_file outputs/pulse.csv --ground_truth_file data/test.csv \
  --metrics full --offline --device cpu \
  --sentence_model_path models/all-mpnet-base-v2 \
  --roberta_model_path models/roberta-large \
  --style_checkpoint models/style_classifier.pt \
  --toxicity_checkpoint models/unbiased_toxicity.ckpt \
  --quotes_file data/quotes.json --nltk_data_path nltk_data \
  --output_dir results/full
```

`--hf_cache` selects an optional Hugging Face cache directory. The default behavior respects the existing environment; it no longer forces offline mode. `--fine_tune 1 --finetune_reference_file PATH` explicitly chooses an alternative reference file and applies the same alignment checks.

## Outputs and protocol changes

For input `pulse.csv`, the output directory contains `pulse_metrics.csv`, `pulse_summary.csv`, and `pulse_metadata.json`. Metadata records arguments, alignment, sample count, input/reference hashes, relevant installed package versions, and metric definition version 2. Quote-corpus and style-checkpoint hashes are included when used. The optional evaluator dependency ranges are not a fully locked research environment; archive the resolved environment and model revisions with a formal experiment.

The original script applied custom length/diversity penalties, clipped negative cosine similarity, used partial quote matching on a 0–100 scale, dropped empty generations, and reported unimplemented NSP as zero. Those behaviors were removed. Scores generated with the corrected protocol must not be compared directly to old reported scores. Full model-based evaluation has not been run with the project's real resources in the repository preparation audit; see `VALIDATION_REPORT.md`.
