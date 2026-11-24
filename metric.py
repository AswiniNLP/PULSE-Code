#!/usr/bin/env python3
"""Evaluate PULSE CSV outputs with explicit reference alignment.

``--metrics basic`` uses only the Python standard library and measures exact
match, multiset word overlap F1, word count and empty outputs. ``--metrics full``
also loads the research evaluators. Missing resources or evaluator failures are
errors, never substitute scores. See README.md for metric definitions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
from statistics import mean


STYLE_LABELS = {"Gandhi": 0, "Mandela": 1}
ALIASES = {"hatespeech": "Hatespeech", "style": "Style", "ID": "id"}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True, help="UTF-8 generated CSV with a CS column.")
    parser.add_argument("--ground_truth_file", required=True, help="Reference CSV with Hatespeech, Style, Counterspeech.")
    parser.add_argument("--output_dir", default="results/metrics")
    parser.add_argument("--metrics", choices=("basic", "full"), default="full",
                        help="basic is offline and dependency-free; full loads ML evaluators (default).")
    parser.add_argument("--alignment", choices=("auto", "id", "keys", "row"), default="auto",
                        help="auto uses shared id, otherwise Hatespeech + Style. row explicitly opts into row order.")
    parser.add_argument("--allow_subset", action="store_true",
                        help="Allow unused reference rows during keyed alignment; report their count.")
    parser.add_argument("--finetune_reference_file", help="Alternative reference CSV, used only with --fine_tune 1.")
    parser.add_argument("--fine_tune", type=int, choices=(0, 1), default=0)
    parser.add_argument("--quotes_file", help="Persona quotes JSON. Required for full metrics; optional for basic.")
    parser.add_argument("--quote_threshold", type=float, default=80.0,
                        help="Minimum RapidFuzz ratio (0-100) counted as a quote match.")
    parser.add_argument("--sentence_model_path", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--roberta_model_path", default="roberta-large",
                        help="RoBERTa model/path for BERTScore and style classifier architecture.")
    parser.add_argument("--style_checkpoint", help="Trained two-class RoBERTa state dict (Gandhi=0, Mandela=1).")
    parser.add_argument("--bertscore_num_layers", type=int, default=17,
                        help="BERTScore layer count; 17 matches the standard roberta-large configuration.")
    parser.add_argument("--toxicity_checkpoint", help="Optional locally cached Detoxify unbiased checkpoint.")
    parser.add_argument("--nltk_data_path", help="Optional local NLTK resource directory containing WordNet.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", help="Evaluation device, e.g. cpu, cuda, cuda:1. Default: CUDA if available.")
    parser.add_argument("--hf_cache", help="Optional Hugging Face cache directory.")
    parser.add_argument("--offline", action="store_true", help="Require locally cached evaluation models and resources.")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_length < 1 or args.bertscore_num_layers < 1:
        parser.error("batch_size, max_length and bertscore_num_layers must be positive.")
    if not 0 <= args.quote_threshold <= 100:
        parser.error("quote_threshold must be between 0 and 100.")
    if args.fine_tune and not args.finetune_reference_file:
        parser.error("--finetune_reference_file is required with --fine_tune 1.")
    if args.metrics == "full" and (not args.style_checkpoint or not args.quotes_file):
        parser.error("--metrics full requires --style_checkpoint and --quotes_file.")
    return args


def clean(text):
    """Normalize whitespace without removing quotes or changing model input case."""
    return re.sub(r"\s+", " ", "" if text is None else str(text)).strip()


def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        canonical = [ALIASES.get(header, header) for header in headers]
        if not headers or len(set(canonical)) != len(canonical):
            raise ValueError(f"{path}: missing or duplicate CSV headers (including aliases).")
        rows = []
        for line_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed CSV at line {line_number}.")
            rows.append({ALIASES.get(key, key): value for key, value in row.items()})
    if not rows:
        raise ValueError(f"{path}: CSV has no data rows.")
    return rows


def _require_columns(rows, columns, name):
    missing = set(columns) - rows[0].keys()
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}.")


def _index_rows(rows, columns, name):
    indexed = {}
    for row_number, row in enumerate(rows, 2):
        key = tuple(clean(row[column]) for column in columns)
        if any(not value for value in key):
            raise ValueError(f"{name}: empty alignment key at CSV line {row_number}.")
        if key in indexed:
            raise ValueError(f"{name}: duplicate alignment key at CSV line {row_number}. "
                             "Provide unique id values or explicitly use --alignment row.")
        indexed[key] = row
    return indexed


def load_data(input_file, ground_truth_file, finetune_reference_file=None, fine_tune=False,
              alignment="auto", allow_subset=False):
    """Return aligned rows and provenance; never silently truncate, reorder, or drop outputs."""
    if fine_tune and not finetune_reference_file:
        raise ValueError("finetune_reference_file is required when fine_tune is true.")
    reference_file = finetune_reference_file if fine_tune else ground_truth_file
    generated = _read_csv(input_file)
    references = _read_csv(reference_file)
    _require_columns(generated, ("CS",), "Generated file")
    _require_columns(references, ("Hatespeech", "Style", "Counterspeech"), "Reference file")
    if alignment == "auto":
        alignment = "id" if "id" in generated[0] and "id" in references[0] else "keys"
    if alignment == "row":
        if allow_subset:
            raise ValueError("--allow_subset cannot be used with --alignment row.")
        if len(generated) != len(references):
            raise ValueError("Row alignment requires exactly equal row counts.")
        matched = references
    else:
        columns = ("id",) if alignment == "id" else ("Hatespeech", "Style")
        _require_columns(generated, columns, "Generated file")
        _require_columns(references, columns, "Reference file")
        gen_index = _index_rows(generated, columns, "Generated file")
        ref_index = _index_rows(references, columns, "Reference file")
        missing = gen_index.keys() - ref_index.keys()
        if missing:
            raise ValueError(f"{len(missing)} generated alignment key(s) have no reference.")
        unused = ref_index.keys() - gen_index.keys()
        if unused and not allow_subset:
            raise ValueError(f"{len(unused)} reference row(s) have no generation. "
                             "Use --allow_subset only for an intentional evaluation subset.")
        matched = [ref_index[tuple(clean(row[column]) for column in columns)] for row in generated]

    aligned = []
    for index, (generation, reference) in enumerate(zip(generated, matched), 1):
        for column in ("Hatespeech", "Style"):
            if column in generation and clean(generation[column]) != clean(reference[column]):
                raise ValueError(f"Generated row {index} has {column} inconsistent with its reference.")
        style = clean(reference["Style"])
        if style not in STYLE_LABELS:
            raise ValueError(f"Unsupported style {style!r} at reference for generated row {index}.")
        if not clean(reference["Hatespeech"]) or not clean(reference["Counterspeech"]):
            raise ValueError(f"Empty Hatespeech or Counterspeech reference for generated row {index}.")
        aligned.append({**generation, "Hatespeech": clean(reference["Hatespeech"]),
                        "Style": style, "CS": clean(generation["CS"]),
                        "Reference CS": clean(reference["Counterspeech"])})
    return aligned, {"alignment": alignment, "reference_file": str(reference_file),
                     "generated_rows": len(generated), "reference_rows": len(references),
                     "unused_reference_rows": len(references) - len(aligned)}


def tokenize_text(text):
    """Lowercase Unicode words, excluding punctuation, for basic overlap diagnostics."""
    return re.findall(r"\b\w+\b", clean(text).lower(), flags=re.UNICODE)


def compute_basic_metrics(data):
    rows = []
    for row in data:
        reference = tokenize_text(row["Reference CS"])
        hypothesis = tokenize_text(row["CS"])
        overlap = sum((Counter(reference) & Counter(hypothesis)).values())
        denominator = len(reference) + len(hypothesis)
        rows.append({"Exact Match": float(row["Reference CS"].casefold() == row["CS"].casefold()),
                     "Token F1": 2 * overlap / denominator if denominator else 0.0,
                     "Generated Tokens": len(hypothesis),
                     "Empty Generation": float(not row["CS"])})
    return rows


def extract_quotes(text):
    """Extract straight/curly double-quoted spans containing at least three words."""
    text = clean(text).replace("“", '"').replace("”", '"')
    return [quote for quote in re.findall(r'"([^"\n]+)"', text) if len(quote.split()) >= 3]


def remove_quotes(text):
    text = clean(text).replace("“", '"').replace("”", '"')
    return clean(re.sub(r'"[^"\n]*"', " ", text))


def _dependency(module, package=None):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(f"Evaluation dependency {package or module!r} is unavailable. "
                           "Install requirements-eval.txt, or use --metrics basic without --quotes_file.") from exc


def load_quotes(quotes_path):
    with open(quotes_path, encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        if "Gandhi" in data and "Mandela" in data:
            quotes = {style: data[style] for style in STYLE_LABELS}
        else:
            quotes = {style: data["authors"][author]["quotes"] for style, author in
                      (("Gandhi", "Mahatma Gandhi"), ("Mandela", "Nelson Mandela"))}
    except (KeyError, TypeError) as exc:
        raise ValueError("Quotes JSON must contain Gandhi/Mandela quote lists, or "
                         "authors -> Mahatma Gandhi/Nelson Mandela -> quotes.") from exc
    if any(not isinstance(items, list) or not items or
           any(not isinstance(item, str) or not clean(item) for item in items)
           for items in quotes.values()):
        raise ValueError("Each author's quotes must be a nonempty list of nonempty strings.")
    return quotes


def calculate_quote_metrics(data, quotes_data, threshold=80.0):
    """Macro-average best full-string quote similarities; values are on [0, 1]."""
    fuzz = _dependency("rapidfuzz.fuzz", "rapidfuzz")
    rows = []
    for row in data:
        generated = extract_quotes(row["CS"])
        references = extract_quotes(row["Reference CS"])
        scores, reference_scores, matches = [], [], []
        for quote in generated:
            normalized = clean(quote).casefold()
            similarity = max(fuzz.ratio(normalized, clean(candidate).casefold())
                             for candidate in quotes_data[row["Style"]])
            scores.append(similarity / 100.0)
            matches.append(float(similarity >= threshold))
            reference_scores.append(max((fuzz.ratio(normalized, clean(candidate).casefold())
                                         for candidate in references), default=0.0) / 100.0)
        rows.append({"Quote Conformity": mean(scores) if scores else 0.0,
                     "Quote Match Rate": mean(matches) if matches else 0.0,
                     "Quote Presence": float(bool(generated)),
                     "Reference Quote Match": mean(reference_scores) if reference_scores else 0.0})
    return rows


def _setup_full_environment(args):
    # Set these before importing Transformers/Hugging Face libraries.
    if args.hf_cache:
        os.environ["HF_HOME"] = str(Path(args.hf_cache).expanduser().resolve())
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def compute_full_metrics(data, args):
    """Compute standard library metrics and model scores, with no custom penalties."""
    _setup_full_environment(args)
    torch = _dependency("torch")
    nltk = _dependency("nltk")
    meteor_score = _dependency("nltk.translate.meteor_score").meteor_score
    rouge_scorer = _dependency("rouge_score.rouge_scorer", "rouge-score")
    bert_score = _dependency("bert_score", "bert-score")
    sentence_transformers = _dependency("sentence_transformers", "sentence-transformers")
    detoxify = _dependency("detoxify")
    transformers = _dependency("transformers")
    if args.nltk_data_path:
        nltk.data.path.insert(0, str(Path(args.nltk_data_path).expanduser().resolve()))
    try:
        nltk.corpus.wordnet.ensure_loaded()
    except LookupError as exc:
        raise RuntimeError("NLTK WordNet is missing. Run: python -m nltk.downloader wordnet omw-1.4 "
                           "(or supply --nltk_data_path). Evaluation does not auto-download NLTK data.") from exc
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for evaluation but is unavailable.")
    if args.offline and not args.toxicity_checkpoint:
        raise ValueError("--offline full metrics requires --toxicity_checkpoint for Detoxify.")
    if not Path(args.style_checkpoint).is_file():
        raise FileNotFoundError(f"Style checkpoint not found: {args.style_checkpoint}")
    if args.toxicity_checkpoint and not Path(args.toxicity_checkpoint).is_file():
        raise FileNotFoundError(f"Toxicity checkpoint not found: {args.toxicity_checkpoint}")

    references = [row["Reference CS"] for row in data]
    hypotheses = [row["CS"] for row in data]
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    metrics = []
    for reference, hypothesis in zip(references, hypotheses):
        scores = rouge.score(reference, hypothesis)
        # Treebank tokenization needs no punkt download; METEOR uses WordNet.
        reference_tokens = nltk.tokenize.TreebankWordTokenizer().tokenize(reference)
        hypothesis_tokens = nltk.tokenize.TreebankWordTokenizer().tokenize(hypothesis)
        metrics.append({"R-1": float(scores["rouge1"].fmeasure),
                        "R-2": float(scores["rouge2"].fmeasure),
                        "R-L": float(scores["rougeL"].fmeasure),
                        "M": float(meteor_score([reference_tokens], hypothesis_tokens))
                        if hypothesis_tokens else 0.0})

    logging.info("Computing BERTScore with one reusable model...")
    scorer = bert_score.BERTScorer(model_type=args.roberta_model_path,
                                  num_layers=args.bertscore_num_layers, device=str(device),
                                  batch_size=args.batch_size, rescale_with_baseline=False)
    _, _, bert_f1 = scorer.score(hypotheses, references, batch_size=args.batch_size)
    bert_values = bert_f1.tolist()
    if len(bert_values) != len(data):
        raise RuntimeError("BERTScore returned the wrong number of scores.")
    for row, score in zip(metrics, bert_values):
        row["BS"] = float(score)
    del scorer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logging.info("Computing sentence embedding cosine similarity...")
    sentence_model = sentence_transformers.SentenceTransformer(
        args.sentence_model_path, device=str(device), local_files_only=args.offline)
    reference_embeddings = sentence_model.encode(references, batch_size=args.batch_size,
                                                  convert_to_tensor=True, normalize_embeddings=True)
    generated_embeddings = sentence_model.encode(hypotheses, batch_size=args.batch_size,
                                                  convert_to_tensor=True, normalize_embeddings=True)
    similarities = (reference_embeddings * generated_embeddings).sum(dim=1).tolist()
    if len(similarities) != len(data):
        raise RuntimeError("Sentence model returned the wrong number of scores.")
    for row, similarity in zip(metrics, similarities):
        row["CoSim"] = float(similarity)  # Preserve legitimate negative cosine similarity.
    del sentence_model, reference_embeddings, generated_embeddings
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logging.info("Computing Detoxify toxicity...")
    toxicity_kwargs = {"device": str(device)}
    if args.toxicity_checkpoint:
        toxicity_kwargs["checkpoint"] = args.toxicity_checkpoint
    toxicity_model = detoxify.Detoxify("unbiased", **toxicity_kwargs)
    for start in range(0, len(data), args.batch_size):
        values = toxicity_model.predict(hypotheses[start:start + args.batch_size])["toxicity"]
        if len(values) != len(metrics[start:start + args.batch_size]):
            raise RuntimeError("Detoxify returned the wrong number of scores.")
        for row, value in zip(metrics[start:start + args.batch_size], values):
            row["T"] = float(value)
    del toxicity_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logging.info("Computing style probability with the supplied classifier...")
    tokenizer = transformers.RobertaTokenizer.from_pretrained(args.roberta_model_path,
                                                              local_files_only=args.offline)
    style_model = transformers.RobertaForSequenceClassification.from_pretrained(
        args.roberta_model_path, num_labels=2, local_files_only=args.offline)
    checkpoint = torch.load(args.style_checkpoint, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    style_model.load_state_dict(checkpoint, strict=True)
    style_model.to(device).eval()
    for start in range(0, len(data), args.batch_size):
        batch = data[start:start + args.batch_size]
        encoded = tokenizer([remove_quotes(row["CS"]) for row in batch], padding=True,
                            truncation=True, max_length=args.max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            probabilities = style_model(**encoded).logits.softmax(dim=-1)
        labels = torch.tensor([STYLE_LABELS[row["Style"]] for row in batch], device=device)
        true_probabilities = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1).tolist()
        for row, value in zip(metrics[start:start + args.batch_size], true_probabilities):
            row["Style Conformity"] = float(value)
    return metrics


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_results(input_file, output_dir, data, metrics, provenance, args):
    """Save every generation, macro means and a machine-readable run record."""
    if len(data) != len(metrics) or not metrics:
        raise ValueError("Metrics must contain exactly one nonempty result per generated row.")
    metric_names = list(metrics[0])
    for row in metrics:
        if set(row) != set(metric_names) or any(not math.isfinite(float(value)) for value in row.values()):
            raise RuntimeError("Evaluator returned missing or non-finite scores; no results were saved.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(input_file).stem
    metrics_path = output / f"{stem}_metrics.csv"
    summary_path = output / f"{stem}_summary.csv"
    metadata_path = output / f"{stem}_metadata.json"
    averages = {name: mean(row[name] for row in metrics) for name in metric_names}
    text_columns = list(dict.fromkeys(key for row in data for key in row))
    collision = set(text_columns) & set(metric_names)
    if collision:
        raise ValueError(f"Input columns conflict with metric output names: {', '.join(sorted(collision))}")
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=text_columns + metric_names)
        writer.writeheader()
        writer.writerows({**row, **scores} for row, scores in zip(data, metrics))
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Samples", *metric_names])
        writer.writeheader()
        writer.writerow({"Samples": len(data), **averages})
    versions = {}
    for package in ("torch", "transformers", "nltk", "rouge-score", "bert-score",
                    "sentence-transformers", "detoxify", "rapidfuzz"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "mode": args.metrics, "sample_count": len(data), **provenance,
                "input_sha256": _sha256(input_file),
                "reference_sha256": _sha256(provenance["reference_file"]),
                "arguments": vars(args), "package_versions": versions,
                "metric_definitions_version": 2,
                "aggregation": "Arithmetic mean over all generated rows, including empty outputs.",
                "notes": ["NSP is not implemented and is not reported.",
                          "Standard full metrics use no custom length/diversity penalties.",
                          "Full metric scores are not directly comparable to the original penalized script."]}
    if args.quotes_file:
        metadata["quotes_sha256"] = _sha256(args.quotes_file)
    if args.style_checkpoint and args.metrics == "full":
        metadata["style_checkpoint_sha256"] = _sha256(args.style_checkpoint)
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metrics_path, summary_path, metadata_path, averages


def evaluate(args):
    data, provenance = load_data(args.input_file, args.ground_truth_file,
                                 args.finetune_reference_file, bool(args.fine_tune),
                                 args.alignment, args.allow_subset)
    logging.info("Evaluating %s rows with %s alignment.", len(data), provenance["alignment"])
    quotes = load_quotes(args.quotes_file) if args.quotes_file else None
    if quotes is not None:
        _dependency("rapidfuzz.fuzz", "rapidfuzz")  # Fail before expensive full model loading.
    metrics = compute_basic_metrics(data)
    if args.metrics == "full":
        full_metrics = compute_full_metrics(data, args)
        expected = {"R-1", "R-2", "R-L", "M", "BS", "CoSim", "T", "Style Conformity"}
        if len(full_metrics) != len(data) or any(set(row) != expected for row in full_metrics):
            raise RuntimeError("Full evaluation returned incomplete scores; no results were saved.")
        for row, full in zip(metrics, full_metrics):
            row.update(full)
    if quotes is not None:
        for row, quote in zip(metrics, calculate_quote_metrics(data, quotes, args.quote_threshold)):
            row.update(quote)
    return save_results(args.input_file, args.output_dir, data, metrics, provenance, args)


def main(argv=None):
    args = parse_arguments(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        metrics_path, summary_path, metadata_path, averages = evaluate(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        logging.error("Evaluation failed: %s", exc)
        return 1
    print(f"Evaluation ({args.metrics}) completed.")
    for name, value in averages.items():
        print(f"{name}: {value:.6f}")
    print(f"Per-instance results: {metrics_path}")
    print(f"Summary: {summary_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
