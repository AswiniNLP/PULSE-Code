"""Offline regression tests for evaluation integrity, with no model downloads."""

import csv
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import metric


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reference = self.write_csv("references.csv", [
            {"id": "a", "Hatespeech": "First input", "Style": "Gandhi", "Counterspeech": "We all deserve dignity."},
            {"id": "b", "Hatespeech": "Second input", "Style": "Mandela", "Counterspeech": "Let us build peace."},
        ])

    def tearDown(self):
        self.temporary.cleanup()

    def write_csv(self, name, rows):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_id_alignment_reorders_references_and_preserves_empty_outputs(self):
        generated = self.write_csv("outputs.csv", [{"id": "b", "CS": ""},
                                                   {"id": "a", "CS": "WE ALL deserve dignity."}])
        rows, provenance = metric.load_data(generated, self.reference)
        self.assertEqual([row["Style"] for row in rows], ["Mandela", "Gandhi"])
        self.assertEqual(provenance["alignment"], "id")
        scores = metric.compute_basic_metrics(rows)
        self.assertEqual(scores[0]["Empty Generation"], 1.0)
        self.assertEqual(scores[0]["Token F1"], 0.0)
        self.assertEqual(scores[1]["Exact Match"], 1.0)
        self.assertEqual(scores[1]["Token F1"], 1.0)

    def test_keys_alignment_supports_legacy_columns_and_unicode(self):
        reference = self.write_csv("unicode.csv", [{"Hatespeech": "A café", "Style": "Gandhi",
                                                     "Counterspeech": "Everyone’s dignity matters."}])
        generated = self.write_csv("outputs.csv", [{"hatespeech": "A café", "style": "Gandhi",
                                                   "CS": "Everyone’s dignity matters."}])
        rows, provenance = metric.load_data(generated, reference)
        self.assertEqual(provenance["alignment"], "keys")
        self.assertEqual(rows[0]["Hatespeech"], "A café")
        self.assertEqual(metric.compute_basic_metrics(rows)[0]["Exact Match"], 1.0)

    def test_missing_outputs_require_explicit_subset_flag(self):
        generated = self.write_csv("outputs.csv", [{"id": "a", "CS": "one"}])
        with self.assertRaisesRegex(ValueError, "no generation"):
            metric.load_data(generated, self.reference)
        rows, provenance = metric.load_data(generated, self.reference, allow_subset=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(provenance["unused_reference_rows"], 1)

    def test_unmatched_or_duplicate_ids_are_errors(self):
        generated = self.write_csv("outputs.csv", [{"id": "unknown", "CS": "one"}])
        with self.assertRaisesRegex(ValueError, "no reference"):
            metric.load_data(generated, self.reference)
        generated = self.write_csv("outputs.csv", [{"id": "a", "CS": "one"}, {"id": "a", "CS": "two"}])
        with self.assertRaisesRegex(ValueError, "duplicate alignment"):
            metric.load_data(generated, self.reference)

    def test_ambiguous_hatespeech_style_pairs_require_ids(self):
        reference = self.write_csv("duplicate.csv", [
            {"Hatespeech": "Same", "Style": "Gandhi", "Counterspeech": "One reference"},
            {"Hatespeech": "Same", "Style": "Gandhi", "Counterspeech": "Another reference"},
        ])
        generated = self.write_csv("outputs.csv", [
            {"Hatespeech": "Same", "Style": "Gandhi", "CS": "One reference"},
            {"Hatespeech": "Same", "Style": "Gandhi", "CS": "Another reference"},
        ])
        with self.assertRaisesRegex(ValueError, "duplicate alignment"):
            metric.load_data(generated, reference)
        rows, _ = metric.load_data(generated, reference, alignment="row")
        self.assertEqual(len(rows), 2)

    def test_row_alignment_is_explicit_and_checks_count(self):
        generated = self.write_csv("outputs.csv", [{"CS": "one"}, {"CS": "two"}])
        with self.assertRaisesRegex(ValueError, "missing columns"):
            metric.load_data(generated, self.reference)
        rows, provenance = metric.load_data(generated, self.reference, alignment="row")
        self.assertEqual(len(rows), 2)
        self.assertEqual(provenance["alignment"], "row")
        generated = self.write_csv("outputs.csv", [{"CS": "one"}])
        with self.assertRaisesRegex(ValueError, "equal row counts"):
            metric.load_data(generated, self.reference, alignment="row")

    def test_id_match_does_not_hide_style_mismatch(self):
        generated = self.write_csv("outputs.csv", [{"id": "a", "Style": "Mandela", "CS": "one"},
                                                   {"id": "b", "Style": "Mandela", "CS": "two"}])
        with self.assertRaisesRegex(ValueError, "Style inconsistent"):
            metric.load_data(generated, self.reference)

    def test_empty_reference_and_unknown_style_are_errors(self):
        generated = self.write_csv("outputs.csv", [{"id": "a", "CS": "one"}])
        reference = self.write_csv("bad.csv", [{"id": "a", "Style": "Other", "Hatespeech": "x", "Counterspeech": "y"}])
        with self.assertRaisesRegex(ValueError, "Unsupported style"):
            metric.load_data(generated, reference)
        reference = self.write_csv("bad.csv", [{"id": "a", "Style": "Gandhi", "Hatespeech": "x", "Counterspeech": ""}])
        with self.assertRaisesRegex(ValueError, "Empty"):
            metric.load_data(generated, reference)

    def test_word_overlap_is_multiset_based(self):
        scores = metric.compute_basic_metrics([{"CS": "peace peace peace", "Reference CS": "peace now"}])[0]
        self.assertAlmostEqual(scores["Token F1"], 2 / 5)
        self.assertEqual(scores["Generated Tokens"], 3)

    def test_quoted_spans_are_preserved_by_cleaning(self):
        text = '“We deserve human dignity”'
        self.assertEqual(metric.extract_quotes(text), ["We deserve human dignity"])
        self.assertEqual(metric.extract_quotes('"We deserve human dignity"'), ["We deserve human dignity"])
        self.assertEqual(metric.remove_quotes('Say "we deserve human dignity" with care'), "Say with care")
        self.assertEqual(metric.clean("words\nneed space"), "words need space")

    def test_optional_dependency_failure_is_explanatory(self):
        with patch.object(importlib, "import_module", side_effect=ImportError("not installed")):
            with self.assertRaisesRegex(RuntimeError, "requirements-eval.txt"):
                metric._dependency("absent")

    def test_quotes_loader_accepts_both_schemas_and_rejects_empty_corpus(self):
        path = self.root / "quotes.json"
        compact = {"Gandhi": ["A test quote only"], "Mandela": ["Another invented test quote"]}
        path.write_text(json.dumps(compact), encoding="utf-8")
        self.assertEqual(metric.load_quotes(path), compact)
        legacy = {"authors": {"Mahatma Gandhi": {"quotes": compact["Gandhi"]},
                              "Nelson Mandela": {"quotes": compact["Mandela"]}}}
        path.write_text(json.dumps(legacy), encoding="utf-8")
        self.assertEqual(metric.load_quotes(path), compact)
        path.write_text(json.dumps({"Gandhi": [], "Mandela": []}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            metric.load_quotes(path)

    def test_full_orchestration_refuses_missing_scores_before_writing_results(self):
        generated = self.write_csv("outputs.csv", [{"id": "a", "CS": "one"}, {"id": "b", "CS": "two"}])
        path = self.root / "quotes.json"
        path.write_text(json.dumps({"Gandhi": ["A test quote only"], "Mandela": ["Another test quote only"]}),
                        encoding="utf-8")
        output_dir = self.root / "metrics"
        args = metric.parse_arguments(["--metrics", "full", "--input_file", str(generated),
                                       "--ground_truth_file", str(self.reference), "--quotes_file", str(path),
                                       "--style_checkpoint", "unused.pt", "--output_dir", str(output_dir)])
        with patch.object(metric, "_dependency"), patch.object(metric, "compute_full_metrics", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "incomplete scores"):
                metric.evaluate(args)
        self.assertFalse(output_dir.exists())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Real torch is needed for full API orchestration test")
    def test_full_api_orchestration_batches_and_preserves_raw_scores(self):
        # All external evaluators are mocks; no downloaded model is represented by these scores.
        import torch

        generated = self.write_csv("outputs.csv", [
            {"id": "a", "CS": '"A test quote only" We all deserve dignity.'},
            {"id": "b", "CS": "Let us build peace."},
        ])
        quote_path = self.root / "quotes.json"
        quote_path.write_text(json.dumps({"Gandhi": ["A test quote only"],
                                          "Mandela": ["Another test quote only"]}), encoding="utf-8")
        checkpoint = self.root / "style.pt"
        torch.save({"state_dict": {"classifier.weight": torch.ones(2, 1)}}, checkpoint)
        args = metric.parse_arguments(["--metrics", "full", "--input_file", str(generated),
                                       "--ground_truth_file", str(self.reference), "--quotes_file", str(quote_path),
                                       "--style_checkpoint", str(checkpoint), "--batch_size", "1", "--device", "cpu",
                                       "--output_dir", str(self.root / "full")])
        rouge = Mock()
        rouge.score.return_value = {name: SimpleNamespace(fmeasure=value) for name, value in
                                    (("rouge1", 0.8), ("rouge2", 0.6), ("rougeL", 0.7))}
        rouge_scorer = SimpleNamespace(RougeScorer=Mock(return_value=rouge))
        meteor_score = Mock(return_value=0.73)
        nltk = SimpleNamespace(corpus=SimpleNamespace(wordnet=SimpleNamespace(ensure_loaded=Mock())),
                               tokenize=SimpleNamespace(TreebankWordTokenizer=lambda: SimpleNamespace(tokenize=str.split)))
        scorer = Mock()
        scorer.score.return_value = (None, None, torch.tensor([0.85, 0.95]))
        bert_score = SimpleNamespace(BERTScorer=Mock(return_value=scorer))
        sentence_model = Mock()
        sentence_model.encode.side_effect = [torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
                                              torch.tensor([[-1.0, 0.0], [0.0, 1.0]])]
        sentence_transformers = SimpleNamespace(SentenceTransformer=Mock(return_value=sentence_model))
        toxicity_model = Mock()
        toxicity_model.predict.side_effect = [{"toxicity": [0.12]}, {"toxicity": [0.34]}]
        detoxify = SimpleNamespace(Detoxify=Mock(return_value=toxicity_model))
        tokenizer = Mock(return_value={"input_ids": torch.ones((1, 2), dtype=torch.long),
                                        "attention_mask": torch.ones((1, 2), dtype=torch.long)})
        style_model = Mock()
        style_model.to.return_value = style_model
        style_model.eval.return_value = style_model
        style_model.return_value = SimpleNamespace(logits=torch.tensor([[1.0986123, 0.0]]))
        transformers = SimpleNamespace(
            RobertaTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
            RobertaForSequenceClassification=SimpleNamespace(from_pretrained=Mock(return_value=style_model)))
        fuzz = SimpleNamespace(ratio=lambda a, b: 100.0 if a == b else 0.0)
        modules = {"torch": torch, "nltk": nltk,
                   "nltk.translate.meteor_score": SimpleNamespace(meteor_score=meteor_score),
                   "rouge_score.rouge_scorer": rouge_scorer, "bert_score": bert_score,
                   "sentence_transformers": sentence_transformers, "detoxify": detoxify,
                   "transformers": transformers, "rapidfuzz.fuzz": fuzz}
        with patch.object(metric, "_dependency", side_effect=lambda module, package=None: modules[module]):
            metrics_path, _, _, averages = metric.evaluate(args)
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[0]["CoSim"]), -1.0)
        self.assertEqual(averages["CoSim"], -0.5)
        self.assertEqual(float(rows[0]["R-1"]), 0.8)
        self.assertAlmostEqual(averages["M"], 0.73)
        self.assertAlmostEqual(float(rows[0]["Style Conformity"]), 0.75, places=6)
        self.assertAlmostEqual(float(rows[1]["Style Conformity"]), 0.25, places=6)
        self.assertEqual(averages["Quote Conformity"], 0.5)
        self.assertEqual(averages["Quote Match Rate"], 0.5)
        self.assertAlmostEqual(averages["T"], 0.23)
        scorer.score.assert_called_once_with([row["CS"] for row in metric._read_csv(generated)],
                                             ["We all deserve dignity.", "Let us build peace."], batch_size=1)
        detoxify.Detoxify.assert_called_once_with("unbiased", device="cpu")
        style_model.load_state_dict.assert_called_once()
        self.assertTrue(style_model.load_state_dict.call_args.kwargs["strict"])
        self.assertNotIn("A test quote only", tokenizer.call_args_list[0].args[0][0])

    def test_basic_cli_saves_both_rows_and_provenance(self):
        generated = self.write_csv("outputs.csv", [{"id": "a", "CS": "We all deserve dignity."},
                                                   {"id": "b", "CS": ""}])
        output_dir = self.root / "metrics"
        args = metric.parse_arguments(["--metrics", "basic", "--input_file", str(generated),
                                       "--ground_truth_file", str(self.reference), "--output_dir", str(output_dir)])
        metrics_file, summary_file, metadata_file, averages = metric.evaluate(args)
        with metrics_file.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(averages["Empty Generation"], 0.5)
        self.assertEqual(averages["Exact Match"], 0.5)
        self.assertTrue(summary_file.is_file())
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        self.assertEqual(metadata["sample_count"], 2)
        self.assertEqual(metadata["metric_definitions_version"], 2)
        self.assertEqual(len(metadata["input_sha256"]), 64)
        self.assertNotIn("NSP", averages)
        self.assertNotIn("BS", averages)

    def test_help_and_basic_mode_work_with_site_packages_disabled(self):
        script = Path(metric.__file__)
        help_run = subprocess.run([sys.executable, "-S", str(script), "--help"],
                                  capture_output=True, text=True, check=False)
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        generated = self.write_csv("outputs.csv", [{"id": "a", "CS": "We all deserve dignity."},
                                                   {"id": "b", "CS": "Let us build peace."}])
        run = subprocess.run([sys.executable, "-S", str(script), "--metrics", "basic", "--input_file", str(generated),
                              "--ground_truth_file", str(self.reference), "--output_dir", str(self.root / "run")],
                             capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("Exact Match: 1.000000", run.stdout)


if __name__ == "__main__":
    unittest.main()
