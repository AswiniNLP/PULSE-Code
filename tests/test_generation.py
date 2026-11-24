"""Offline regression tests for masked-token sampling and inference CSV I/O."""

import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from generate_vq import (add_gumbel_noise, generate, get_num_transfer_tokens,
                         validate_generation_options)
from pulse_inference import read_inputs, write_predictions


class RecordingPredictor(torch.nn.Module):
    """Small mask predictor with different output tokens for each style."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, tokens, **kwargs):
        self.calls.append((tokens.clone(), {key: value.clone() if value is not None else None
                                            for key, value in kwargs.items()}))
        logits = torch.full((*tokens.shape, 10), -5.0)
        labels = kwargs["style_label"]
        for row in range(tokens.shape[0]):
            logits[row, :, 1 + int(labels[row])] = 5
        # The sampler must suppress mask predictions rather than leave holes.
        logits[..., 9] = 100
        return SimpleNamespace(logits=logits)


class GenerationTests(unittest.TestCase):
    def test_gumbel_scores_are_stable_at_fractional_temperature(self):
        logits = torch.tensor([1000.0, -1000.0, 0.0])
        torch.manual_seed(8)
        scores = add_gumbel_noise(logits, 0.6)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(int(scores.argmax()), 0)
        self.assertIs(add_gumbel_noise(logits, 0), logits)
        with patch("torch.rand_like", return_value=torch.zeros(3, dtype=torch.float64)):
            self.assertTrue(torch.isfinite(add_gumbel_noise(logits, 0.6)).all())

    def test_transfer_schedule_preserves_each_row_count(self):
        masks = torch.tensor([[True, True, True, True, True],
                              [True, False, False, False, False]])
        schedule = get_num_transfer_tokens(masks, 3)
        self.assertEqual(schedule.tolist(), [[2, 2, 1], [1, 0, 0]])

    def test_blockwise_batch_generation_preserves_conditioning(self):
        predictor = RecordingPredictor()
        prompts = torch.tensor([[3, 4], [0, 5]])
        result = generate(
            predictor, prompts, steps=4, gen_length=6, block_length=3,
            mask_id=9, style_label=torch.tensor([0, 1]),
            attention_mask=torch.tensor([[1, 1], [0, 1]]),
            bert_input_ids=torch.tensor([[4], [5]]),
            bert_attention_mask=torch.ones(2, 1, dtype=torch.long),
        )
        self.assertTrue(torch.equal(result[:, :2], prompts))
        self.assertEqual(result[:, 2:].tolist(), [[1] * 6, [2] * 6])
        self.assertEqual(len(predictor.calls), 4)
        first_tokens, _ = predictor.calls[0]
        self.assertTrue((first_tokens[:, 2:] == 9).all())
        third_tokens, kwargs = predictor.calls[2]
        self.assertEqual(third_tokens[:, 2:5].tolist(), [[1] * 3, [2] * 3])
        self.assertTrue((third_tokens[:, 5:] == 9).all())
        self.assertEqual(kwargs["attention_mask"].shape, result.shape)
        self.assertEqual(kwargs["attention_mask"][1, 0].item(), 0)

    def test_classifier_free_guidance_duplicates_all_style_inputs(self):
        predictor = RecordingPredictor()
        prompts = torch.tensor([[3, 4], [0, 5]])
        result = generate(
            predictor, prompts, steps=2, gen_length=2, block_length=2,
            cfg_scale=1.5, mask_id=9, style_label=torch.tensor([0, 1]),
            bert_input_ids=torch.tensor([[4], [5]]),
            bert_attention_mask=torch.ones(2, 1, dtype=torch.long),
            attention_mask=torch.ones_like(prompts),
        )
        tokens, kwargs = predictor.calls[0]
        self.assertEqual(tokens.shape[0], 4)
        self.assertEqual(kwargs["style_label"].tolist(), [0, 1, 0, 1])
        self.assertEqual(kwargs["bert_input_ids"].tolist(), [[4], [5], [4], [5]])
        self.assertTrue((tokens[2:, :2] == 9).all())
        self.assertEqual(result[:, 2:].tolist(), [[1, 1], [2, 2]])

    def test_invalid_schedule_and_prompt_fail_before_model_execution(self):
        for values in [(0, 4, 2), (4, 5, 2), (3, 8, 2), (10, 8, 2)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_generation_options(*values)
        predictor = RecordingPredictor()
        with self.assertRaisesRegex(ValueError, "mask token"):
            generate(predictor, torch.tensor([[9]]), mask_id=9,
                     gen_length=2, block_length=2, steps=2)
        self.assertEqual(predictor.calls, [])


class InferenceCsvTests(unittest.TestCase):
    def test_preserves_ids_references_and_quoted_text(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            destination = Path(directory) / "out" / "predictions.csv"
            original = {"id": "sample-1", "Hatespeech": "A quoted,\ninput", "Style": "Gandhi",
                        "Counterspeech": "Reference text"}
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(original))
                writer.writeheader()
                writer.writerow(original)
            rows, columns = read_inputs(source)
            rows[0]["CS"] = "Generated,\ncounterspeech"
            write_predictions(destination, rows, columns)
            with destination.open(newline="") as handle:
                generated = list(csv.DictReader(handle))
            self.assertEqual(generated, [{**original, "CS": "Generated,\ncounterspeech"}])
            self.assertEqual(list(destination.parent.glob(".predictions-*")), [])

    def test_rejects_unknown_styles_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("Hatespeech,Style\nInput,Unknown\n")
            with self.assertRaisesRegex(ValueError, "Gandhi or Mandela"):
                read_inputs(path)

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("id,Hatespeech,Style\n1,Input,Gandhi\n1,Other,Mandela\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                read_inputs(path)


if __name__ == "__main__":
    unittest.main()
