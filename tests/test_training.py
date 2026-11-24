"""Training correctness and a real HF/PEFT checkpoint round trip on CPU."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from pulse_core import load_checkpoint_model, create_model_and_tokenizers, resolve_device, set_seed
from pulse_train import HateSpeechDataset, diffusion_loss, forward_process, read_training_rows, train, main
from scripts.smoke_pipeline import build_tiny_models


ROOT = Path(__file__).resolve().parents[1]


class MaskingTests(unittest.TestCase):
    def test_prompt_and_padding_are_never_corrupted(self):
        ids = torch.tensor([[4, 5, 6, 7, 0, 0]])
        eligible = torch.tensor([[False, False, True, True, False, False]])
        for seed in range(10):
            set_seed(seed)
            noisy, selected, probabilities = forward_process(ids, 2, eligible)
            self.assertTrue(torch.equal(noisy[~eligible], ids[~eligible]))
            self.assertFalse(selected[~eligible].any())
            self.assertTrue((probabilities > 0).all())

    def test_diffusion_loss_uses_response_lengths_and_probability(self):
        logits = torch.zeros(1, 4, 2, requires_grad=True)
        targets = torch.tensor([[0, 1, 0, 0]])
        selected = torch.tensor([[False, True, False, False]])
        eligible = torch.tensor([[False, True, True, False]])
        probabilities = torch.full((1, 4), 0.5)
        loss = diffusion_loss(logits, targets, selected, probabilities, eligible)
        self.assertAlmostEqual(loss.item(), 0.69314718, places=6)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[~selected], torch.zeros_like(logits.grad[~selected])))

    def test_invalid_style_fails_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Hatespeech", "Style", "Target", "Counterspeech"])
                writer.writeheader()
                writer.writerow(dict(Hatespeech="text", Style="Unknown", Target="people", Counterspeech="reply"))
            with self.assertRaisesRegex(ValueError, "unsupported style"):
                read_training_rows(path)

    def test_malformed_rows_duplicate_headers_and_ids_fail(self):
        examples = [
            ("Hatespeech,Style,Target,Counterspeech,Style\ntext,Gandhi,people,reply,Gandhi\n", "duplicate column"),
            ("Hatespeech,Style,Target,Counterspeech\ntext,Gandhi,people,reply,extra\n", "number of values"),
            ("id,Hatespeech,Style,Target,Counterspeech\nx,text,Gandhi,people,reply\nx,text,Mandela,people,reply\n", "duplicate id"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            for content, error in examples:
                path.write_text(content)
                with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                    read_training_rows(path)

    def test_nonfinite_cli_values_fail_before_loading_models(self):
        for option, value in (("--learning-rate", "nan"), ("--grad-clip", "inf"),
                              ("--target-loss-weight", "nan"), ("--vq-loss-weight", "inf")):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["--stage", "sslcl", "--train-csv", str(ROOT / "examples/train.csv"),
                      "--validate-only", option, value])
            self.assertEqual(error.exception.code, 2)


class ConfigurationTests(unittest.TestCase):
    def test_checkpoint_loader_pins_resolved_commit_without_downloads(self):
        metadata = dict(model_name="organization/model", revision="main", resolved_model_revision="frozen-commit")
        with patch("pulse_core.read_checkpoint_metadata", return_value=metadata), patch(
                "pulse_core.load_base_model", side_effect=RuntimeError("stopped before model loading")) as loader:
            with self.assertRaisesRegex(RuntimeError, "stopped before"):
                load_checkpoint_model("unused-checkpoint", device="cpu")
            loader.assert_called_once_with("organization/model", torch.device("cpu"), "none", "frozen-commit")
        # Older checkpoints may record only the requested revision.
        metadata["resolved_model_revision"] = None
        with patch("pulse_core.read_checkpoint_metadata", return_value=metadata), patch(
                "pulse_core.load_base_model", side_effect=RuntimeError("stopped before model loading")) as loader:
            with self.assertRaisesRegex(RuntimeError, "stopped before"):
                load_checkpoint_model("unused-checkpoint", device="cpu")
            self.assertEqual(loader.call_args.args[-1], "main")

    def test_local_base_override_does_not_reuse_remote_revision(self):
        metadata = dict(model_name="organization/model", revision="main", resolved_model_revision="frozen-commit")
        with tempfile.TemporaryDirectory() as local_model, patch(
                "pulse_core.read_checkpoint_metadata", return_value=metadata), patch(
                "pulse_core.load_base_model", side_effect=RuntimeError("stopped before model loading")) as loader:
            with self.assertRaisesRegex(RuntimeError, "stopped before"):
                load_checkpoint_model("unused-checkpoint", device="cpu", model_name=local_model)
            loader.assert_called_once_with(local_model, torch.device("cpu"), "none", None)
        self.assertEqual(metadata["model_name"], "organization/model")

    def test_sccg_accepts_inherited_resolved_revision(self):
        metadata = dict(stage="sslcl", target_labels=["people"], codebook_dim=8,
                        model_name="organization/model", revision=None, resolved_model_revision="frozen-commit")
        with patch("pulse_train.read_checkpoint_metadata", return_value=metadata), patch(
                "pulse_train.read_training_rows", return_value=([{}], ["people"])), patch(
                "pulse_train.load_checkpoint_model", side_effect=RuntimeError("revision accepted")):
            with self.assertRaisesRegex(RuntimeError, "revision accepted"):
                main(["--stage", "sccg", "--train-csv", "unused.csv", "--sslcl-checkpoint", "unused-checkpoint",
                      "--revision", "frozen-commit", "--device", "cpu", "--quantization", "none"])

    def test_cuda_selection_validates_index_and_sets_current_device(self):
        with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.device_count", return_value=2), patch("torch.cuda.set_device") as select:
            self.assertEqual(resolve_device("cuda:1"), torch.device("cuda:1"))
            select.assert_called_once_with(1)
            with self.assertRaisesRegex(ValueError, "index 2 is unavailable"):
                resolve_device("cuda:2")
            select.assert_called_once_with(1)
        with self.assertRaisesRegex(ValueError, "Invalid device"):
            resolve_device("cuda:invalid")


class CheckpointTests(unittest.TestCase):
    def test_training_updates_and_complete_checkpoint_roundtrip(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                base_path, style_path = build_tiny_models(directory)
                rows, labels = read_training_rows(ROOT / "examples/train.csv")
                model, tokenizer, style_tokenizer = create_model_and_tokenizers(
                    str(base_path), str(style_path), codebook_dim=8,
                    target_labels=labels, lora_target_modules=["c_attn"], lora_rank=2)
                initial_encoder = {k: v.clone() for k, v in model.style_encoder.state_dict().items()}
                initial_projection = model.pre_quant_projection.weight.detach().clone()
                checkpoint = train(model, tokenizer, style_tokenizer, rows, "sslcl",
                                   directory / "sslcl", epochs=1, batch_size=2,
                                   max_length=128, style_max_length=64, max_steps=1)
                self.assertTrue(any(not torch.equal(v, initial_encoder[k])
                                    for k, v in model.style_encoder.state_dict().items()))
                self.assertFalse(torch.equal(initial_projection, model.pre_quant_projection.weight))
                restored, restored_tokenizer, restored_style_tokenizer, metadata = load_checkpoint_model(
                    checkpoint, is_trainable=True)
                self.assertEqual(metadata["target_labels"], labels)
                self.assertEqual(metadata["codebook_dim"], 8)
                for name in ("llada_model", "style_encoder", "pre_quant_projection", "style_quantizer",
                             "style_projection", "fusion", "output_layer", "target_classifier"):
                    for key, value in getattr(model, name).state_dict().items():
                        self.assertTrue(torch.equal(value, getattr(restored, name).state_dict()[key]),
                                        f"Roundtrip changed {name}.{key}")
                sample = HateSpeechDataset(rows, tokenizer, style_tokenizer, "sccg", 128, 64)[0]
                arguments = dict(llada_input_ids=sample["llada_input_ids"].unsqueeze(0),
                                 bert_input_ids=sample["bert_input_ids"].unsqueeze(0),
                                 bert_attention_mask=sample["bert_attention_mask"].unsqueeze(0),
                                 style_label=sample["leader_label"].unsqueeze(0),
                                 attention_mask=sample["attention_mask"].unsqueeze(0))
                model.eval()
                restored.eval()
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(**arguments).logits, restored(**arguments).logits))
                before_style = {name: {key: value.clone() for key, value in getattr(restored, name).state_dict().items()}
                                for name in ("style_encoder", "pre_quant_projection", "style_quantizer")}
                initial_classifier = restored.target_classifier[-1].weight.detach().clone()
                train(restored, restored_tokenizer, restored_style_tokenizer, rows, "sccg",
                      directory / "sccg", epochs=1, batch_size=2, max_length=128,
                      style_max_length=64, max_steps=1)
                for name, state in before_style.items():
                    for key, value in state.items():
                        self.assertTrue(torch.equal(value, getattr(restored, name).state_dict()[key]),
                                        f"SCCG modified frozen {name}.{key}")
                self.assertFalse(torch.equal(initial_classifier, restored.target_classifier[-1].weight))
                self.assertFalse(restored.style_encoder.training)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
