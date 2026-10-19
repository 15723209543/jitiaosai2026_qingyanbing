"""五模型独立轮次与按概率排名加权预测的关键行为测试。"""

import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import ImageModel  # noqa: E402
from predict import collect_probabilities  # noqa: E402
from train import is_better, train, training_order  # noqa: E402
from utils import MODEL_NAMES, rank_weighted_probabilities, save_ensemble  # noqa: E402


class TinyDataset(torch.utils.data.Dataset):
    labels = [0, 1]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.tensor([index], dtype=torch.float32), self.labels[index]

    def preload(self):
        pass


class FixedModel(torch.nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.register_buffer("probabilities", torch.tensor(probabilities))

    def forward(self, x):
        p = self.probabilities[x[:, 0].long()]
        return torch.log(torch.stack([1 - p, p], dim=1))


class FiveModelTests(unittest.TestCase):
    def test_names_and_cnn_are_distinct(self):
        self.assertEqual(len(MODEL_NAMES), 5)
        self.assertIn("wide5_slim", MODEL_NAMES)
        self.assertIn("convnext_tiny", MODEL_NAMES)
        model = ImageModel("wide5_slim")
        self.assertEqual(model.network.conv5.out_channels, 512)
        self.assertEqual(tuple(model(torch.zeros(1, 3, 64, 64)).shape), (1, 2))

    def test_equal_rank_weights_average_five_probabilities(self):
        scores = rank_weighted_probabilities([0.9, 0.9], [0.8, 0.1], [0.7, 0.2],
                                             [0.1, 0.3], [0.2, 0.4])
        self.assertAlmostEqual(scores[0], 0.54)
        self.assertAlmostEqual(scores[1], 0.38)
        self.assertEqual([x >= 0.5 for x in scores], [True, False])
        with self.assertRaises(ValueError):
            rank_weighted_probabilities([0.9], [0.8], [0.7])

    def test_custom_rank_weights_apply_to_sorted_values_not_model_names(self):
        with patch("utils.RANK_WEIGHTS", (0.4, 0.3, 0.2, 0.1, 0.0)):
            score = rank_weighted_probabilities([0.1], [0.9], [0.3], [0.5], [0.7])
        self.assertAlmostEqual(score[0], 0.70)
        with patch("utils.RANK_WEIGHTS", (0.4, 0.3, 0.2, 0.1, 0.1)):
            with self.assertRaises(ValueError):
                rank_weighted_probabilities([0.1], [0.9], [0.3], [0.5], [0.7])

    def test_predict_uses_weighted_rank_score(self):
        models = {name: FixedModel([p, 1 - p]) for name, p in zip(
            MODEL_NAMES, [0.95, 0.85, 0.75, 0.15, 0.05])}
        labels, scores = collect_probabilities(TinyDataset(), models, "valid")
        self.assertEqual(labels, [0, 1])
        self.assertAlmostEqual(scores[0], 0.55, places=5)
        self.assertAlmostEqual(scores[1], 0.45, places=5)

    def test_zero_round_skips_branch_and_rounds_interleave(self):
        rounds = dict.fromkeys(MODEL_NAMES, 2)
        rounds["wide5_slim"] = 0
        rounds["convnext_tiny"] = 1
        order = list(training_order(rounds))
        self.assertEqual(order, [(1, name) for name in MODEL_NAMES if rounds[name] > 0] +
                         [(2, name) for name in MODEL_NAMES if rounds[name] > 1])

    def test_completed_epochs_prevent_retraining_exhausted_model_after_restart(self):
        rounds = dict.fromkeys(MODEL_NAMES, 3)
        completed = {name: 3 for name in MODEL_NAMES}
        completed["densenet121"] = 1
        completed["convnext_tiny"] = 2
        self.assertEqual(list(training_order(rounds, completed)),
                         [(2, "densenet121"), (3, "densenet121"), (3, "convnext_tiny")])

    def test_auroc_alone_selects_both_individual_and_combination(self):
        self.assertTrue(is_better({"auroc": 0.91}, {"auroc": 0.90}))
        self.assertFalse(is_better({"auroc": 0.90, "accuracy": 1},
                                   {"auroc": 0.90, "accuracy": 0}))

    def test_single_file_has_individual_and_combined_snapshots(self):
        models = {name: torch.nn.Linear(1, 2) for name in MODEL_NAMES}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "best_model.pth"
            save_ensemble(models, models, path, {"auroc": 0.92},
                          {"resnet18": {"auroc": 0.93}}, epoch=2,
                          completed_epochs={name: 2 for name in MODEL_NAMES})
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(set(checkpoint["individual_models"]), set(MODEL_NAMES))
            self.assertEqual(set(checkpoint["ensemble_models"]), set(MODEL_NAMES))
            self.assertEqual(checkpoint["completed_epochs"]["resnet18"], 2)
            self.assertEqual([item.name for item in path.parent.iterdir()], ["best_model.pth"])

    def test_train_validates_baseline_then_every_branch_and_whole_group(self):
        calls = []

        def fake_evaluate(dataset, models):
            calls.append(tuple(models))
            value = 0.5 if len(calls) <= 6 else (0.7 if len(models) == 5 else 0.6)
            return {"auroc": value, "auprc": value, "tp": 1, "fp": 0}

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)
            with patch("train.MODEL_PATH", path / "model" / "best_model.pth"), \
                 patch("train.RECORD_DIR", path / "records"), \
                 patch("train.EPOCHS", {name: 1 for name in MODEL_NAMES}), \
                 patch("train.ImageModel", lambda name: torch.nn.Linear(1, 2)), \
                 patch("train.load_bootstrap"), patch("train.evaluate", fake_evaluate):
                with redirect_stdout(StringIO()):
                    result = train(TinyDataset(), TinyDataset())
            checkpoint = torch.load(path / "model" / "best_model.pth",
                                    map_location="cpu", weights_only=True)
            self.assertEqual(len(calls), 12)
            self.assertEqual(calls[5], MODEL_NAMES)
            self.assertEqual(calls[-1], MODEL_NAMES)
            self.assertEqual(result["auroc"], 0.7)
            self.assertTrue(checkpoint["complete"])
            self.assertEqual(len(checkpoint["individual_models"]), 5)


if __name__ == "__main__":
    unittest.main()
