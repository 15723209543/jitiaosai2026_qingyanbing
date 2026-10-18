"""submission3 的三模型投票与单文件权重回归测试。"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import median_probabilities, save_ensemble  # noqa: E402
from predict import collect_probabilities  # noqa: E402
from train import is_better  # noqa: E402


class TinyDataset(torch.utils.data.Dataset):
    labels = [0, 1]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.tensor([index], dtype=torch.float32), self.labels[index]


class FixedModel(torch.nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.register_buffer("probabilities", torch.tensor(probabilities))

    def forward(self, x):
        p = self.probabilities[x[:, 0].long()]
        return torch.log(torch.stack([1 - p, p], dim=1))


class EnsembleTests(unittest.TestCase):
    def test_median_keeps_probability_and_matches_two_of_three_votes(self):
        scores = median_probabilities([0.1, 0.8, 0.7],
                                      [0.9, 0.6, 0.2],
                                      [0.7, 0.3, 0.9])
        self.assertEqual(scores, [0.7, 0.6, 0.7])
        self.assertEqual([value >= 0.5 for value in scores], [True, True, True])

    def test_median_rejects_missing_model_or_invalid_probability(self):
        with self.assertRaises(ValueError):
            median_probabilities([0.1], [0.2])
        with self.assertRaises(ValueError):
            median_probabilities([0.1], [0.2], [float("nan")])

    def test_save_ensemble_creates_only_one_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "best_model.pth"
            models = {name: torch.nn.Linear(1, 2) for name in
                      ("resnet18", "densenet121", "efficientnet_b0")}
            save_ensemble(models, path, {"auroc": 0.91}, epoch=2)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(set(checkpoint["models"]), set(models))
            self.assertEqual(checkpoint["validation"]["auroc"], 0.91)
            self.assertEqual([item.name for item in Path(directory).iterdir()],
                             ["best_model.pth"])

    def test_prediction_uses_median_not_mean(self):
        models = {"resnet18": FixedModel([0.9, 0.2]),
                  "densenet121": FixedModel([0.1, 0.9]),
                  "efficientnet_b0": FixedModel([0.8, 0.8])}
        labels, scores = collect_probabilities(TinyDataset(), models, "valid")
        self.assertEqual(labels, [0, 1])
        self.assertAlmostEqual(scores[0], 0.8, places=5)
        self.assertAlmostEqual(scores[1], 0.8, places=5)

    def test_individual_model_update_uses_only_its_auroc(self):
        self.assertTrue(is_better({"auroc": 0.91}, {"auroc": 0.90}))
        self.assertFalse(is_better({"auroc": 0.89}, {"auroc": 0.90}))
        self.assertFalse(is_better({"auroc": 0.90}, {"auroc": 0.90}))


if __name__ == "__main__":
    unittest.main()
