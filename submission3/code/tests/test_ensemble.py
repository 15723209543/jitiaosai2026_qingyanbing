"""submission3 的三模型投票与单文件权重回归测试。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import ImageDataset, median_probabilities, save_ensemble  # noqa: E402
from predict import collect_probabilities  # noqa: E402
from train import create_class_weights, create_train_loader, is_better, training_order  # noqa: E402


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

    def test_untrained_model_runs_first_then_each_round_has_three_models(self):
        order = list(training_order({"resnet18": {"auroc": 0.94},
                                     "densenet121": {"auroc": 0.93}}, 2))
        self.assertEqual(order, [(1, "efficientnet_b0"), (1, "resnet18"),
                                 (1, "densenet121"), (2, "resnet18"),
                                 (2, "densenet121"), (2, "efficientnet_b0")])

    def test_each_epoch_uses_every_training_image_once(self):
        class NumberedDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 65

            def __getitem__(self, index):
                return index

        dataloader = create_train_loader(NumberedDataset())
        self.assertEqual(len(dataloader), 2)
        for _ in range(2):
            indices = [int(index) for batch in dataloader for index in batch]
            self.assertEqual(sorted(indices), list(range(65)))

    def test_preload_caches_resized_images_before_training(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            paths = []
            for index in range(3):
                path = Path(directory) / f"{index}.png"
                Image.new("RGB", (80, 80), (index * 40, 10, 20)).save(path)
                paths.append(path)
            dataset = ImageDataset(paths, [0, 1, 0], training=True)
            dataset.preload(workers=2)
            self.assertEqual(set(dataset.cache), {0, 1, 2})
            self.assertEqual([dataset.cache[index].size for index in range(3)],
                             [(64, 64)] * 3)
            x, y = dataset[1]
            self.assertEqual(tuple(x.shape), (3, 64, 64))
            self.assertEqual(int(y), 1)

    def test_full_dataset_loss_retains_positive_class_balance(self):
        weights = create_class_weights([0] * 9 + [1])
        self.assertAlmostEqual(weights[0].item() * 9, 0.75, places=6)
        self.assertAlmostEqual(weights[1].item(), 0.25, places=6)

    def test_checkpoint_save_retries_temporary_windows_lock(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "best_model.pth"
            models = {name: torch.nn.Linear(1, 2) for name in
                      ("resnet18", "densenet121", "efficientnet_b0")}
            original_replace = Path.replace
            attempts = []

            def replace_after_unlock(source, target):
                attempts.append((source, target))
                if len(attempts) < 3:
                    raise PermissionError("[WinError 5] target temporarily locked")
                return original_replace(source, target)

            with patch.object(Path, "replace", replace_after_unlock), patch("utils.time.sleep"):
                save_ensemble(models, path, {"auroc": 0.91})
            self.assertEqual(len(attempts), 3)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()
