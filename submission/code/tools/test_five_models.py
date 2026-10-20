"""五模型独立轮次与按概率排名加权预测的关键行为测试。"""

import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import ImageModel  # noqa: E402
from predict import collect_probabilities, evaluate, predict  # noqa: E402
from train import (create_criterion, get_model_learning_rate, is_better,
                   print_training_profiles, train, training_order)  # noqa: E402
from utils import EPOCHS, MODEL_NAMES, MODEL_TRAINING, print_rank_weights, rank_weighted_probabilities, save_ensemble  # noqa: E402


class TinyDataset(torch.utils.data.Dataset):
    labels = [0, 1]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.tensor([index], dtype=torch.float32), self.labels[index]

    def preload(self):
        pass


class FourValidationDataset(torch.utils.data.Dataset):
    labels = [0, 1, 0, 1]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.tensor([index], dtype=torch.float32), self.labels[index]


class FixedModel(torch.nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.register_buffer("probabilities", torch.tensor(probabilities))

    def forward(self, x):
        p = self.probabilities[x[:, 0].long()]
        return torch.log(torch.stack([1 - p, p], dim=1))


class BrightnessModel(torch.nn.Module):
    def forward(self, x):
        value = x[:, 0, 0, 0]
        return torch.stack([-value, value], dim=1)


class FiveModelTests(unittest.TestCase):
    def test_four_models_have_independent_finetune_profiles(self):
        self.assertEqual(EPOCHS["wide5_slim"], 30)
        self.assertEqual(MODEL_TRAINING["wide5_slim"]["positive_share"], 0.25)
        self.assertEqual(len({id(MODEL_TRAINING[name]) for name in MODEL_NAMES}), 5)
        for name in MODEL_NAMES[1:]:
            self.assertGreater(EPOCHS[name], 30)
            self.assertGreater(MODEL_TRAINING[name]["positive_share"], 0.25)
            self.assertAlmostEqual(get_model_learning_rate(name, 31),
                                   MODEL_TRAINING[name]["finetune_lr"])

    def test_training_start_prints_each_model_profile(self):
        output = StringIO()
        with redirect_stdout(output):
            print_training_profiles()
        result = output.getvalue()
        for name in MODEL_NAMES:
            self.assertIn(name, result)
        self.assertIn("梯度截断", result)
        self.assertIn("阳性损失占比", result)

    def test_four_model_criteria_are_separate_and_focus_hard_positives(self):
        labels = [0] * 30 + [1]
        cnn = create_criterion(labels, "wide5_slim")
        resnet = create_criterion(labels, "resnet18")
        densenet = create_criterion(labels, "densenet121")
        self.assertIsNot(resnet, densenet)
        logits = torch.tensor([[4.0, -4.0], [4.0, -4.0]], requires_grad=True)
        targets = torch.tensor([0, 1])
        loss = resnet(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), cnn(logits.detach(), targets).item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_names_and_cnn_are_distinct(self):
        self.assertEqual(len(MODEL_NAMES), 5)
        self.assertIn("wide5_slim", MODEL_NAMES)
        self.assertIn("convnext_tiny", MODEL_NAMES)
        model = ImageModel("wide5_slim")
        self.assertEqual(model.network.conv5.out_channels, 512)
        self.assertEqual(tuple(model(torch.zeros(1, 3, 64, 64)).shape), (1, 2))

    def test_equal_rank_weights_average_five_probabilities(self):
        with patch("utils.RANK_WEIGHTS", (0.2,) * 5):
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
        with patch("utils.RANK_WEIGHTS", (0.2,) * 5):
            labels, scores = collect_probabilities(TinyDataset(), models, "valid")
        self.assertEqual(labels, [0, 1])
        self.assertAlmostEqual(scores[0], 0.55, places=5)
        self.assertAlmostEqual(scores[1], 0.45, places=5)

    def test_rank_weights_are_printed_in_descending_probability_order(self):
        with patch("utils.RANK_WEIGHTS", (0.3, 0.3, 0.2, 0.1, 0.1)):
            output = StringIO()
            with redirect_stdout(output):
                print_rank_weights()
        self.assertIn("高到低", output.getvalue())
        self.assertIn("0.3, 0.3, 0.2, 0.1, 0.1", output.getvalue())

    def test_predict_validates_labeled_images_before_writing_test_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_dir, test_dir = root / "valid", root / "test"
            valid_dir.mkdir()
            test_dir.mkdir()
            for folder in (valid_dir, test_dir):
                Image.new("RGB", (64, 64), "black").save(folder / "negative.png")
                Image.new("RGB", (64, 64), "white").save(folder / "positive.png")
            valid_csv, test_csv = root / "valid.csv", root / "test.csv"
            valid_csv.write_text("Image_name,True_Label\nnegative.png,0\npositive.png,1\n", encoding="utf-8")
            test_csv.write_text("Image_name\nnegative.png\npositive.png\n", encoding="utf-8")
            model_path, output_csv = root / "best_model.pth", root / "submission.csv"
            model_path.touch()
            models = {name: BrightnessModel() for name in MODEL_NAMES}
            with patch("predict.MODEL_PATH", model_path), \
                 patch("predict.VALID_CSV", valid_csv, create=True), \
                 patch("predict.VALID_DIR", valid_dir, create=True), \
                 patch("predict.TEST_CSV", test_csv), \
                 patch("predict.load_ensemble", return_value=(models, {"complete": True})):
                output = StringIO()
                with redirect_stdout(output):
                    predict(test_dir=test_dir, output_csv=output_csv)
            self.assertIn("开始验证", output.getvalue())
            self.assertLess(output.getvalue().index("开始验证"), output.getvalue().index("开始读取test"))
            self.assertIn("AUROC: 1.000000", output.getvalue())
            self.assertIn("真实阳性:1, 报对阳性:1, 误报:0, 漏报:0", output.getvalue())
            self.assertNotIn("Acc:", output.getvalue())
            self.assertEqual(output_csv.read_text(encoding="utf-8").splitlines()[0],
                             "Image_name,Probability,Pred_Label")

    def test_validation_prints_positive_counts_without_accuracy(self):
        models = {MODEL_NAMES[0]: FixedModel([0.1, 0.9])}
        output = StringIO()
        with redirect_stdout(output):
            evaluate(TinyDataset(), models)
        result = output.getvalue()
        self.assertIn("真实阳性:1", result)
        self.assertIn("报对阳性:1", result)
        self.assertIn("误报:0", result)
        self.assertIn("漏报:0", result)
        self.assertIn("AUROC: 1.000000", result)
        self.assertNotIn("Acc:", result)

    def test_validation_logs_interval_counts_not_running_totals(self):
        models = {MODEL_NAMES[0]: FixedModel([0.1, 0.9, 0.8, 0.2])}
        output = StringIO()
        with patch("predict.BATCH_SIZE", 1), patch("utils.LOG_EVERY", 2), redirect_stdout(output):
            collect_probabilities(FourValidationDataset(), models, "valid")
        lines = [line for line in output.getvalue().splitlines() if line.startswith("valid, batch:")]
        self.assertEqual(len(lines), 3)
        for index, line in zip((1, 2, 4), lines):
            self.assertIn(f"batch: {index}/4", line)
            self.assertIn("time:", line)
            self.assertIn("remaining:", line)
        self.assertIn("真实阳性:0, 报对阳性:0, 误报:0", lines[0])
        self.assertIn("真实阳性:1, 报对阳性:1, 误报:0", lines[1])
        self.assertIn("真实阳性:1, 报对阳性:0, 误报:1", lines[2])
        self.assertNotIn("loss:", lines[2])

    def test_intermediate_validation_hides_total_counts_but_returns_metrics(self):
        output = StringIO()
        with redirect_stdout(output):
            metrics = evaluate(TinyDataset(), {MODEL_NAMES[0]: FixedModel([0.1, 0.9])},
                               show_counts=False)
        self.assertEqual(metrics["tp"], 1)
        self.assertIn("AUROC: 1.000000", output.getvalue())
        self.assertNotIn("漏报:", output.getvalue())

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
        show_counts_calls = []

        def fake_evaluate(dataset, models, show_counts=True):
            calls.append(tuple(models))
            show_counts_calls.append(show_counts)
            value = 0.5 if len(calls) <= 6 else (0.7 if len(models) == 5 else 0.6)
            return {"auroc": value, "auprc": value, "tp": 1, "fp": 0, "fn": 1}

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)
            with patch("train.MODEL_PATH", path / "model" / "best_model.pth"), \
                 patch("train.RECORD_DIR", path / "records"), \
                 patch("train.EPOCHS", {name: 1 for name in MODEL_NAMES}), \
                 patch("train.ImageModel", lambda name: torch.nn.Linear(1, 2)), \
                 patch("train.evaluate", fake_evaluate):
                output = StringIO()
                with redirect_stdout(output):
                    result = train(TinyDataset(), TinyDataset())
            checkpoint = torch.load(path / "model" / "best_model.pth",
                                    map_location="cpu", weights_only=True)
            self.assertEqual(len(calls), 12)
            self.assertEqual(calls[5], MODEL_NAMES)
            self.assertEqual(calls[-1], MODEL_NAMES)
            self.assertTrue(all(not show_counts for show_counts in show_counts_calls))
            self.assertEqual(result["auroc"], 0.7)
            final_line = next(line for line in output.getvalue().splitlines()
                              if line.startswith("训练结束,"))
            self.assertIn("真实阳性:2, 报对阳性:1, 误报:0, 漏报:1", final_line)
            self.assertIn("真实阳性:", output.getvalue())
            self.assertIn("误报:", output.getvalue())
            self.assertNotIn("acc:", output.getvalue())
            self.assertNotIn("Acc:", output.getvalue())
            self.assertTrue(checkpoint["complete"])
            self.assertEqual(len(checkpoint["individual_models"]), 5)


if __name__ == "__main__":
    unittest.main()
