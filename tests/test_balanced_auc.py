"""submission2 的训练期采样、AUC 损失和旧参数暖启动；不用比赛图片。"""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "submission2_auc_utils", ROOT / "submission2" / "code" / "utils.py")
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)
MODEL_SPEC = importlib.util.spec_from_file_location(
    "submission2_auc_model", ROOT / "submission2" / "code" / "model.py")
candidate_model = importlib.util.module_from_spec(MODEL_SPEC)
MODEL_SPEC.loader.exec_module(candidate_model)
TRAIN_SPEC = importlib.util.spec_from_file_location(
    "submission2_auc_train", ROOT / "submission2" / "code" / "train.py")
candidate_train = importlib.util.module_from_spec(TRAIN_SPEC)
with patch.dict(sys.modules, {"utils": utils, "model": candidate_model}):
    TRAIN_SPEC.loader.exec_module(candidate_train)
PREDICT_SPEC = importlib.util.spec_from_file_location(
    "submission2_auc_predict", ROOT / "submission2" / "code" / "predict.py")
candidate_predict = importlib.util.module_from_spec(PREDICT_SPEC)
with patch.dict(sys.modules, {"utils": utils, "model": candidate_model}):
    PREDICT_SPEC.loader.exec_module(candidate_predict)
COST_SPEC = importlib.util.spec_from_file_location(
    "submission2_auc_cost", ROOT / "submission2" / "cost.py")
candidate_cost = importlib.util.module_from_spec(COST_SPEC)
with patch.dict(sys.modules, {"utils": utils, "predict": candidate_predict}):
    COST_SPEC.loader.exec_module(candidate_cost)


class BalancedSamplerTests(unittest.TestCase):
    def test_all_examples_are_covered_and_every_batch_has_both_classes(self):
        labels = [0] * 23 + [1] * 2
        sampler = utils.BalancedBatchSampler(labels, batch_size=4, seed=17)
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(set(sum(batches, [])), set(range(25)))
        self.assertTrue(all(2 <= len(batch) <= 4 for batch in batches))
        self.assertTrue(all({labels[i] for i in batch} == {0, 1} for batch in batches))
        self.assertEqual(sum(labels[i] == 0 for batch in batches for i in batch), 23)

    def test_same_seed_and_epoch_are_reproducible_but_next_epoch_changes_order(self):
        a = utils.BalancedBatchSampler([0] * 23 + [1] * 2, batch_size=4, seed=17)
        b = utils.BalancedBatchSampler([0] * 23 + [1] * 2, batch_size=4, seed=17)
        self.assertEqual(list(a), list(b))
        a.set_epoch(2)
        self.assertNotEqual(list(a), list(b))

    def test_single_class_and_too_small_batch_are_rejected(self):
        with self.assertRaises(ValueError):
            utils.BalancedBatchSampler([0, 0], batch_size=4)
        with self.assertRaises(ValueError):
            utils.BalancedBatchSampler([0, 1], batch_size=1)

    def test_official_scale_shape_covers_every_index_without_oversampling_negatives(self):
        labels = [0] * 49739 + [1] * 1657
        batches = list(utils.BalancedBatchSampler(labels, batch_size=16, seed=17))
        selected = [index for batch in batches for index in batch]
        self.assertEqual(set(selected), set(range(51396)))
        self.assertEqual(sum(index < 49739 for index in selected), 49739)
        self.assertTrue(all({labels[i] for i in batch} == {0, 1} for batch in batches))


class AucLossTests(unittest.TestCase):
    def test_pairwise_loss_rewards_positive_above_negative_and_has_gradient(self):
        labels = torch.tensor([1.0, 0.0])
        logits = torch.tensor([0.0, 0.0], requires_grad=True)
        loss = utils.pairwise_auc_loss(logits, labels)
        self.assertAlmostEqual(loss.item(), 0.693147, places=5)
        loss.backward()
        self.assertLess(logits.grad[0].item(), 0)
        self.assertGreater(logits.grad[1].item(), 0)
        self.assertLess(utils.pairwise_auc_loss(torch.tensor([2.0, -2.0]), labels).item(),
                        loss.item())

    def test_combined_loss_includes_auc_term(self):
        logits = torch.tensor([0.0, 0.0])
        labels = torch.tensor([1.0, 0.0])
        without = utils.classification_loss(logits, labels, 1.0, auc_weight=0.0)
        with_auc = utils.classification_loss(logits, labels, 1.0, auc_weight=0.1)
        self.assertAlmostEqual((with_auc - without).item(), 0.1 * 0.693147, places=5)


class BootstrapTests(unittest.TestCase):
    def test_previous_candidate_recipe_is_valid_warm_start_for_new_training_recipe(self):
        name = "convnext_tiny.fb_in1k"
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            old_path = Path(folder) / "formal.pth"
            candidate_path = Path(folder) / "candidate.pth"
            torch.save({"recipe": "imagenet1k_convnext_efficientv2_rgb384_fullframe_balanced_pairauc_v2",
                        "states": {name: {"w": torch.tensor([3.0])}},
                        "best_metrics": {name: {"auroc": 0.98}},
                        "pretraining": {name: {"url": "existing-origin"}},
                        "training_complete": False, "selection": None}, candidate_path)
            checkpoint = utils.load_candidate_checkpoint(
                candidate_path, old_path, (name,), "old-formal-recipe")
            self.assertNotEqual(utils.RECIPE, checkpoint["recipe"])
            self.assertEqual(checkpoint["states"][name]["w"].item(), 3.0)
            self.assertEqual(checkpoint["recipe"],
                             "imagenet1k_convnext_efficientv2_rgb384_fullframe_balanced_pairauc_v2")

    def test_fine_tune_models_use_small_head_dropout_without_changing_checkpoint_keys(self):
        for name in candidate_model.MODEL_NAMES:
            model, origin = candidate_model.create_model(name)
            self.assertIsNone(origin)
            self.assertAlmostEqual(model.drop_rate, 0.1)
            old_layout = candidate_model.timm.create_model(
                name, pretrained=False, num_classes=1, drop_rate=0.0)
            self.assertEqual(set(model.state_dict()), set(old_layout.state_dict()))
            if name.startswith("convnext"):
                self.assertAlmostEqual(model.head.drop.p, 0.1)

    def test_candidate_paths_are_separate_and_share_the_official_dataset(self):
        self.assertEqual(utils.DATA, ROOT / "glaucoma_first")
        self.assertEqual(utils.MODEL_PATH, ROOT / "submission2" / "model" / "best_model.pth")
        self.assertEqual(utils.BASELINE_MODEL_PATH, ROOT / "submission" / "model" / "best_model.pth")
        self.assertEqual(utils.OUTPUT_CSV, ROOT / "submission2" / "submission.csv")

    def test_old_checkpoint_is_read_only_bootstrap_for_new_recipe(self):
        name = "convnext_tiny.fb_in1k"
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            old_path = Path(folder) / "old.pth"
            new_path = Path(folder) / "new.pth"
            torch.save({"recipe": "old-recipe", "states": {name: {"w": torch.tensor([2.0])}},
                        "best_metrics": {name: {"auroc": 0.98}},
                        "pretraining": {name: {"url": "old-origin"}},
                        "training_complete": False, "selection": None}, old_path)
            before = old_path.read_bytes()
            checkpoint = utils.load_candidate_checkpoint(
                new_path, old_path, (name,), "old-recipe")
            self.assertEqual(checkpoint["states"][name]["w"].item(), 2.0)
            self.assertEqual(old_path.read_bytes(), before)
            self.assertFalse(new_path.exists())

    def test_candidate_checkpoint_takes_priority_over_baseline(self):
        name = "convnext_tiny.fb_in1k"
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            old_path = Path(folder) / "old.pth"
            new_path = Path(folder) / "new.pth"
            torch.save({"recipe": utils.RECIPE, "states": {name: {"w": torch.tensor([3.0])}},
                        "best_metrics": {name: {"auroc": 0.99}},
                        "pretraining": {name: {"url": "new-origin"}},
                        "training_complete": False, "selection": None}, new_path)
            checkpoint = utils.load_candidate_checkpoint(
                new_path, old_path, (name,), "old-recipe")
            self.assertEqual(checkpoint["states"][name]["w"].item(), 3.0)


class TrainIntegrationTests(unittest.TestCase):
    def test_training_loader_uses_balanced_sampler_but_validation_does_not(self):
        records = [(Path(f"sample{i}.jpg"), int(i >= 23)) for i in range(25)]
        train_loader = candidate_train.loader(records, training=True, batch_size=4)
        valid_loader = candidate_train.loader(records, training=False, batch_size=4)
        self.assertIsInstance(train_loader.batch_sampler, utils.BalancedBatchSampler)
        self.assertNotIsInstance(valid_loader.batch_sampler, utils.BalancedBatchSampler)

    def test_one_synthetic_step_uses_auc_combined_loss(self):
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        data = DataLoader(TensorDataset(torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                                        torch.tensor([1.0, 0.0])), batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        expected = utils.classification_loss(torch.zeros(2), torch.tensor([1.0, 0.0]),
                                             1.0, auc_weight=0.1).item()
        actual = candidate_train.train_epoch(model, data, optimizer, scaler,
                                             torch.tensor(1.0), torch.device("cpu"), 1,
                                             "synthetic", auc_weight=0.1)
        self.assertAlmostEqual(actual, expected, places=5)


class CostLimitTests(unittest.TestCase):
    def test_cost_rejects_mean_of_101_milliseconds(self):
        model = torch.nn.Linear(1, 1)
        def pictures(_directory, count):
            return [Path(f"image{i}.jpg") for i in range(count)]
        def round_duration(paths, _models, _output):
            return [0.101] * len(paths), {}
        with patch.object(candidate_cost, "list_images", side_effect=pictures), \
             patch.object(candidate_cost, "run_round", side_effect=round_duration), \
             patch.object(candidate_cost, "load_inference_models", return_value=[model]):
            with self.assertRaisesRegex(RuntimeError, "100"):
                candidate_cost.main(Path("100"), Path("5"))


if __name__ == "__main__":
    unittest.main()
