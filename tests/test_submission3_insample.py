"""submission3 的独立实验契约；仅使用合成路径，不运行正式训练。"""

import importlib.util
import io
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "submission3" / "code"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InSampleExperimentTests(unittest.TestCase):
    def test_one_synthetic_round_saves_separate_single_checkpoint(self):
        utils = load_module("submission3_utils", CODE / "utils.py")
        model_module = load_module("submission3_model", CODE / "model.py")
        with patch.dict(sys.modules, {"utils": utils, "model": model_module}):
            train = load_module("submission3_train", CODE / "train.py")

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(3, 1)

            def forward(self, images):
                return self.head(images.mean(dim=(2, 3)))

            def get_classifier(self):
                return self.head

        names = ("toy_a", "toy_b")
        config = {name: {"epochs": 1, "backbone_lr": 0.001,
                         "head_lr": 0.001, "weight_decay": 0.0, "patience": 1}
                  for name in names}
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            records = []
            for index, (color, label) in enumerate((("white", 1), ("black", 0),
                                                     ("gray", 1), ("navy", 0))):
                path = root / f"sample_{index}.jpg"
                Image.new("RGB", (8, 8), color).save(path)
                records.append((path, label))
            checkpoint_path = root / "best_model.pth"
            origin = {"url": "synthetic", "sha256": "synthetic"}
            with patch.multiple(train, MODEL_NAMES=names, CONFIG=config,
                                MODEL_PATH=checkpoint_path, BASELINE_MODEL_PATH=root / "absent.pth",
                                NUM_WORKERS=0), \
                 patch.object(train, "select_device", return_value=torch.device("cpu")), \
                 patch.object(train, "ensure_license", return_value=root / "license.txt"), \
                 patch.object(train, "fit_records", return_value=records), \
                 patch.object(train, "create_model", side_effect=lambda name: (TinyModel(), None)), \
                 patch.object(train, "restore_candidate_model",
                              side_effect=lambda *args: (TinyModel(), origin, False)):
                output = io.StringIO()
                with redirect_stdout(output):
                    train.train()
            saved = utils.load_resume_checkpoint(checkpoint_path, names)
            self.assertEqual(set(saved["states"]), set(names))
            self.assertEqual(set(saved["best_metrics"]), set(names))
            self.assertTrue(all(item["metric_scope"] == "insample"
                                for item in saved["best_metrics"].values()))
            self.assertTrue(saved["training_complete"])
            self.assertIn("数据泄漏", output.getvalue())
            with patch.dict(sys.modules, {"utils": utils, "model": model_module}):
                predict = load_module("submission3_predict", CODE / "predict.py")
            real_loader = predict.DataLoader
            with patch.multiple(predict, MODEL_NAMES=names, MODEL_PATH=checkpoint_path), \
                 patch.object(predict, "create_model", side_effect=lambda name: (TinyModel(), None)), \
                 patch.object(predict, "fit_records", return_value=records), \
                 patch.object(predict, "read_records", return_value=[(records[0][0], None)]), \
                 patch.object(predict, "DataLoader", side_effect=lambda dataset, **kwargs:
                              real_loader(dataset, batch_size=kwargs["batch_size"],
                                          shuffle=False, num_workers=0)):
                with redirect_stdout(io.StringIO()):
                    csv_path = predict.predict(root / "submission.csv")
            with csv_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(list(rows[0]), ["Image_name", "Probability", "Pred_Label"])
            self.assertEqual(rows[0]["Image_name"], records[0][0].name)

    def test_fit_pool_contains_official_train_and_valid_but_not_test(self):
        utils = load_module("submission3_utils", CODE / "utils.py")
        records = {
            "train": [(Path("train_a.jpg"), 1), (Path("train_b.jpg"), 0)],
            "valid": [(Path("valid_a.jpg"), 1)],
        }
        with patch.object(utils, "read_records", side_effect=lambda split: records[split]):
            self.assertEqual(utils.fit_records(), records["train"] + records["valid"])

    def test_formal_checkpoint_uses_measured_insample_auc_as_new_baseline(self):
        utils = load_module("submission3_utils", CODE / "utils.py")
        model = load_module("submission3_model", CODE / "model.py")
        with patch.dict(sys.modules, {"utils": utils, "model": model}):
            train = load_module("submission3_train", CODE / "train.py")
        old = {"recipe": utils.BASE_RECIPE,
               "best_metrics": {model.MODEL_NAMES[0]: {"auroc": 0.98}}}
        self.assertAlmostEqual(train.starting_auc(old, model.MODEL_NAMES[0], 0.997), 0.997)
        new = {"recipe": utils.RECIPE,
               "best_metrics": {model.MODEL_NAMES[0]: {"auroc": 0.99}}}
        self.assertAlmostEqual(train.starting_auc(new, model.MODEL_NAMES[0], 0.991), 0.991)
        mixed = {"recipe": utils.RECIPE,
                 "best_metrics": {model.MODEL_NAMES[0]: {
                     "auroc": 0.98, "metric_scope": "official_valid"}}}
        self.assertAlmostEqual(train.starting_auc(mixed, model.MODEL_NAMES[0], 0.997), 0.997)

    def test_candidate_has_separate_paths_and_new_recipe(self):
        utils = load_module("submission3_utils", CODE / "utils.py")
        self.assertEqual(utils.MODEL_PATH, ROOT / "submission3" / "model" / "best_model.pth")
        self.assertEqual(utils.OUTPUT_CSV, ROOT / "submission3" / "submission.csv")
        self.assertNotEqual(utils.RECIPE, utils.BASE_RECIPE)
        self.assertFalse((ROOT / "submission3" / "submission.csv").exists())


if __name__ == "__main__":
    unittest.main()
