"""仅用临时小图片检验 submission2 的额外训练数据接口，不运行正式训练。"""

import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "submission2" / "code"


def load_train_module():
    """测试专用：按 submission2 的模块路径加载真实 train.py。"""
    loaded = {}
    for name in ("utils", "model"):
        spec = importlib.util.spec_from_file_location(name, CODE / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[name] = module
    spec = importlib.util.spec_from_file_location("submission2_extra_train", CODE / "train.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, loaded):
        spec.loader.exec_module(module)
    return module


TRAIN = load_train_module()


class ExternalRecordsTests(unittest.TestCase):
    def make_sample(self, root, split, name, label):
        """测试专用：生成一张小图与同名的 YOLO 标签文件。"""
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        path = images / (name + ".jpg")
        Image.new("RGB", (8, 8), "white").save(path)
        (labels / (name + ".txt")).write_text(label, encoding="utf-8")
        return path

    def test_all_three_external_splits_are_training_records_with_inverted_labels(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            positive = self.make_sample(root, "train", "a", "0 0.5 0.5 0.2 0.2\n")
            negative = self.make_sample(root, "valid", "b", "1 0.4 0.5 0.2 0.2\n")
            other = self.make_sample(root, "test", "c", "0 0.3 0.4 0.2 0.2\n")
            self.assertEqual(TRAIN.read_my_records(root),
                             [(positive, 1), (negative, 0), (other, 1)])

    def test_missing_label_and_unknown_class_fail_before_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            path = self.make_sample(root, "train", "a", "2 0.5 0.5 0.2 0.2\n")
            with self.assertRaisesRegex(ValueError, "0 或 1"):
                TRAIN.read_my_records(root)
            (root / "train" / "labels" / "a.txt").unlink()
            with self.assertRaises(FileNotFoundError):
                TRAIN.read_my_records(root)
            self.assertTrue(path.exists())


class ExternalTrainingStageTests(unittest.TestCase):
    def test_each_epoch_trains_official_then_extra_then_validates_once(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(3, 1, bias=False)
                torch.nn.init.zeros_(self.head.weight)

            def forward(self, images):
                return self.head(images.mean(dim=(2, 3)))

            def get_classifier(self):
                return self.head

        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            official_positive = root / "official_positive.jpg"
            official_negative = root / "official_negative.jpg"
            extra_positive = root / "extra_positive.jpg"
            extra_negative = root / "extra_negative.jpg"
            valid_positive = root / "valid_positive.jpg"
            valid_positive2 = root / "valid_positive2.jpg"
            valid_negative = root / "valid_negative.jpg"
            for path, color in ((official_positive, "white"), (official_negative, "black"),
                                (extra_positive, "white"), (extra_negative, "black"),
                                (valid_positive, "white"), (valid_positive2, "white"),
                                (valid_negative, "black")):
                Image.new("RGB", (8, 8), color).save(path)
            model = TinyModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            with patch.object(TRAIN, "NUM_WORKERS", 0):
                official_loader = TRAIN.loader([(official_positive, 1), (official_negative, 0)], True)
                extra_loader = TRAIN.loader([(extra_positive, 1), (extra_negative, 0)], True)
                valid_loader = TRAIN.loader([(valid_positive, 1), (valid_positive2, 1),
                                             (valid_negative, 0)], False)
                output = io.StringIO()
                with redirect_stdout(output):
                    for epoch in (1, 2):
                        result = TRAIN.train_round(model, official_loader, extra_loader,
                                                   valid_loader, optimizer, scaler,
                                                   torch.tensor(1.0), torch.tensor(1.0),
                                                   torch.device("cpu"), epoch, "tiny", 0.1)
            text = output.getvalue()
            first_official = text.index("当前使用 glaucoma_first 官方训练集训练")
            first_extra = text.index("当前使用 glaucoma_first_my 额外训练集训练")
            first_valid = text.index("valid, batch:")
            second_official = text.index("当前使用 glaucoma_first 官方训练集训练", first_official + 1)
            second_extra = text.index("当前使用 glaucoma_first_my 额外训练集训练", first_extra + 1)
            second_valid = text.index("valid, batch:", first_valid + 1)
            self.assertTrue(first_official < first_extra < first_valid <
                            second_official < second_extra < second_valid)
            self.assertEqual(text.count("valid, batch:"), 2)
            self.assertIn("真实阳性:2", text)
            self.assertIn("auroc", result)
            self.assertGreater(model.head.weight.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
