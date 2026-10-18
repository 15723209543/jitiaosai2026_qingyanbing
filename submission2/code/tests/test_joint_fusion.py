"""联合融合模型的最小回归测试；只用随机小张量，不读取训练图片。"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import ImageModel
from train import train_step
from utils import initialize_backbones, load_model, save_model


class JointFusionTest(unittest.TestCase):
    def setUp(self):
        """【来源】本项目自写；固定随机种子和线程数，保证测试可重复。"""
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_one_forward_pass_uses_both_backbones(self):
        """【来源】本项目自写；删掉任一分支或输出两个独立预测时应失败。"""
        model = ImageModel().eval()
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(hasattr(model, "resnet"))
        self.assertTrue(hasattr(model, "densenet"))
        self.assertTrue(hasattr(model, "fusion"))

    def test_training_step_updates_both_backbones(self):
        """【来源】本项目自写；冻结任一骨干或改回概率平均时应失败。"""
        model = ImageModel()
        x = torch.rand(2, 3, 64, 64)
        y = torch.tensor([0, 1])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before_resnet = model.resnet.conv1.weight.detach().clone()
        before_dense = model.densenet.features.conv0.weight.detach().clone()
        loss, correct = train_step(model, x, y, optimizer)
        self.assertGreater(loss, 0)
        self.assertGreaterEqual(correct, 0)
        self.assertLessEqual(correct, 2)
        self.assertFalse(torch.equal(before_resnet, model.resnet.conv1.weight))
        self.assertFalse(torch.equal(before_dense, model.densenet.features.conv0.weight))

    def test_baseline_weights_initialize_both_feature_extractors(self):
        """【来源】本项目自写；误读旧权重或只初始化一个分支时应失败。"""
        source, model = ImageModel(), ImageModel()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as folder:
            paths = []
            for name, backbone in (("resnet18", source.resnet),
                                   ("densenet121", source.densenet)):
                path = Path(folder) / (name + ".pth")
                torch.save({"model_config": {"model_name": name, "image_size": 64},
                            "state_dict": {"network." + key: value
                                           for key, value in backbone.state_dict().items()}}, path)
                paths.append(path)
            initialize_backbones(model, *paths)
        self.assertTrue(torch.equal(model.resnet.conv1.weight, source.resnet.conv1.weight))
        self.assertTrue(torch.equal(model.densenet.features.conv0.weight,
                                    source.densenet.features.conv0.weight))

    def test_one_checkpoint_roundtrips_and_stays_below_limit(self):
        """【来源】本项目自写；拆权重、配置不匹配或超过500MB时应失败。"""
        model = ImageModel().eval()
        x = torch.rand(1, 3, 64, 64)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as folder:
            path = Path(folder) / "best_model.pth"
            save_model(model, path, {"auroc": 0.8}, 1)
            self.assertEqual([item.name for item in Path(folder).glob("*.pth")],
                             ["best_model.pth"])
            self.assertLess(path.stat().st_size, 500_000_000)
            restored = load_model(path).eval()
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(x), restored(x), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
