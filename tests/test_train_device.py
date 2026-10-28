"""保留上游v1训练实现的设备及存档格式回归；现用入口见test_training_launcher。"""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset


CODE = Path(__file__).resolve().parents[1] / "training" / "20261028" / "source" / "submission" / "code"
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
historical_utils=load('historical_v1_utils',CODE/'utils.py')
historical_model=load('historical_v1_model',CODE/'model.py')
with patch.dict(sys.modules,{'utils':historical_utils,'model':historical_model}):
    train=load('historical_v1_train',CODE/'train.py')


class TrainDeviceTests(unittest.TestCase):
    def test_new_checkpoint_keeps_best_auc_but_no_epoch_number(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best_model.pth"
            old_path = train.MODEL_PATH
            train.MODEL_PATH = path
            try:
                train.save_checkpoint({"example": {"weight": torch.tensor([1.0])}},
                                      {"example": {"epoch": 7, "auroc": 0.91}},
                                      {"example": {"url": "saved-origin"}})
            finally:
                train.MODEL_PATH = old_path
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["best_metrics"]["example"]["auroc"], 0.91)
            self.assertNotIn("epoch", checkpoint["best_metrics"]["example"])

    @unittest.skipIf(torch.cuda.is_available(), "该回归用例针对无 CUDA 的解释器")
    def test_cpu_device_is_reported_and_can_train_one_batch(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            device = train.select_device()
        self.assertEqual(device.type, "cpu")
        self.assertIn("CPU", output.getvalue())

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        data = DataLoader(TensorDataset(torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                                        torch.tensor([1.0, 0.0])), batch_size=2)
        before = model.weight.detach().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            loss = train.train_epoch(model, data, optimizer, scaler,
                                     torch.tensor(1.0), device, 1, "cpu-test")
        self.assertGreaterEqual(loss, 0)
        self.assertFalse(torch.equal(before, model.weight.detach()))


if __name__ == "__main__":
    unittest.main()
