"""回归检查：无 CUDA 时明确告知设备，并允许 CPU 执行训练步骤。"""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset


CODE = Path(__file__).resolve().parents[1] / "submission" / "code"
sys.path.insert(0, str(CODE))
import train  # noqa: E402


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
