"""回归检查：无 CUDA 时明确告知设备，并允许 CPU 执行训练步骤。"""

import contextlib
import io
from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset


CODE = Path(__file__).resolve().parents[1] / "submission" / "code"
sys.path.insert(0, str(CODE))
import train  # noqa: E402


class TrainDeviceTests(unittest.TestCase):
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
