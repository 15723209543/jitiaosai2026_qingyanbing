"""检查候选模型训练与验证进度使用当前批次而非累计数字。"""

import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "submission_utils_for_batch_test", ROOT / "submission" / "code" / "utils.py")
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


class BatchReportingTests(unittest.TestCase):
    def test_only_first_each_25th_and_last_batch_is_logged(self):
        self.assertEqual(
            [index for index in range(1, 53) if utils.should_log_batch(index, 52)],
            [1, 25, 50, 52],
        )

    def test_report_counts_only_the_given_batch(self):
        first = utils.batch_report(
            torch.tensor([2.0, -2.0, 0.0, -1.0]),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
        )
        second = utils.batch_report(
            torch.tensor([-2.0, -1.0]), torch.tensor([0.0, 0.0]))
        self.assertEqual(first, ", 真实阳性:2, 报阳:2, 报对阳性:1, 误报:1, 漏报:1")
        self.assertEqual(second, ", 真实阳性:0, 报阳:0, 报对阳性:0, 误报:0, 漏报:0")

    @unittest.skipUnless(torch.cuda.is_available(), "本机没有 CUDA；GPU 训练机运行此用例")
    def test_validation_targets_on_cpu_and_logits_on_gpu(self):
        self.assertEqual(
            utils.batch_report(torch.tensor([1.0, -1.0], device="cuda"),
                               torch.tensor([1.0, 0.0])),
            ", 真实阳性:1, 报阳:1, 报对阳性:1, 误报:0, 漏报:0",
        )


if __name__ == "__main__":
    unittest.main()
