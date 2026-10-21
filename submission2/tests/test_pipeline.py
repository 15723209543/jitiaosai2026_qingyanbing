"""新方案的最小行为检查；不读取比赛数据，也不启动训练。"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))


class PipelineTests(unittest.TestCase):
    def test_disc_detection_and_two_views_preserve_detail(self):
        self.assertTrue((CODE_DIR / "utils.py").exists(), "需要独立的图像处理工具")
        from utils import create_views, find_disc_center
        image = Image.new("RGB", (800, 600), (80, 20, 20))
        ImageDraw.Draw(image).ellipse((160, 240, 250, 330), fill=(255, 255, 180))
        x, y = find_disc_center(image)
        self.assertLess(abs(x - 205), 80)
        self.assertLess(abs(y - 285), 80)
        full, disc = create_views(image)
        self.assertEqual(full.size, (320, 320))
        self.assertEqual(disc.size, (192, 192))
        self.assertEqual(full.getpixel((0, 0)), (0, 0, 0))

    def test_model_outputs_one_finite_logit_per_image(self):
        self.assertTrue((CODE_DIR / "model.py").exists(), "需要双视野模型")
        from model import FundusNet
        model = FundusNet().eval()
        with torch.no_grad():
            logits = model(torch.zeros(2, 3, 320, 320),
                           torch.zeros(2, 3, 192, 192))
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(torch.isfinite(logits).all())

    def test_submission_schema_and_threshold(self):
        self.assertTrue((CODE_DIR / "utils.py").exists(), "需要提交格式工具")
        from utils import write_submission
        with tempfile.TemporaryDirectory(dir=CODE_DIR.parent) as folder:
            path = Path(folder) / "submission.csv"
            write_submission(["a.jpg", "b.jpg"], np.array([0.2, 0.8]), path)
            data = pd.read_csv(path)
            self.assertEqual(list(data.columns), ["Image_name", "Probability", "Pred_Label"])
            self.assertEqual(data["Pred_Label"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
