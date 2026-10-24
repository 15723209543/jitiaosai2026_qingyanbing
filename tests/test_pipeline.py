import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission' / 'code'))
sys.path.insert(0, str(ROOT / 'submission'))
from utils import letterbox, create_transform, calculate_metrics, write_submission, assert_inference_checkpoint_ready
from cost import compare_submission

class PipelineTests(unittest.TestCase):
    def test_letterbox_preserves_full_field(self):
        image = Image.new('RGB', (512, 256), (255, 0, 0))
        result = np.asarray(letterbox(image))
        self.assertEqual(result.shape, (256, 256, 3))
        self.assertTrue((result[:64] == 0).all())
        self.assertTrue((result[64:192, :, 0] == 255).all())
        self.assertTrue((result[192:] == 0).all())

    def test_upstream_normalization(self):
        tensor = create_transform(False)(Image.new('RGB', (256, 256), (255, 0, 128)))
        self.assertAlmostEqual(float(tensor[0, 0, 0]), 1.)
        self.assertAlmostEqual(float(tensor[1, 0, 0]), -1.)
        self.assertAlmostEqual(float(tensor[2, 0, 0]), 128/255*2-1, places=6)

    def test_metric_uses_continuous_probability(self):
        metrics = calculate_metrics([0, 0, 1, 1], [.1, .4, .3, .8])
        self.assertEqual(metrics['auroc'], .75)
        self.assertEqual((metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']), (2, 0, 1, 1))

    def test_reject_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.pth'
            torch.save({'models': {'resnet18': {'validation': {'auroc': .9}}}}, path)
            with self.assertRaises(RuntimeError):
                assert_inference_checkpoint_ready(path, ['resnet18'])

    def test_submission_rounding_order_and_casefold(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'predictions.csv'
            write_submission(['b.jpg', 'a.jpg'], [.4999996, .2], path)
            data = pd.read_csv(path)
            self.assertEqual(list(data['Image_name']), ['b.jpg', 'a.jpg'])
            self.assertEqual(list(data['Pred_Label']), [1, 0])
            self.assertEqual(compare_submission(path, {'B.JPG': .5, 'A.JPG': .2}), 0)

if __name__ == '__main__':
    unittest.main()
