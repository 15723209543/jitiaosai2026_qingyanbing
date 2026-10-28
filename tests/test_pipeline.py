import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('submission_utils_for_pipeline_test', ROOT / 'submission' / 'code' / 'utils.py')
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)

class PipelineTests(unittest.TestCase):
    def test_letterbox_preserves_full_field(self):
        image = Image.new('RGB', (512, 256), (255, 0, 0))
        result = np.asarray(utils.letterbox(image))
        self.assertEqual(result.shape, (384, 384, 3))
        self.assertTrue((result[:96] == 0).all())
        self.assertTrue((result[96:288, :, 0] == 255).all())
        self.assertTrue((result[288:] == 0).all())

    def test_upstream_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.png'
            Image.new('RGB', (384, 384), (255, 0, 0)).save(path)
            tensor = utils.FundusDataset([(path, None)], training=False)[0]
        self.assertAlmostEqual(float(tensor[0, 0, 0]), (1 - utils.MEAN[0]) / utils.STD[0], places=2)

    def test_metric_uses_continuous_probability(self):
        metrics = utils.metrics([0, 0, 1, 1], [.1, .4, .3, .8])
        self.assertEqual(metrics['auroc'], .75)
        self.assertEqual((metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']), (2, 0, 1, 1))

    def test_submission_rounding_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'predictions.csv'
            utils.write_submission([(Path('b.jpg'), None), (Path('a.jpg'), None)], [.4999996, .2], path)
            import csv
            with path.open(newline='', encoding='utf-8') as file:
                data = list(csv.DictReader(file))
            self.assertEqual([row['Image_name'] for row in data], ['b.jpg', 'a.jpg'])
            self.assertEqual([row['Pred_Label'] for row in data], ['1', '0'])

    def test_submission_image_suffix_matches_official_lowercase(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'submission.csv'
            utils.write_submission([(Path('test_00000.JPG'), None),
                                    (Path('Other.PNG'), None)], [0.8, 0.2], output)
            import csv
            with output.open(newline='', encoding='utf-8') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row['Image_name'] for row in rows],
                             ['test_00000.jpg', 'Other.png'])

    def test_submission_output_filename_uses_lowercase_csv_suffix(self):
        with tempfile.TemporaryDirectory() as folder:
            output = utils.write_submission([(Path('test_00000.JPG'), None)],
                                            [0.8], Path(folder) / 'submission.CSV')
            self.assertEqual(output.name, 'submission.csv')
            self.assertTrue(output.is_file())

if __name__ == '__main__':
    unittest.main()
