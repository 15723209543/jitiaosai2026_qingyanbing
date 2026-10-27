"""预训练候选的静态交付约束；不下载权重，也不启动训练。"""

import ast
import csv
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "submission2"


class Submission2ContractTests(unittest.TestCase):
    def test_required_code_layout(self):
        code = CANDIDATE / "code"
        self.assertEqual(
            {p.name for p in code.glob("*.py")},
            {"train.py", "predict.py", "model.py", "utils.py"},
        )
        self.assertTrue((CANDIDATE / "cost.py").is_file())

    def test_all_python_files_parse(self):
        for path in [*(CANDIDATE / "code").glob("*.py"), CANDIDATE / "cost.py"]:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_no_test_fitting_or_random_inference(self):
        predict = (CANDIDATE / "code" / "predict.py").read_text(encoding="utf-8")
        self.assertNotIn("optimizer.step", predict)
        self.assertNotIn("RandomHorizontalFlip", predict)
        self.assertIn("write_submission", predict)

    def test_official_csv_is_preserved_and_candidate_does_not_prewrite_one(self):
        official = ROOT / "submission" / "submission.csv"
        with official.open(newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 5855)
        self.assertEqual(list(rows[0]), ["Image_name", "Probability", "Pred_Label"])
        self.assertFalse((CANDIDATE / "submission.csv").exists())


if __name__ == "__main__":
    unittest.main()
