"""正式提交目录与官方数据目录应平级，且只保留一套代码。"""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"


class SubmissionLayoutTests(unittest.TestCase):
    def test_only_candidate_promoted_to_official_submission(self):
        self.assertTrue((ROOT / "glaucoma_first").is_dir())
        self.assertTrue(SUBMISSION.is_dir())
        self.assertFalse((ROOT / "submission2").exists())
        self.assertEqual(
            {path.name for path in SUBMISSION.iterdir()},
            {"code", "model", "pretrain_model", "licenses", "cost.py",
             "模型核验材料与数据申明清单.xlsx", "submission.csv"},
        )
        self.assertEqual(
            {path.name for path in (SUBMISSION / "code").glob("*.py")},
            {"model.py", "train.py", "predict.py", "utils.py"},
        )
        self.assertEqual(
            {path.name for path in (SUBMISSION / "code").iterdir()},
            {"model.py", "train.py", "predict.py", "utils.py", "requirements.txt"},
        )
        self.assertEqual({path.name for path in (SUBMISSION / "model").iterdir()}, {"best_model.pth"})
        self.assertFalse(list(SUBMISSION.rglob("__pycache__")))

    def test_official_scripts_parse_and_share_candidate_loader(self):
        for path in [*(SUBMISSION / "code").glob("*.py"), SUBMISSION / "cost.py"]:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        source = (SUBMISSION / "code" / "predict.py").read_text(encoding="utf-8")
        self.assertIn("load_resume_checkpoint", source)


if __name__ == "__main__":
    unittest.main()
