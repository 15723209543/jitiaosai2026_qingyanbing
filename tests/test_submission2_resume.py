"""阶段性单文件权重应能安全恢复，不误把指针或错配配方当作模型。"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "submission_utils_for_resume_test", ROOT / "submission" / "code" / "utils.py")
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)
NAMES = ("convnext_tiny.fb_in1k", "efficientnetv2_rw_s.ra2_in1k")


class ResumeCheckpointTests(unittest.TestCase):
    def test_partial_checkpoint_retains_best_epoch_and_untrained_branch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best_model.pth"
            original = {"recipe": utils.RECIPE,
                        "states": {NAMES[0]: {"weight": torch.tensor([3.0])}},
                        "best_metrics": {NAMES[0]: {"epoch": 7, "auroc": 0.91}},
                        "pretraining": {NAMES[0]: {"url": "https://example.invalid/model"}},
                        "selection": None, "training_complete": False}
            torch.save(original, path)
            loaded = utils.load_resume_checkpoint(path, NAMES)
            self.assertEqual(utils.resume_start_epoch(loaded, NAMES[0]), 8)
            self.assertEqual(utils.resume_start_epoch(loaded, NAMES[1]), 1)
            self.assertEqual(loaded["best_metrics"][NAMES[0]]["auroc"], 0.91)
            self.assertEqual(torch.load(path, weights_only=True)["states"][NAMES[0]]["weight"].item(), 3.0)

    def test_rejects_changed_recipe_without_overwriting_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best_model.pth"
            torch.save({"recipe": "incompatible", "states": {}, "best_metrics": {},
                        "pretraining": {}}, path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "配方"):
                utils.load_resume_checkpoint(path, NAMES)
            self.assertEqual(path.read_bytes(), before)

    def test_rejects_git_lfs_pointer_as_not_a_real_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best_model.pth"
            path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n")
            with self.assertRaisesRegex(RuntimeError, "Git LFS"):
                utils.load_resume_checkpoint(path, NAMES)

    def test_legacy_torch_version_object_loads_in_restricted_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best_model.pth"
            old = {"recipe": utils.RECIPE,
                   "states": {NAMES[0]: {"weight": torch.tensor([3.0])}},
                   "best_metrics": {NAMES[0]: {"epoch": 7, "auroc": 0.91}},
                   "pretraining": {NAMES[0]: {"url": "saved-origin"}},
                   "selection": None, "training_complete": False,
                   "torch_version": torch.__version__}
            torch.save(old, path)
            before = path.read_bytes()
            loaded = utils.load_resume_checkpoint(path, NAMES)
            self.assertEqual(loaded["best_metrics"][NAMES[0]]["epoch"], 7)
            self.assertEqual(path.read_bytes(), before)

    def test_full_validation_auc_sets_safe_starting_threshold(self):
        checkpoint = {"best_metrics": {NAMES[0]: {"auroc": 0.91}}}
        self.assertEqual(utils.resume_baseline_auc(checkpoint, NAMES[0], 0.9102), 0.9102)
        self.assertEqual(utils.resume_baseline_auc(checkpoint, NAMES[0], 0.9099), 0.91)
        with self.assertRaisesRegex(ValueError, "复核 AUROC"):
            utils.resume_baseline_auc(checkpoint, NAMES[0], 0.85)

    def test_incomplete_checkpoint_is_rejected_before_prediction(self):
        checkpoint = {"training_complete": False, "selection": None,
                      "states": {NAMES[0]: {"weight": torch.tensor([1.0])}}}
        with self.assertRaisesRegex(RuntimeError, "模型未完成"):
            utils.inference_model_names(checkpoint, NAMES)

    def test_saved_branch_loads_its_weights_without_redownloading_pretrain(self):
        source = torch.nn.Linear(1, 1)
        with torch.no_grad():
            source.weight.fill_(2.0)
            source.bias.fill_(0.5)
        checkpoint = {"states": {NAMES[0]: source.state_dict()},
                      "pretraining": {NAMES[0]: {"url": "saved-origin"}}}

        def factory(name, pretrain_dir=None):
            self.assertEqual(name, NAMES[0])
            self.assertIsNone(pretrain_dir)
            return torch.nn.Linear(1, 1), None

        model, origin, resumed = utils.restore_candidate_model(
            NAMES[0], checkpoint, factory, Path("unused"))
        self.assertTrue(resumed)
        self.assertEqual(origin["url"], "saved-origin")
        self.assertEqual(model(torch.tensor([[1.0]])).item(), 2.5)


if __name__ == "__main__":
    unittest.main()
