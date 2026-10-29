import sys
from pathlib import Path

import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict import prediction_model_names


NAMES = ("convnext_tiny.fb_in1k", "efficientnetv2_rw_s.ra2_in1k")


class PausedPredictionTests(unittest.TestCase):
    def test_paused_checkpoint_requires_explicit_recovery(self):
        checkpoint = {"states": {name: {"weight": object()} for name in NAMES},
                      "training_complete": False, "selection": None}
        with self.assertRaises(RuntimeError):
            prediction_model_names(checkpoint, NAMES, recover_paused=False)
        self.assertEqual(prediction_model_names(checkpoint, NAMES, recover_paused=True), NAMES)

    def test_recovery_rejects_missing_branch(self):
        checkpoint = {"states": {NAMES[0]: {"weight": object()}},
                      "training_complete": False, "selection": None}
        with self.assertRaises(RuntimeError):
            prediction_model_names(checkpoint, NAMES, recover_paused=True)


if __name__ == "__main__":
    unittest.main()
