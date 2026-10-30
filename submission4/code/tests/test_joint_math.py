"""Regression checks for the shared dual-backbone fusion math."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_train import fused_logits, objective  # noqa: E402


class Saturated(torch.nn.Module):
    def forward(self, images):
        return torch.full((len(images), 1), 20, dtype=torch.float16)


class JointMathTests(unittest.TestCase):
    def test_training_fusion_matches_prediction_probability(self):
        models = [torch.nn.Linear(3, 1), torch.nn.Linear(3, 1)]
        images = torch.randn(4, 3)
        expected = (torch.sigmoid(models[0](images).flatten()) +
                    torch.sigmoid(models[1](images).flatten())) / 2
        actual = torch.sigmoid(fused_logits(models, images))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_saturated_fp16_probability_has_finite_logit(self):
        output = fused_logits([Saturated(), Saturated()], torch.zeros(2, 1))
        self.assertTrue(torch.isfinite(output).all())

    def test_both_heads_receive_gradient(self):
        models = [torch.nn.Linear(3, 1), torch.nn.Linear(3, 1)]
        labels = torch.tensor([1., 0., 1., 0.])
        for method in ("pairauc", "marginauc"):
            for model in models:
                model.zero_grad(set_to_none=True)
            loss = objective(fused_logits(models, torch.randn(4, 3)), labels, method)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for model in models:
                self.assertGreater(float(model.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
