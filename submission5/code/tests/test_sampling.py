"""Small, dependency-free checks for the submission5 data split."""

import json
import importlib.util
from pathlib import Path
import unittest


module_path = Path(__file__).resolve().parents[1] / "sampling.py"
spec = importlib.util.spec_from_file_location("submission5_sampling", module_path)
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)
split_records = sampling.split_records


def records():
    return [(Path(f"/synthetic/train/image_{index:03d}.jpg"), int(index < 20))
            for index in range(100)]


class SamplingTests(unittest.TestCase):
    def test_holdout_uses_exact_percentage_and_complement(self):
        pool = records()
        train, valid, meta = split_records(pool, percent=80, mode="holdout", seed=7)
        self.assertEqual((len(train), len(valid)), (80, 20))
        self.assertEqual((sum(label for _, label in train),
                          sum(label for _, label in valid)), (16, 4))
        self.assertEqual(set(train) & set(valid), set())
        self.assertEqual(set(train) | set(valid), set(pool))
        self.assertEqual(meta["overlap_count"], 0)
        self.assertEqual((meta["train_fraction"], meta["valid_fraction"]), (0.8, 0.2))
        self.assertEqual(train, [item for item in pool if item in set(train)])
        self.assertEqual(valid, [item for item in pool if item in set(valid)])

    def test_independent_draws_are_reproducible_and_may_overlap(self):
        pool = records()
        first = split_records(pool, percent=80, seed=2026)
        second = split_records(pool, percent=80, mode="independent", seed=2026)
        train, valid, meta = first
        self.assertEqual(first, second)
        self.assertEqual(meta["mode"], "independent")
        self.assertEqual((len(train), len(valid)), (80, 80))
        self.assertEqual((sum(label for _, label in train),
                          sum(label for _, label in valid)), (16, 16))
        self.assertEqual(meta["overlap_count"], len(set(train) & set(valid)))
        self.assertGreaterEqual(meta["overlap_count"], 60)
        self.assertEqual((meta["train_fraction"], meta["valid_fraction"]), (0.8, 0.8))
        other = split_records(pool, percent=80, mode="independent", seed=2027)
        self.assertNotEqual((train, valid), other[:2])

    def test_fingerprints_identify_pool_and_split_without_paths(self):
        pool = records()
        _, _, meta = split_records(pool, percent=80, mode="holdout", seed=3)
        relocated = [(Path("D:/another-computer/train") / path.name, label)
                     for path, label in pool]
        _, _, relocated_meta = split_records(relocated, percent=80,
                                             mode="holdout", seed=3)
        self.assertEqual(meta, relocated_meta)
        changed = pool.copy()
        changed[99] = (Path("/synthetic/train/replaced.jpg"), 0)
        _, _, changed_meta = split_records(changed, percent=80, mode="holdout", seed=3)
        self.assertNotEqual(meta["pool_sha256"], changed_meta["pool_sha256"])
        self.assertNotEqual(meta["train_sha256"], meta["valid_sha256"])
        self.assertNotIn("/synthetic", json.dumps(meta))

    def test_invalid_splits_fail_before_training(self):
        pool = records()
        for percent in (0, 100, -1, float("nan"), True):
            with self.subTest(percent=percent), self.assertRaises(ValueError):
                split_records(pool, percent=percent)
        with self.assertRaises(ValueError):
            split_records(pool, mode="unknown")
        with self.assertRaises(ValueError):
            split_records(pool + [(pool[0][0], 0)])
        with self.assertRaises(ValueError):
            split_records(pool[:-1] + [(pool[-1][0], 2)])
        with self.assertRaises(ValueError):
            split_records([(Path("a.jpg"), 1), (Path("b.jpg"), 0),
                           (Path("c.jpg"), 0)], percent=80)


if __name__ == "__main__":
    unittest.main()
