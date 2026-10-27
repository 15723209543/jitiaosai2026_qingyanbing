# submission2 Balanced AUC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Copy the current submission into an independent trainable experiment with positive-containing batches and AUC-oriented loss, without training.

**Architecture:** Reuse the four Python entry files and cost implementation. New logic stays in `submission2/code/utils.py` and `train.py`; baseline weights are read-only warm-start and candidate checkpoints use a new recipe.

**Tech Stack:** Python, PyTorch, timm, unittest.

**Spec:** `docs/superpowers/specs/2026-10-27-balanced-auc-design.md`

## Global Constraints

- No full training, prediction, PTH download, or change to `submission/` outputs.
- Official validation is AUROC-only selection and never training data.
- Keep exactly `train.py`, `predict.py`, `model.py`, `utils.py`, `requirements.txt` under candidate `code/`.

## Review Focus

- A batch sampler must cover every original index and include both labels in each batch, including tiny imbalanced examples.
- Pairwise AUC loss must have finite gradients under AMP and increase when positive logit falls below negative.
- A Git LFS pointer may not be read as a real bootstrap checkpoint.
- A candidate checkpoint must not be mistaken for the old recipe, and only `predict.py` writes candidate CSV.
- Running tests must not leave cache or temp files in the submission folders.

---

### Task 1: Independent candidate and sampling

**Files:** copy `submission/code/*` and `submission/cost.py` to `submission2/`; modify `submission2/code/utils.py`; test `tests/test_balanced_auc.py`.

**Interfaces:** `BalancedBatchSampler(labels, batch_size, seed)` yields lists of indices; `set_epoch(epoch)` changes deterministic order.

- [ ] Write tests for full coverage, positive/negative per batch, reproducibility, and one-class rejection; verify RED.
- [ ] Implement sampler and new candidate RECIPE; verify GREEN.

### Task 2: AUC loss and warm-start

**Files:** modify `submission2/code/train.py`, `submission2/code/utils.py`; test `tests/test_balanced_auc.py`.

**Interfaces:** `pairwise_auc_loss(logits, targets)`; `classification_loss(logits, targets, pos_weight, auc_weight)`; candidate source fallback to old-model path.

- [ ] Write tests for loss ordering/gradient and bootstrap recipe/path behavior; verify RED.
- [ ] Implement the loss, use the balanced sampler for train only, and load old best states read-only when candidate PTH is absent; verify GREEN.

### Task 3: Delivery checks

**Files:** update `tests/test_submission_layout.py`, `tests/test_submission2_contract.py`, `AI_HANDOFF.md`, `docs/使用说明.md`, `docs/训练结果.md`.

- [ ] Update stale layout/CSV assertions; verify the relevant tests.
- [ ] Run full synthetic/unit suite and parse/import/smoke checks, with no full training.
- [ ] Confirm no large model or cache is staged, then commit and push.
