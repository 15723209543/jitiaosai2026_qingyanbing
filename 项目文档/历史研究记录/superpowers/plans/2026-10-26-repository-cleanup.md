# Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Archive approved historical experiments locally, remove them from current worktree/GitHub, and preserve the active submission and documentation.

**Architecture:** The original repository checkout must be cleaned in place because the user requested matching local/remote trees. Create a ZIP in a git-ignored local archive directory; verify it before deleting exact approved paths. Update human docs to identify recovery and keep active code/data intact.

**Tech Stack:** PowerShell, Git, ZIP, Python unittest.

**Spec:** `docs/superpowers/specs/2026-10-26-repository-cleanup-design.md`

## Global Constraints

- Never delete or recurse into `glaucoma_first` or other junction targets.
- Keep `submission/model/best_model.pth`, `submission/submission.csv`, `submission2`, `docs`, `项目文档`, and both existing ZIPs unchanged.
- Backup is local-only, ignored by Git. Do not push it or call it a full formal-model backup because the formal PTH is a local LFS pointer.
- Run no training or prediction.

## Review Focus

- ZIP must contain all approved source paths and readable full local weights where present.
- No symlinks/junctions inside targets; all delete paths must resolve within workspace.
- Runtime imports must not point into removed directories.
- Current CSV and formal PTH pointer bytes must remain unchanged.
- GitHub main must equal local commit after push.

---

### Task 1: Verify and archive the approved historical paths

**Files:** Create only a local ignored ZIP; modify `.gitignore` to ignore its folder.

**Interfaces:** Consumes the exact source list in the spec; produces a ZIP plus entry/size verification evidence.

- [ ] Record pre-cleanup SHA-256 of the formal CSV and PTH pointer; inspect all target paths and reparse points.
- [ ] Demonstrate the post-cleanup absence assertion fails before deletion.
- [ ] Add the local archive folder to `.gitignore`; create ZIP using explicit relative source paths.
- [ ] Compare archive entries and uncompressed sizes against source files; inspect key `.pth` entries and save ZIP SHA-256.

### Task 2: Remove only the archived paths and verify

**Files:** Remove the paths named in the spec; create `submission/pretrain_model/README.md`; update `submission/README.md`, `README.md`, `AI_HANDOFF.md`, and a recovery note.

**Interfaces:** Consumes the verified Task 1 ZIP; produces a clean current tree and recovery instructions.

- [ ] Resolve every exact target under workspace before any recursive removal.
- [ ] Remove only the approved paths; recreate the minimal pretrain directory README.
- [ ] Verify absence of targets, preservation and SHA of protected files, and relevant tests.
- [ ] Review `git diff --check` and staged path list; commit as `v1.13.3 chore(repo):归档旧实验并精简提交目录` and push main.
- [ ] Verify remote main commit equals local HEAD.
