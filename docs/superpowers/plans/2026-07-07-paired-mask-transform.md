# Paired Mask Transform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `train_repro_dtheta.py` keep CUB images and external patch masks spatially aligned under resize, crop and horizontal flip.

**Architecture:** Add a small paired image-mask transform module used only by the mask-guided AFGCD entrypoint. The dataset will call transforms with both image and mask, and train collate will preserve one patch mask per augmented view.

**Tech Stack:** Python, PIL, NumPy, lazy PyTorch tensor conversion for training runtime.

## Global Constraints

- Do not modify source files under `D:\llm-wiki\raw`.
- Keep the existing `train_repro_dtheta.py`, `--mask_root` and `--dtheta` entrypoint.
- Test geometry synchronization before modifying production code.

---

### Task 1: Paired image-mask geometry

**Files:**
- Create: `D:\llm-wiki\AFGCD\tests\test_paired_mask_transforms.py`
- Create: `D:\llm-wiki\AFGCD\data\paired_mask_transforms.py`

**Interfaces:**
- Produces: `crop_pair(image, mask, top, left, size) -> (image, mask)`
- Produces: `hflip_pair(image, mask) -> (image, mask)`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run tests and verify import failure**
- [ ] **Step 3: Implement paired geometry helpers**
- [ ] **Step 4: Run tests and verify pass**

### Task 2: Route CUB mask dataset through paired transforms

**Files:**
- Modify: `D:\llm-wiki\AFGCD\data\cub_mask.py`
- Modify: `D:\llm-wiki\AFGCD\train_repro_dtheta.py`

**Interfaces:**
- Produces: `PairedMaskViewGenerator(base_transform, n_views)` returning `([image_view...], [mask_view...])`
- Consumes: `CustomCub2011Mask.__getitem__` can call paired transforms as `transform(img, patch_mask)`.

- [ ] **Step 1: Let dataset call paired transform when available**
- [ ] **Step 2: Use paired train/eval transforms in `train_repro_dtheta.py`**
- [ ] **Step 3: Update collate/train code to concatenate per-view masks**
- [ ] **Step 4: Compile touched files**

