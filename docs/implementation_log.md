# Implementation Log

This file documents code changes made in this repository. Going forward, each code modification should be accompanied by an update to this Markdown log.

## 2026-04-17 - Initial DAVE-2 PyTorch Pipeline

### Summary

Created a modular end-to-end behavioral cloning project for Udacity simulator data using PyTorch.

### Files Added

- `requirements.txt`
- `src/dave2/__init__.py`
- `src/dave2/config.py`
- `src/dave2/data.py`
- `src/dave2/dataset.py`
- `src/dave2/model.py`
- `src/dave2/training.py`
- `train.py`

### What Changed

- Added Phase 1 data engineering utilities to read `driving_log.csv`, resolve image paths, expand center/left/right camera views, apply steering correction, and create train/validation splits.
- Added Phase 2 preprocessing utilities and a custom PyTorch `Dataset` for cropping, resizing to `66x200`, RGB-to-YUV conversion, normalization to `[-1, 1]`, and tensor formatting in `C,H,W`.
- Added an initial DAVE-2 model module.
- Added a complete training loop with device selection, MSE loss, Adam optimizer, epoch-level train/validation tracking, and best-checkpoint saving.
- Added a root training entrypoint that wires together data ingestion, preprocessing, model creation, dataloaders, and training.

## 2026-04-17 - ViT-Based DAVE-2 Refactor

### Summary

Replaced the CNN backbone with a Vision Transformer-based DAVE-2 family and introduced scalable model presets.

### Files Modified

- `src/dave2/model.py`
- `src/dave2/__init__.py`
- `train.py`

### What Changed

- Replaced the convolutional DAVE-2 backbone with a patch-embedding plus transformer architecture for steering-angle regression.
- Added configurable ViT presets: `mini`, `medium`, and `large`.
- Introduced explicit model configuration via `DAVE2ViTConfig`.
- Added trainable CLS token and positional embeddings.
- Implemented custom transformer blocks with multi-head self-attention, MLP sublayers, dropout, and layer normalization.
- Added helper utilities to build a selected model variant, count parameters, and describe preset settings.
- Updated the training CLI to accept `--model-size`.
- Added `--list-variants` to print the available ViT presets.
- Changed the default checkpoint naming behavior so each variant writes to its own file, such as `artifacts/best_dave2_vit_medium.pth`.

## 2026-04-19 - Hybrid SAM-Style Perception + ViT-DAVE2 Upgrade

### Summary

Restructured the project into a hybrid perception-policy architecture with explicit road understanding, lane geometry, object tracking, ablation-ready fusion, and a multi-action driving head.

### Files Added

- `src/dave2/perception.py`
- `src/dave2/fusion.py`
- `src/dave2/hybrid_model.py`
- `docs/hybrid_sam_vit_architecture.md`

### Files Modified

- `src/dave2/config.py`
- `src/dave2/model.py`
- `src/dave2/__init__.py`
- `docs/implementation_log.md`

### What Changed

- Refactored the original ViT DAVE-2 model so the visual encoder can be reused independently of the legacy steering-only regression head.
- Added hybrid configuration objects for perception, fusion, policy, and top-level hybrid model assembly.
- Added a SAM-style perception wrapper with a multi-scale encoder adapter, feature pyramid aggregation, scene tokenization, semantic segmentation, drivable road segmentation, lane boundary decoding, object query decoding, tracking, and control-feature extraction.
- Added an ablation-ready fusion module that supports `early_fusion`, `late_fusion`, `cross_attention`, `multi_branch`, and `token_level` strategies.
- Added a hybrid driving model that fuses ViT-DAVE2 visual tokens with perception tokens and structured control features, then predicts steering, throttle, and brake with optional uncertainty.
- Added a research-grade architecture note covering system design, fusion tradeoffs, data strategy, failure analysis, and publishable upgrade directions.

## Documentation Rule

For every future code change:

- update this file in the same turn
- include the date
- list the files added or modified
- summarize the architectural or behavioral impact
