---
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - autonomous-driving
  - vision-language-model
  - trajectory-prediction
  - qwen3.5
---

# Neural-Symbolic Drive

Neural-Symbolic Drive is a compact Qwen3.5-based vision-language driving inference repo. The code in this GitHub repository loads the public checkpoint from Hugging Face and runs 3-camera trajectory planning examples end to end.

This implementation builds on ideas and release patterns from Impromptu-VLA, while this repository is packaged as a standalone Neural-Symbolic Drive release.

## Checkpoint

The model checkpoint is hosted separately on Hugging Face:

```text
xiangbog/Neural-Symbolic-Drive
```

The code defaults to that checkpoint, so users do not need to manually download weights before running the example.

## Contents

```text
neural_symbolic_drive/
  infer.py              # Qwen3.5 multimodal batch inference
  metrics.py            # ADE/FDE/AHE/FHE/MR evaluation for <PLANNING> outputs
examples/
  neural_symbolic_drive_examples.json
  example_01/           # front, front-right, front-left images
  example_02/
  example_03/
scripts/
  run_example.sh        # one-command example inference
requirements.txt
```

## Installation

Create a fresh environment:

```bash
conda create -n nsdrive python=3.10 -y
conda activate nsdrive
```

If conda is unavailable:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for your CUDA version, then install the remaining dependencies:

```bash
pip install torch torchvision
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers.git
```

The development Transformers install is recommended because this checkpoint uses the Qwen3.5 architecture.

## Quick Start

From the repository root:

```bash
bash scripts/run_example.sh
```

This downloads and loads `xiangbog/Neural-Symbolic-Drive` through Hugging Face Transformers, then writes:

```text
results/example_predictions.jsonl
```

To use a local checkpoint directory instead:

```bash
bash scripts/run_example.sh /path/to/checkpoint
```

## Direct Inference

```bash
python -m neural_symbolic_drive.infer \
  --model_path xiangbog/Neural-Symbolic-Drive \
  --data_json examples/neural_symbolic_drive_examples.json \
  --save_path results/example_predictions.jsonl \
  --max_new_tokens 2048
```

The input JSON is a ShareGPT-style list. Each item contains three image paths and a user prompt with three `<image>` placeholders. Image paths may be absolute or relative to the JSON file.

## Evaluation

If the input JSON includes assistant labels with `<PLANNING>...</PLANNING>`, compute trajectory metrics with:

```bash
python -m neural_symbolic_drive.metrics \
  --pred_jsonl results/example_predictions.jsonl \
  --gt_json examples/neural_symbolic_drive_examples.json \
  --name Neural-Symbolic-Drive-Qwen3.5 \
  --comparison_json results/comparison.json
```

Metrics:

- `ADE`: average displacement error in meters.
- `FDE`: final displacement error in meters.
- `AHE`: average heading error in degrees.
- `FHE`: final heading error in degrees.
- `MR`: miss rate, where `FDE > 2m`.

## Notes

- The GitHub repository contains code and examples only.
- The Hugging Face repository contains the public Qwen3.5 checkpoint.
- The example set includes real front, front-right, and front-left camera images with trajectory labels.
- Please verify the final release license against the upstream base model license before broad redistribution.
