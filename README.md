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

## Installation

Create a fresh environment:

```bash
conda create -n nsdrive python=3.10 -y
conda activate nsdrive
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

