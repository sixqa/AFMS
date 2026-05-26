# AFMS — Protocol-Driven Multi-Species Aquaculture Phenotyping Framework (Trainer)

> 中文版说明请见 [README.zh.md](./README.zh.md)  
> For Chinese documentation, see [README.zh.md](./README.zh.md).

This repository contains the model training and evaluation module (Trainer) for AFMS, a protocol-driven phenotyping framework that enables multi-species aquaculture animal measurement without core runtime code modification. Species-specific measurement knowledge is encoded in compact YAML protocol files (~20 lines), from which training configurations, inference pipelines, and phenotype calculations are automatically generated.

## Repository Structure

```
trainer/
├── configs/                   # Species protocols + auto-generated training configs
│   ├── carp.yaml              # Common carp protocol
│   ├── crab.yaml              # Chinese mitten crab protocol
│   ├── crayfish.yaml          # Red swamp crayfish protocol
│   ├── ms.yaml                # Largemouth bass protocol
│   ├── wuli.yaml              # Northern snakehead protocol
│   ├── label.yaml             # Label inspection protocol
│   ├── det/                   # Auto-generated detection configs
│   └── kp/                    # Auto-generated keypoint configs
├── hooks/
│   └── hooks.py               # Custom training hooks (EarlyStopping, etc.)
├── tools/
│   ├── generate.py            # YAML protocol → training config generator
│   ├── train_det.py           # Detection model training entry
│   ├── train_kp.py            # Keypoint model training entry
│   ├── export_onnx.py         # ONNX export
│   ├── evaluate_det.py        # Detection evaluation
│   ├── evaluate_kp.py         # Keypoint evaluation
│   ├── batch_evaluate.py      # Batch evaluation
│   └── validate_datasets.py   # Dataset validation
├── evals/                     # Evaluation results (JSON + figures)
└── requirements.txt           # Dependencies
```

## Environment Setup

```bash
conda create -n mmpose python=3.10
conda activate mmpose
pip install torch torchvision
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
mim install "mmpose>=1.0.0"
pip install mmdeploy pyyaml
```

## Adding a New Species

### 1. Write a YAML Protocol (~20 lines)

```yaml
species:
  name: MYFISH
dataset:
  root: "datasets/myfish/"
keypoints:
  - P1
  - P2
  - P3
skeleton:
  - [P1, P2]
  - [P2, P3]
training:
  det_epochs: 500
  kp_epochs: 1000
```

### 2. Generate Training Configurations

```bash
cd trainer
python tools/generate.py configs/myfish.yaml
```

→ Auto-generates `configs/det/myfish.py` + `configs/kp/myfish.py`

### 3. Train Models

```bash
python tools/train_det.py --species myfish
python tools/train_kp.py --species myfish
```

### 4. Export to ONNX

```bash
python tools/export_onnx.py --species myfish
```

## Dataset Format

COCO annotation format, with the following directory structure:

```
trainer/datasets/<species>/
├── train/              # Training images
├── val/                # Validation images
├── train_coco.json     # Training annotations (COCO)
└── val_coco.json       # Validation annotations (COCO)
```

## Supported Species

| Species | Scientific Name | Keypoints | Protocol File |
|---------|----------------|-----------|---------------|
| Common carp | *Cyprinus carpio* | 13 | `configs/carp.yaml` |
| Chinese mitten crab | *Eriocheir sinensis* | 12 | `configs/crab.yaml` |
| Red swamp crayfish | *Procambarus clarkii* | 3 | `configs/crayfish.yaml` |
| Largemouth bass | *Micropterus salmoides* | 12 | `configs/ms.yaml` |
| Northern snakehead | *Channa argus* | 7 | `configs/wuli.yaml` |

## Benchmark

The `benchmark/` directory contains batch-processing stress-test data and visualization scripts.

## Contact

For datasets and model weights, please contact: xsji@sdau.edu.cn / yzhao@sdau.edu.cn
