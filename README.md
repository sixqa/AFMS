# AFMS — 训练器 (Trainer)

协议驱动的多物种水产表型测量框架 — 训练模块。

## 目录结构

```
trainer/
├── configs/                   # 物种协议 + 生成的训练配置
│   ├── carp.yaml              # 鲤鱼协议
│   ├── crab.yaml              # 螃蟹协议
│   ├── crayfish.yaml          # 小龙虾协议
│   ├── ms.yaml                # 大口黑鲈协议
│   ├── wuli.yaml              # 乌鳢协议
│   ├── label.yaml             # 标签协议
│   ├── det/                   # 自动生成的检测配置
│   └── kp/                    # 自动生成的关键点配置
├── hooks/
│   └── hooks.py               # 自定义训练钩子 (EarlyStopping 等)
├── tools/
│   ├── generate.py            # YAML → 训练配置 自动生成
│   ├── train_det.py           # 检测模型训练入口
│   ├── train_kp.py            # 关键点模型训练入口
│   ├── export_onnx.py         # ONNX 导出
│   ├── evaluate_det.py        # 检测评估
│   ├── evaluate_kp.py         # 关键点评估
│   ├── batch_evaluate.py      # 批量评估
│   └── validate_datasets.py   # 数据集校验
├── evals/                     # 评估结果 (JSON + 图)
└── requirements.txt           # 依赖
```

## 环境

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

## 新增物种流程

### 1. 写 YAML 协议 (~20 行)

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

### 2. 生成训练配置

```bash
cd trainer
python tools/generate.py configs/myfish.yaml
```

→ 自动生成 `configs/det/myfish.py` + `configs/kp/myfish.py`

### 3. 训练

```bash
python tools/train_det.py --species myfish
python tools/train_kp.py --species myfish
```

### 4. 导出 ONNX

```bash
python tools/export_onnx.py --species myfish
```

## 数据集格式

COCO 格式，目录结构：

```
trainer/datasets/<species>/
├── train/              # 训练图片
├── val/                # 验证图片
├── train_coco.json     # 训练标注 (COCO)
└── val_coco.json       # 验证标注 (COCO)
```

## 已有物种

| 物种 | 学名 | 关键点 | 协议文件 |
|------|------|--------|---------|
| 鲤鱼 | *Cyprinus carpio* | 13 | `configs/carp.yaml` |
| 中华绒螯蟹 | *Eriocheir sinensis* | 12 | `configs/crab.yaml` |
| 克氏原螯虾 | *Procambarus clarkii* | 3 | `configs/crayfish.yaml` |
| 大口黑鲈 | *Micropterus salmoides* | 12 | `configs/ms.yaml` |
| 乌鳢 | *Channa argus* | 7 | `configs/wuli.yaml` |

## Benchmark

`benchmark/` 目录包含批量处理压力测试的数据和可视化脚本。

数据集和模型权重请联系：xsji@sdau.edu.cn / yzhao@sdau.edu.cn
