#!/usr/bin/env python3
"""
数据集验证脚本
==============
检查所有物种数据集是否规范可用：
  1. 目录结构：train/ + val/ + train_coco.json + val_coco.json
  2. COCO 格式：images/annotations/categories 齐全，file_name 字段存在
  3. 图片完整性：COCO 引用的每张图在对应目录中存在
  4. 标注完整性：每个 annotation 有 bbox 和 keypoints
  5. 类别一致性：categories 定义与标注匹配
"""

import json, os, sys
from pathlib import Path

DATASETS = Path(__file__).resolve().parent.parent / "datasets"
SPECIES = ['carp', 'crab', 'crayfish', 'label', 'ms', 'wuli']

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def check(msg: str, ok: bool, detail: str = ""):
    icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"  {icon} {msg}" + (f"  ({YELLOW}{detail}{RESET})" if detail else ""))


def validate_species(species: str) -> list[str]:
    """返回该物种的所有问题列表"""
    problems = []
    sd = DATASETS / species

    # ── 目录结构 ──
    if not sd.exists():
        return [f"目录不存在"]
    for split in ['train', 'val']:
        d = sd / split
        if not d.exists():
            problems.append(f"缺少 {split}/ 目录")
        elif not d.is_dir():
            problems.append(f"{split}/ 不是目录")

    for split in ['train_coco.json', 'val_coco.json']:
        f = sd / split
        if not f.exists():
            problems.append(f"缺少 {split}")
        elif f.stat().st_size == 0:
            problems.append(f"{split} 为空文件")

    if problems:
        return problems

    # ── COCO 格式校验 ──
    for split in ['train', 'val']:
        coco_path = sd / f"{split}_coco.json"
        img_dir = sd / split

        with open(coco_path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                problems.append(f"{split}_coco.json JSON格式错误: {e}")
                continue

        # 必须字段
        if 'images' not in data:
            problems.append(f"{split}_coco.json 缺少 images")
            continue
        if 'annotations' not in data:
            problems.append(f"{split}_coco.json 缺少 annotations")
        if 'categories' not in data:
            problems.append(f"{split}_coco.json 缺少 categories")
        if problems:
            continue

        images = data['images']
        annotations = data['annotations']
        categories = data['categories']

        # images 不能为空
        if not images:
            problems.append(f"{split}_coco.json images 为空")
            continue

        # 检查每个 image 的字段
        for img in images:
            if 'id' not in img:
                problems.append(f"{split}_coco.json 某 image 缺少 id")
                break
            if 'file_name' not in img:
                problems.append(f"{split}_coco.json 某 image 缺少 file_name")
                break
            if 'height' not in img or 'width' not in img:
                problems.append(f"{split}_coco.json 某 image 缺少 height/width")
                break

        # 检查 categories
        if not categories:
            problems.append(f"{split}_coco.json categories 为空")
        else:
            for cat in categories:
                if 'name' not in cat:
                    problems.append(f"{split}_coco.json categories 缺少 name")
                    break
                if 'keypoints' not in cat:
                    problems.append(f"{split}_coco.json categories 缺少 keypoints")
                    break

        # 图片完整性：COCO 引用的图在目录中存在
        img_files = {f for f in os.listdir(img_dir) if os.path.isfile(img_dir / f)}
        missing_imgs = 0
        for img in images:
            fn = os.path.basename(img['file_name'])
            if fn not in img_files:
                missing_imgs += 1
        if missing_imgs > 0:
            problems.append(f"{split}: COCO 引用 {len(images)} 张图, 目录中有 {len(img_files)} 张, 缺 {missing_imgs} 张")

        # 标注完整性
        ann_without_bbox = 0
        ann_without_kp = 0
        ann_with_partial_kp = 0
        for ann in annotations:
            if 'bbox' not in ann or len(ann.get('bbox', [])) != 4:
                ann_without_bbox += 1
            kps = ann.get('keypoints', [])
            if not kps:
                ann_without_kp += 1
            elif sum(1 for i in range(2, len(kps), 3) if kps[i] > 0) < len(kps) // 3:
                ann_with_partial_kp += 1

        if ann_without_bbox > 0:
            problems.append(f"{split}: {ann_without_bbox} 条标注缺 bbox")
        if ann_without_kp > 0:
            problems.append(f"{split}: {ann_without_kp} 条标注缺 keypoints")

        # image_id 一致性
        img_ids = {img['id'] for img in images}
        orphan_anns = sum(1 for ann in annotations if ann.get('image_id') not in img_ids)
        if orphan_anns > 0:
            problems.append(f"{split}: {orphan_anns} 条标注引用了不存在的 image_id")

    return problems


def main():
    print(f"\n{BOLD}{'═' * 56}{RESET}")
    print(f"{BOLD}  AFMS 数据集验证{RESET}")
    print(f"{BOLD}{'═' * 56}{RESET}")
    print(f"  路径: {DATASETS}")
    print()

    total_problems = 0
    for s in SPECIES:
        print(f"{CYAN}── {s.upper()} ──{RESET}")
        problems = validate_species(s)
        if not problems:
            # 统计信息
            sd = DATASETS / s
            train_coco = json.load(open(sd / "train_coco.json"))
            val_coco = json.load(open(sd / "val_coco.json"))
            n_kp = len(train_coco['categories'][0]['keypoints'])
            train_ann = len(train_coco['annotations'])
            val_ann = len(val_coco['annotations'])
            check(f"数据集完整 | {n_kp}关键点", True)
            check(f"train: {len(train_coco['images'])}张图, {train_ann}条标注", True)
            check(f"val:   {len(val_coco['images'])}张图, {val_ann}条标注", True)
        else:
            total_problems += len(problems)
            for p in problems:
                check(p, False)

        print()

    print(f"{BOLD}{'═' * 56}{RESET}")
    if total_problems == 0:
        print(f"  {GREEN}{BOLD}全部 {len(SPECIES)} 个数据集验证通过 ✓{RESET}")
    else:
        print(f"  {RED}{BOLD}共发现 {total_problems} 个问题{RESET}")
    print(f"{BOLD}{'═' * 56}{RESET}\n")

    return 0 if total_problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
