#!/usr/bin/env python3
"""
export_onnx.py — 统一 ONNX 导出工具

将训练好的模型（.pth）导出为 ONNX 格式，供测量软件端推理使用。
支持模型精简（publish），去除冗余的优化器状态等信息。

用法：
    # 精简 + 导出 ONNX
    python tools/export_onnx.py --species carp --variant v20250502 --task all

    # 仅导出 ONNX（不精简）
    python tools/export_onnx.py --species carp --no-publish

    # 仅精简模型（不导出 ONNX）
    python tools/export_onnx.py --species carp --task all --publish-only

    # 直接指定配置和权重
    python tools/export_onnx.py --config configs/carp.py --checkpoint models/carp/v1/detection/best_*.pth

要求：ONNX 导出依赖 mmdeploy。请先安装：
    pip install mmdeploy
"""

import argparse, hashlib, json, os, sys, shutil
from datetime import datetime
from pathlib import Path

def _yaml_to_species(yaml_path: str) -> str:
    """从 YAML 文件中读取物种名"""
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["species"]["name"].lower()


def parse_args():
    p = argparse.ArgumentParser(description="AFMS ONNX 导出")
    p.add_argument("--yaml", default=None, help="物种 YAML 配置文件路径")
    p.add_argument("--species", help="物种名")
    p.add_argument("--variant", default=None, help="训练变体名（默认取最新的）")
    p.add_argument("--config", default=None, help="配置文件路径（与 --checkpoint 配合使用）")
    p.add_argument("--checkpoint", default=None, help="权重文件路径")
    p.add_argument("--task", choices=["detection", "keypoint", "all"], default="all",
                    help="导出任务类型")
    p.add_argument("--publish", action="store_true", default=True,
                    help="精简模型（去除优化器状态等，默认开启）")
    p.add_argument("--no-publish", dest="publish", action="store_false",
                    help="不精简模型（保留所有状态）")
    p.add_argument("--publish-only", action="store_true", default=False,
                    help="仅精简模型，不导出 ONNX")
    return p.parse_args()


def find_latest_variant(root: Path, species: str):
    models_dir = root / "models" / species
    if not models_dir.exists():
        return None
    variants = sorted(models_dir.iterdir())
    return variants[-1].name if variants else None


def find_best(d: Path):
    pths = list(d.glob("best_*.pth")) or sorted(d.glob("epoch_*.pth"))
    return str(pths[0]) if pths else None


def find_config(d: Path):
    cfgs = list(d.glob("*.py"))
    return str(cfgs[0]) if cfgs else None


def publish_checkpoint(in_file: Path, out_file: Path) -> Path:
    """精简模型：只保留 meta 和 state_dict，去除优化器状态等冗余数据"""
    import torch

    checkpoint = torch.load(in_file, map_location='cpu')

    save_keys = ['meta', 'state_dict']
    ckpt_keys = list(checkpoint.keys())
    removed = []
    for k in ckpt_keys:
        if k not in save_keys:
            removed.append(k)
            checkpoint.pop(k, None)

    if removed:
        print(f"  精简：移除 {len(removed)} 个键: {removed[:3]}{'...' if len(removed) > 3 else ''}")

    torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)

    with open(out_file, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:8]

    date_str = datetime.now().strftime('%Y%m%d%H%M%S')
    final_file = out_file.parent / f"{out_file.stem}-{sha}_{date_str}.pth"
    shutil.move(str(out_file), str(final_file))
    print(f"  精简完成: {final_file.name}")
    return final_file


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    # ── 确定配置和权重 ──
    # ── 从 YAML 解析物种名 ──
    if args.yaml:
        args.species = _yaml_to_species(args.yaml)

    if args.config and args.checkpoint:
        # 直接指定模式
        tasks = ["detection", "keypoint"] if args.task == "all" else [args.task]
        config_path = Path(args.config)
        ckpt_path = Path(args.checkpoint)
        species_from_path = config_path.stem

        for task in tasks:
            print(f"\n--- 导出 {task} ONNX: {species_from_path} ---")
            onnx_dir = root / f"export/{species_from_path}/onnx/{task}"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            _export_one(task, config_path, ckpt_path, onnx_dir,
                        do_publish=args.publish, publish_only=args.publish_only)

    elif args.species:
        # 从归档模式
        variant = args.variant or find_latest_variant(root, args.species)
        if not variant:
            print(f"[ERROR] {args.species} 没有训练产物，请先训练")
            print(f"       python tools/train_det.py --species {args.species}")
            sys.exit(1)

        tasks = ["detection", "keypoint"] if args.task == "all" else [args.task]
        for task in tasks:
            model_dir = root / f"models/{args.species}/{variant}/{task}"
            if not model_dir.exists():
                print(f"  [SKIP] {model_dir} 不存在")
                continue

            config_path = find_config(model_dir)
            ckpt_path = find_best(model_dir)
            if not config_path or not ckpt_path:
                print(f"  [SKIP] {model_dir} 中缺少配置或权重")
                continue

            print(f"\n--- 导出 {task} ONNX: {args.species}/{variant} ---")
            onnx_dir = root / f"models/{args.species}/{variant}/onnx/{task}"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            _export_one(task, Path(config_path), Path(ckpt_path), onnx_dir,
                        do_publish=args.publish, publish_only=args.publish_only)

    else:
        print("[ERROR] 请指定 --species 或 --config --checkpoint")
        sys.exit(1)

    print("\n[DONE]")


def _export_one(task: str, config_path: Path, ckpt_path: Path, onnx_dir: Path,
                do_publish: bool = True, publish_only: bool = False):
    """执行单个 ONNX 导出

    Args:
        task: 任务类型 (detection/keypoint)
        config_path: 配置文件路径
        ckpt_path: 权重文件路径
        onnx_dir: ONNX 输出目录
        do_publish: 是否精简模型
        publish_only: 是否仅精简不导出 ONNX
    """
    species = config_path.stem

    # ── 模型精简（可选） ──
    published_ckpt = ckpt_path
    if do_publish:
        publish_dir = onnx_dir.parent / "published"
        publish_dir.mkdir(parents=True, exist_ok=True)
        publish_prefix = publish_dir / f"{species}_{task}_publish"
        published_ckpt = publish_checkpoint(ckpt_path, publish_prefix)
        print(f"  精简后文件: {published_ckpt}")

    if publish_only:
        print(f"  [--publish-only] 跳过 ONNX 导出")
        return

    # ── 导出 ONNX ──
    onnx_path = onnx_dir / "end2end.onnx"

    # ── 尝试用 mmdeploy 导出 ──
    deploy_cfg_map = {
        "detection": "mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py",
        "keypoint": "mmdeploy/configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py",
    }

    dc = deploy_cfg_map.get(task)
    if dc and Path(dc).exists():
        import subprocess
        cmd = [
            "python", "-m", "mmdeploy.tools.deploy",
            dc, str(config_path), str(published_ckpt),
            str(onnx_dir / "dummy.jpg"),
            "--work-dir", str(onnx_dir / "out"),
        ]
        print(f"  运行: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  [ERROR] mmdeploy 导出失败，请检查 mmdeploy 是否正确安装")
            return
        print(f"  [DONE] ONNX -> {onnx_path}")
    else:
        print(f"  [ERROR] mmdeploy 未安装或配置文件不存在")
        print(f"         请先安装 mmdeploy: pip install mmdeploy")
        return

    # ── 写入导出元信息 ──
    meta = {
        "species": species,
        "task": task,
        "date": datetime.now().isoformat(),
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "published_checkpoint": str(published_ckpt) if do_publish else None,
        "onnx": str(onnx_path),
    }
    (onnx_dir / "export_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  [META] {onnx_dir}/export_meta.json")


if __name__ == "__main__":
    main()
