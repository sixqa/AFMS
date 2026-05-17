#!/usr/bin/env python3
"""
train_kp.py — 关键点模型统一训练入口

三种调用方式：
    python tools/train_kp.py --yaml configs/shrimp.yaml   # 用户添加新物种：传 YAML 文件
    python tools/train_kp.py --species carp               # 快捷方式：直接选物种
    python tools/train_kp.py --config path/to/config.py   # 高级：自定义配置文件

可选参数：
    --epochs N      覆盖训练轮数
    --work-dir PATH 自定义工作目录
    --variant NAME  自定义归档目录名
    --no-archive    不归档到 models/
"""

import argparse, os, sys, shutil, yaml
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmengine.config import Config
from mmengine.runner import Runner

import hooks


def get_available_configs() -> list:
    """扫描 configs/kp/ 获取所有配置文件"""
    cfg_dir = Path(__file__).parent.parent / "configs" / "kp"
    if not cfg_dir.exists():
        return []
    return sorted([f.stem for f in cfg_dir.glob("*.py") if f.stem != "__init__"])


def list_species_and_exit():
    configs = get_available_configs()
    print("可用物种:")
    for s in configs:
        print(f"  - {s}")
    print()
    print("用法: python tools/train_kp.py --yaml configs/shrimp.yaml")
    print("  或: python tools/train_kp.py --species 物种名")
    print("  或: python tools/train_kp.py --config path/to/config.py")
    sys.exit(0)


def parse_args():
    p = argparse.ArgumentParser(
        description="AFMS 关键点模型训练",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--yaml", help="物种 YAML 配置文件路径（推荐）")
    g.add_argument("--config", help="配置文件路径")
    g.add_argument("--species", "-s",
                   help="物种名，自动读取 configs/kp/{物种}.py")
    p.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    p.add_argument("--work-dir", default=None, help="自定义工作目录")
    p.add_argument("--variant", default=None, help="自定义归档目录名")
    p.add_argument("--load-from", default=None, help="从指定权重文件继续训练（pretrained）")
    p.add_argument("--no-archive", action="store_true", help="不归档到 models/")
    args = p.parse_args()
    if not args.config and not args.species and not args.yaml:
        list_species_and_exit()
    return args


def resolve_config(args) -> Path:
    if args.config:
        return Path(args.config)
    if args.yaml:
        with open(args.yaml) as f:
            cfg_data = yaml.safe_load(f)
        species_name = cfg_data["species"]["name"].lower()
        cfg = Path(__file__).parent.parent / "configs" / "kp" / f"{species_name}.py"
        if not cfg.exists():
            print(f"[ERROR] 配置文件不存在: {cfg}")
            print(f"       请先运行: python tools/generate.py {args.yaml}")
            sys.exit(1)
        return cfg
    cfg = Path(__file__).parent.parent / "configs" / "kp" / f"{args.species}.py"
    if not cfg.exists():
        print(f"[ERROR] 配置文件不存在: {cfg}")
        print(f"       请先运行: python tools/generate.py configs/{args.species}.yaml")
        sys.exit(1)
    return cfg


def fix_paths(cfg, trainer_root: Path):
    import re
    for loader_key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
        loader = cfg.get(loader_key)
        if loader and 'dataset' in loader:
            ds = loader['dataset']
            if 'data_root' in ds and isinstance(ds['data_root'], str) and ds['data_root'].startswith('..'):
                rel = re.sub(r'^(\.\./)+', '', ds['data_root']).lstrip('/')
                ds['data_root'] = str((trainer_root / rel).resolve())
    for eval_key in ['val_evaluator', 'test_evaluator']:
        ev = cfg.get(eval_key)
        if ev:
            items = ev if isinstance(ev, list) else [ev]
            for e in items:
                if isinstance(e, dict) and 'ann_file' in e and isinstance(e['ann_file'], str) and e['ann_file'].startswith('..'):
                    rel = re.sub(r'^(\.\./)+', '', e['ann_file']).lstrip('/')
                    e['ann_file'] = str((trainer_root / rel).resolve())


def main():
    args = parse_args()
    cfg_path = resolve_config(args)
    species_name = cfg_path.stem
    trainer_root = Path(__file__).resolve().parent.parent

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant = args.variant or f"v{ts[:8]}"
    wd = Path(args.work_dir or f"runs/kp_{species_name}_{ts}").resolve()
    wd.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  AFMS Keypoint Trainer")
    print(f"  Species:    {species_name}")
    print(f"  Config:     {cfg_path}")
    print(f"  Work Dir:   {wd}")
    print("=" * 60)

    cfg = Config.fromfile(str(cfg_path))
    cfg.work_dir = str(wd)
    if args.epochs:
        cfg.train_cfg.max_epochs = args.epochs
    if args.load_from:
        cfg.load_from = str(Path(args.load_from).resolve())
        print(f"[LOAD] 从权重继续训练: {cfg.load_from}")
    fix_paths(cfg, trainer_root)

    runner = Runner.from_cfg(cfg)
    runner.train()

    # 归档最佳模型
    if not args.no_archive:
        archive = trainer_root / f"models/{species_name}/{variant}/keypoint"
        archive.mkdir(parents=True, exist_ok=True)
        for pth in wd.glob("best_*.pth"):
            shutil.copy2(str(pth), str(archive / pth.name))
        shutil.copy2(str(cfg_path), str(archive / f"{species_name}_kp.py"))
        meta = f"species: {species_name}\nvariant: {variant}\ntimestamp: {ts}\n"
        (archive / "meta.txt").write_text(meta)
        print(f"\n[ARCHIVE] Best model -> {archive}")

    print(f"\n[DONE] Work dir: {wd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
