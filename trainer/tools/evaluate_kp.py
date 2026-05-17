#!/usr/bin/env python3
"""
evaluate_kp.py — 关键点模型统一评估

用法：
    # 从训练归档评估
    python tools/evaluate_kp.py --species carp --variant v20250502

    # 直接指定配置和权重
    python tools/evaluate_kp.py --config runs/kp_carp_xxx/carp_kp.py \\
                                --checkpoint runs/kp_carp_xxx/best_PCK_xxx.pth
"""

import argparse, os, sys
from datetime import datetime
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    p = argparse.ArgumentParser(description="AFMS 关键点模型评估")
    p.add_argument("--species", help="物种名")
    p.add_argument("--variant", default=None, help="归档变体名")
    p.add_argument("--config", default=None, help="配置文件路径")
    p.add_argument("--checkpoint", default=None, help="权重文件路径")
    return p.parse_args()


def find_best(d: Path):
    pths = list(d.glob("best_*.pth")) or sorted(d.glob("epoch_*.pth"))
    return pths[0] if pths else None


def fix_paths(cfg, trainer_root: Path):
    for loader_key in ['test_dataloader', 'val_dataloader']:
        loader = cfg.get(loader_key)
        if loader and 'dataset' in loader:
            ds = loader['dataset']
            if 'data_root' in ds and isinstance(ds['data_root'], str) and ds['data_root'].startswith('..'):
                rel = ds['data_root'].lstrip('../').lstrip('/')
                ds['data_root'] = str((trainer_root / rel).resolve())
    for eval_key in ['val_evaluator', 'test_evaluator']:
        ev = cfg.get(eval_key)
        if ev:
            items = ev if isinstance(ev, list) else [ev]
            for e in items:
                if isinstance(e, dict) and 'ann_file' in e and isinstance(e['ann_file'], str) and e['ann_file'].startswith('..'):
                    rel = e['ann_file'].lstrip('../').lstrip('/')
                    e['ann_file'] = str((trainer_root / rel).resolve())


def main():
    args = parse_args()
    trainer_root = Path(__file__).resolve().parent.parent

    config_path = None
    ckpt_path = None

    if args.config and args.checkpoint:
        config_path = Path(args.config)
        ckpt_path = Path(args.checkpoint)
    elif args.species:
        variant = args.variant or "v1"
        if args.checkpoint:
            # 直接指定 checkpoint，从 configs/ 读取配置
            config_path = trainer_root / f"configs/kp/{args.species}.py"
            ckpt_path = Path(args.checkpoint)
        else:
            # 从 models/ 目录自动查找配置和最佳权重
            model_dir = trainer_root / f"models/{args.species}/{variant}/keypoint"
            if not model_dir.exists():
                print(f"[ERROR] 评估目录不存在: {model_dir}")
                print(f"       请先训练: python tools/train_kp.py --species {args.species}")
                sys.exit(1)
            config_path = next(model_dir.glob("*.py"), None)
            ckpt_path = find_best(model_dir)
    else:
        print("[ERROR] 请指定 --species --variant 或 --config --checkpoint")
        sys.exit(1)

    if not config_path or not ckpt_path:
        print(f"[ERROR] 找不到配置或权重文件")
        print(f"  Config: {config_path}")
        print(f"  Ckpt:   {ckpt_path}")
        sys.exit(1)

    print(f"评估 {config_path.stem} 关键点模型...")
    print(f"  配置: {config_path}")
    print(f"  权重: {ckpt_path}")

    cfg = Config.fromfile(str(config_path))
    cfg.load_from = str(ckpt_path)
    fix_paths(cfg, trainer_root)
    
    # 动态设置 work_dir，避免冲突
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.work_dir = str(trainer_root / f"evals/tmp_{config_path.stem}_{ts}")

    runner = Runner.from_cfg(cfg)

    import time
    _t0 = time.time()
    metrics = runner.test()
    _elapsed = time.time() - _t0

    # 推理速度统计
    _speed_info = ""
    try:
        _n = len(runner.test_loop.dataloader.dataset)
        _avg_ms = _elapsed / _n * 1000
        _fps = _n / _elapsed
        _speed_info = f"Total: {_elapsed:.1f}s | Samples: {_n} | Avg: {_avg_ms:.1f}ms/img ({_fps:.1f} FPS)"
        print(f"\n⏱️  {_speed_info}")
    except Exception:
        print(f"\n⏱️  Total test time: {_elapsed:.1f}s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = f"# Keypoint Report: {config_path.stem}\n\n"
    report += f"Config: {config_path}\nCheckpoint: {ckpt_path}\nDate: {ts}\n\n"
    report += "| Metric | Value |\n|--------|-------|\n"
    for k, v in metrics.items():
        report += f"| {k} | {v:.4f} |\n"
    if _speed_info:
        report += f"\n---\n### Speed\n```\n{_speed_info}\n```\n"

    evals_dir = trainer_root / "evals"
    evals_dir.mkdir(exist_ok=True)
    report_path = evals_dir / f"{config_path.stem}_kp_report.html"
    html_content = f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{config_path.stem} KP Report</title></head><body><pre>{report}</pre></body></html>'
    report_path.write_text(html_content, encoding="utf-8")
    print(f"\n[REPORT] {report_path}")


if __name__ == "__main__":
    main()
