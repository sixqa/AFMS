#!/usr/bin/env python3
"""
batch_evaluate.py — 批量评估所有物种的 det/kp 模型并提取训练日志指标
"""
import json, os, sys, subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
SPECIES = ['carp', 'crab', 'crayfish', 'ms', 'wuli']
TASKS = ['det', 'kp']

def find_run(species, task):
    pattern = f"runs/{task}_{species}_*"
    dirs = sorted(ROOT.glob(pattern))
    return dirs[-1] if dirs else None

def find_best_ckpt(run_dir):
    best = list(run_dir.glob("best_*.pth"))
    if best:
        return best[0]
    epochs = sorted(run_dir.glob("epoch_*.pth"))
    return epochs[-1] if epochs else None

def find_config(run_dir):
    cfgs = list(run_dir.glob("*.py"))
    return cfgs[0] if cfgs else None

def extract_log_metrics(species, task):
    run_dir = find_run(species, task)
    if not run_dir:
        return {}
    subdirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith('20')]
    if not subdirs:
        return {}
    scalar_file = subdirs[0] / "vis_data" / "scalars.json"
    if not scalar_file.exists():
        return {}
    with open(scalar_file) as f:
        lines = f.readlines()
    result = {}
    # Collect all eval entries and find the best
    best_values = {}
    last_train_line = None
    for line in lines:
        d = json.loads(line)
        if task == 'det' and 'coco/bbox_mAP' in d:
            mAP = d['coco/bbox_mAP']
            if best_values.get('bbox_mAP', -1) < mAP:
                best_values['bbox_mAP'] = mAP
                best_values['bbox_mAP_50'] = d.get('coco/bbox_mAP_50', None)
                best_values['bbox_mAP_75'] = d.get('coco/bbox_mAP_75', None)
        elif task == 'kp' and 'PCK' in d:
            pck = d['PCK']
            if best_values.get('PCK', -1) < pck:
                best_values['PCK'] = pck
                best_values['AUC'] = d.get('AUC', None)
                best_values['NME'] = d.get('NME', None)
                best_values['EPE'] = d.get('EPE', None)
        elif 'loss' in d:
            last_train_line = d
    result.update(best_values)
    if last_train_line:
        result['final_loss'] = last_train_line.get('loss', None)
    return result

def run_evaluation(species, task):
    run_dir = find_run(species, task)
    if not run_dir:
        return None
    ckpt = find_best_ckpt(run_dir)
    config = find_config(run_dir)
    if not ckpt or not config:
        return None
    script = "evaluate_det.py" if task == "det" else "evaluate_kp.py"
    cmd = [
        sys.executable, str(ROOT / "tools" / script),
        "--config", str(config),
        "--checkpoint", str(ckpt)
    ]
    print(f"\n{'='*60}")
    print(f"  Evaluating {species.upper()} - {task}")
    print(f"  Config: {config}")
    print(f"  Ckpt: {ckpt}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
        out = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        print(out)
        if result.stderr:
            print("STDERR:", result.stderr[-1000:])
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {species} {task}")
    except Exception as e:
        print(f"[ERROR] {species} {task}: {e}")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--species", default=None)
    p.add_argument("--task", default=None)
    args = p.parse_args()
    
    sp_list = [args.species] if args.species else SPECIES
    t_list = [args.task] if args.task else TASKS
    results = {}
    
    for task in t_list:
        for sp in sp_list:
            if not args.eval_only:
                metrics = extract_log_metrics(sp, task)
                results[f"{sp}_{task}"] = metrics
                print(f"{sp:10s} {task:3s} | ", end="")
                if metrics:
                    if task == 'det':
                        ma = metrics.get('bbox_mAP')
                        print(f"mAP={ma}" if ma is not None else "mAP=N/A", end=" ")
                        m50 = metrics.get('bbox_mAP_50')
                        print(f"mAP50={m50}" if m50 is not None else "mAP50=N/A", end=" ")
                        fl = metrics.get('final_loss')
                        print(f"loss={fl}" if fl is not None else "loss=N/A")
                    else:
                        pck = metrics.get('PCK')
                        print(f"PCK={pck}" if pck is not None else "PCK=N/A", end=" ")
                        fl = metrics.get('final_loss')
                        print(f"loss={fl}" if fl is not None else "loss=N/A")
                else:
                    print("NO DATA")
            if not args.extract_only:
                run_evaluation(sp, task)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "evals" / f"batch_results_{ts}.json"
    out_path.parent.mkdir(exist_ok=True)
    clean = {}
    for k, v in results.items():
        clean[k] = {kk: (float(vv) if vv is not None else None) for kk, vv in v.items()}
    with open(out_path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"\n[SAVED] {out_path}")

if __name__ == "__main__":
    main()
