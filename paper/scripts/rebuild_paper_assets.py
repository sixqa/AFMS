#!/usr/bin/env python3
"""
Rebuild final paper assets for AFMS.

This orchestration script intentionally does not contain plotting logic or
manuscript-writing logic. Figure drawing lives in dedicated plot_*.py scripts,
and manuscript prose stays in paper/main.tex.

Run from the repository root:
    conda run -n mmpose python paper/scripts/rebuild_paper_assets.py
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

from paper_asset_data import BENCHMARK, FINAL, PAPER, ROOT, collect_trainer_table


def write_trainer_table(rows: list[dict]) -> Path:
    out = FINAL / "trainer_metrics.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def copy_if_exists(src: Path, dst_name: str, manifest: list[dict], purpose: str) -> None:
    if not src.exists():
        manifest.append({"file": dst_name, "source": str(src.relative_to(ROOT)), "purpose": purpose, "status": "missing"})
        return
    dst = FINAL / dst_name
    shutil.copy2(src, dst)
    manifest.append({"file": dst.name, "source": str(src.relative_to(ROOT)), "purpose": purpose, "status": "copied"})


def write_manifest(entries: list[dict]) -> Path:
    out = FINAL / "manifest.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "source", "purpose", "status"])
        writer.writeheader()
        writer.writerows(entries)
    return out


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    rows = collect_trainer_table()
    metrics_csv = write_trainer_table(rows)

    manifest: list[dict] = [
        {
            "file": metrics_csv.name,
            "source": "trainer/configs/*.yaml; trainer/datasets/*/*_coco.json; trainer/evals/batch_results_*.json",
            "purpose": "Tabular source data for protocol-defined validation report",
            "status": "generated",
        }
    ]

    copies = [
        (PAPER / "figure" / "模式图.jpg", "fig_system_architecture.jpg", "Hand-drawn AFMS hardware-software-model overview"),
        (PAPER / "figure" / "模型.jpg", "fig_model_trainer_workflow.jpg", "Hand-drawn trainer and inference workflow"),
        (PAPER / "figure" / "3d结构光+标签可选模块.jpg", "fig_structured_light.jpg", "3D structured-light hardware concept with optional tag module"),
        (PAPER / "figure" / "3d结构光实验.jpg", "fig_structured_light_experiment.jpg", "3D structured-light experiment photograph"),
        (BENCHMARK / "figures" / "fig_stress_batch_combined.pdf", "fig_stress_batch_combined.pdf", "Combined operational batch-processing timeline and throughput"),
        (BENCHMARK / "figures" / "fig_stress_batch_combined.svg", "fig_stress_batch_combined.svg", "Combined operational batch-processing timeline and throughput"),
        (BENCHMARK / "figures" / "fig_stress_batch_combined.tiff", "fig_stress_batch_combined.tiff", "Combined operational batch-processing timeline and throughput"),
    ]
    for src, dst_name, purpose in copies:
        copy_if_exists(src, dst_name, manifest, purpose)

    manifest_path = write_manifest(manifest)
    print(f"Wrote {metrics_csv.relative_to(ROOT)}")
    print(f"Wrote {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
