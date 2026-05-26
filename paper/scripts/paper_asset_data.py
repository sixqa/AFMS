"""Data collection helpers for AFMS paper assets.

This module reads project outputs and normalizes them into tabular records.
It intentionally contains no plotting code and no manuscript-writing code.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FINAL = PAPER / "figures" / "final"
TRAINER = ROOT / "trainer"
BENCHMARK = ROOT / "benchmark"
SOFTWARE_BACKEND = ROOT / "software" / "backend"

SPECIES = ["carp", "crab", "crayfish", "ms", "wuli"]
SPECIES_LABEL = {
    "carp": "Carp",
    "crab": "Crab",
    "crayfish": "Crayfish",
    "ms": "Bass",
    "wuli": "Snakehead",
}
SCIENTIFIC_NAME = {
    "carp": "Cyprinus carpio",
    "crab": "Eriocheir sinensis",
    "crayfish": "Procambarus clarkii",
    "ms": "Micropterus salmoides",
    "wuli": "Channa argus",
}
RUNTIME_NAME = {
    "carp": "carp",
    "crab": "crab",
    "crayfish": "crayfish",
    "ms": "ms",
    "wuli": "wuli",
}


def latest_batch_results() -> dict:
    files = sorted((TRAINER / "evals").glob("batch_results_*.json"))
    if not files:
        raise FileNotFoundError("No trainer/evals/batch_results_*.json file found")
    with files[-1].open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dataset_counts(species: str) -> tuple[int, int, int, int]:
    counts = []
    for split in ("train", "val"):
        path = TRAINER / "datasets" / species / f"{split}_coco.json"
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        counts.extend([len(data.get("images", [])), len(data.get("annotations", []))])
    train_img, train_ann, val_img, val_ann = counts
    return train_img, val_img, train_ann, val_ann


def protocol_summary(species: str) -> dict:
    runtime_name = RUNTIME_NAME[species]
    protocol_path = SOFTWARE_BACKEND / "protocols" / f"{runtime_name}.yaml"
    model_dir = SOFTWARE_BACKEND / "models" / runtime_name
    detector = model_dir / "rtmdet2onnx" / "end2end.onnx"
    pose = model_dir / "rtmpose2onnx" / "end2end.onnx"
    summary = {
        "runtime_protocol": int(protocol_path.exists()),
        "runtime_detector": int(detector.exists()),
        "runtime_pose": int(pose.exists()),
        "protocol_keypoints": None,
        "protocol_measurements": None,
        "protocol_skeleton": None,
        "protocol_lines": None,
        "protocol_angles": None,
        "protocol_ratios": None,
    }
    if not protocol_path.exists():
        return summary
    protocol = read_yaml(protocol_path)
    phenotype = protocol.get("phenotype", {}) or {}
    lines = phenotype.get("lines", {}) or {}
    angles = phenotype.get("angles", {}) or {}
    scoring = protocol.get("scoring", {}) or {}
    ratios = scoring.get("ratio", {}) or {}
    summary.update(
        {
            "protocol_keypoints": len(protocol.get("keypoints", []) or []),
            "protocol_measurements": len(lines) + len(angles) + len(ratios),
            "protocol_skeleton": len(protocol.get("skeleton", []) or []),
            "protocol_lines": len(lines),
            "protocol_angles": len(angles),
            "protocol_ratios": len(ratios),
        }
    )
    return summary


def collect_trainer_table() -> list[dict]:
    batch = latest_batch_results()
    rows = []
    for sp in SPECIES:
        cfg_path = TRAINER / "configs" / f"{sp}.yaml"
        cfg = read_yaml(cfg_path)
        protocol = protocol_summary(sp)
        train_img, val_img, train_ann, val_ann = dataset_counts(sp)
        det = batch[f"{sp}_det"]
        kp = batch[f"{sp}_kp"]
        protocol_keypoints = protocol["protocol_keypoints"] or len(cfg["keypoints"])
        protocol_measurements = protocol["protocol_measurements"] if protocol["protocol_measurements"] is not None else 0
        rows.append(
            {
                "species": sp,
                "label": SPECIES_LABEL[sp],
                "scientific_name": SCIENTIFIC_NAME[sp],
                "keypoints": len(cfg["keypoints"]),
                "protocol_keypoints": protocol_keypoints,
                "protocol_measurements": protocol_measurements,
                "protocol_skeleton": protocol["protocol_skeleton"] or 0,
                "protocol_lines": protocol["protocol_lines"] or 0,
                "protocol_angles": protocol["protocol_angles"] or 0,
                "protocol_ratios": protocol["protocol_ratios"] or 0,
                "runtime_protocol": protocol["runtime_protocol"],
                "runtime_detector": protocol["runtime_detector"],
                "runtime_pose": protocol["runtime_pose"],
                "train_images": train_img,
                "val_images": val_img,
                "train_annotations": train_ann,
                "val_annotations": val_ann,
                "det_map": det["bbox_mAP"],
                "det_map50": det["bbox_mAP_50"],
                "det_map75": det["bbox_mAP_75"],
                "kp_pck": kp["PCK"],
                "kp_auc": kp["AUC"],
                "kp_nme": kp["NME"],
            }
        )
    return rows
