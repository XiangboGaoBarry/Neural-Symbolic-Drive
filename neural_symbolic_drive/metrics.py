"""Trajectory metrics for Neural-Symbolic Drive <PLANNING> outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

PLANNING_RE = re.compile(r"<PLANNING>(.*?)(?:</PLANNING>|$)", re.DOTALL)
POINT_RE = re.compile(r"\[(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]")


def extract_trajectory(text: str | None) -> np.ndarray | None:
    match = PLANNING_RE.search(text or "")
    if not match:
        return None
    points = POINT_RE.findall(match.group(1))
    if not points:
        return None
    return np.array([[float(x), float(y)] for x, y in points], dtype=float)


def headings_deg(trajectory: np.ndarray) -> np.ndarray:
    headings = np.zeros(len(trajectory))
    previous = np.array([0.0, 0.0])
    for i, point in enumerate(trajectory):
        dx = point[0] - previous[0]
        dy = point[1] - previous[1]
        headings[i] = math.degrees(math.atan2(dy, dx))
        previous = point
    return headings


def angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def assistant_text(item: dict) -> str:
    for message in item.get("messages", []):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def compute_metrics(pred_jsonl: str, gt_json: str, name: str) -> dict:
    preds = [json.loads(line) for line in Path(pred_jsonl).read_text().splitlines() if line.strip()]
    gts = json.loads(Path(gt_json).read_text())
    gt_by_id = {item.get("id", str(i)): item for i, item in enumerate(gts)}

    horizons = {"1s": 2, "2s": 4, "3s": 6}
    buckets = {
        key: {"ade": [], "fde": [], "ahe": [], "fhe": [], "miss": 0, "n": 0}
        for key in horizons
    }
    valid = 0

    for pred in preds:
        gt = gt_by_id.get(pred.get("id"))
        if gt is None and "orig_idx" in pred and 0 <= pred["orig_idx"] < len(gts):
            gt = gts[pred["orig_idx"]]
        if gt is None:
            continue
        pred_traj = extract_trajectory(pred.get("predict", ""))
        gt_traj = extract_trajectory(assistant_text(gt))
        if pred_traj is None or gt_traj is None:
            continue
        valid += 1
        pred_heading = headings_deg(pred_traj)
        gt_heading = headings_deg(gt_traj)

        for horizon, n_points in horizons.items():
            n_eff = min(n_points, len(pred_traj), len(gt_traj))
            if n_eff < 1:
                continue
            distances = np.linalg.norm(pred_traj[:n_eff] - gt_traj[:n_eff], axis=1)
            heading_errors = [
                angle_diff_deg(pred_heading[i], gt_heading[i]) for i in range(n_eff)
            ]
            bucket = buckets[horizon]
            bucket["ade"].append(float(np.mean(distances)))
            bucket["fde"].append(float(distances[-1]))
            bucket["ahe"].append(float(np.mean(heading_errors)))
            bucket["fhe"].append(float(heading_errors[-1]))
            bucket["n"] += 1
            if horizon == "3s" and distances[-1] > 2.0:
                bucket["miss"] += 1

    out = {"name": name, "valid": valid, "total": len(preds)}
    for horizon in ("1s", "2s", "3s"):
        bucket = buckets[horizon]
        if bucket["n"] == 0:
            continue
        out[f"ADE@{horizon}"] = float(np.mean(bucket["ade"]))
        out[f"FDE@{horizon}"] = float(np.mean(bucket["fde"]))
        out[f"AHE@{horizon}"] = float(np.mean(bucket["ahe"]))
        out[f"FHE@{horizon}"] = float(np.mean(bucket["fhe"]))
        out[f"MR@{horizon}"] = sum(1 for value in bucket["fde"] if value > 2.0) / bucket["n"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_jsonl", required=True)
    parser.add_argument("--gt_json", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--comparison_json", required=True)
    args = parser.parse_args()

    metrics = compute_metrics(args.pred_jsonl, args.gt_json, args.name)
    print(json.dumps(metrics, indent=2))

    out_path = Path(args.comparison_json)
    comparison = {}
    if out_path.exists():
        comparison = json.loads(out_path.read_text())
    comparison[args.name] = metrics
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
