#!/usr/bin/env python3
"""Create a reproducible train/val/test manifest from CSV metadata."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    raise ValueError(f"None of columns {candidates} found in CSV columns: {list(df.columns)}")


def _read_csv_flexible(csv_path: Path) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
        return _normalize_columns(df)
    except Exception as exc:
        last_error = exc

    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(csv_path, sep=sep)
            return _normalize_columns(df)
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError(f"Could not parse CSV file: {csv_path}") from last_error


def _normalize_label_value(value) -> Optional[str]:
    """Normalize common CSV labels to 'real' or 'fake'."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"0", "0.0", "real", "original", "authentic", "genuine", "pristine"}:
        return "real"
    if text in {"1", "1.0", "fake", "deepfake", "manipulated", "forged", "attack"}:
        return "fake"
    return None


def _split_items(
    items: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[str], List[str], List[str]]:
    if not items:
        return [], [], []
    temp = items[:]
    rng.shuffle(temp)
    n = len(temp)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train = temp[:n_train]
    val = temp[n_train:n_train + n_val]
    test = temp[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(
        description="Create split manifest directly from CSV metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=str, required=True, help="Dataset root folder containing video subfolders")
    parser.add_argument("--csv-dir", type=str, required=True, help="Folder containing CSV files")
    parser.add_argument("--csv-files", type=str, nargs="*", default=None, help="Optional explicit CSV filenames")
    parser.add_argument(
        "--label-driven",
        action="store_true",
        help="Use the CSV label column to create two classes instead of deriving class from path. "
             "Supports 0=real and 1=fake.",
    )
    parser.add_argument("--real-class-name", type=str, default="original",
                        help="Class name to use for label-driven real samples.")
    parser.add_argument("--fake-class-name", type=str, default="fake",
                        help="Class name to use for label-driven fake samples.")
    parser.add_argument("--train-ratio", type=float, default=0.72,
                        help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.14,
                        help="Val split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.14,
                        help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True, help="Output split manifest JSON")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    dataset_root = Path(args.dataset_root).resolve()
    csv_dir = Path(args.csv_dir).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV dir not found: {csv_dir}")

    if args.csv_files:
        csv_paths = [(csv_dir / f).resolve() for f in args.csv_files]
    else:
        csv_paths = sorted(csv_dir.glob("*.csv"))

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    rng = random.Random(args.seed)

    by_class: Dict[str, List[str]] = {}

    for csv_path in csv_paths:
        df = _read_csv_flexible(csv_path)
        try:
            file_col = _find_col(df, ["File Path", "filepath", "path", "file_path"])
        except ValueError:
            print(f"[SKIP] {csv_path.name}: missing 'File Path' column")
            continue
        label_col = None
        try:
            label_col = _find_col(df, ["Label", "label"])
        except ValueError:
            label_col = None
        if args.label_driven and label_col is None:
            raise ValueError(f"--label-driven requires a Label column in {csv_path.name}")

        for _, row in df.iterrows():
            rel_path = str(row[file_col]).strip()
            if not rel_path:
                continue
            rel = Path(rel_path)

            label = _normalize_label_value(row[label_col]) if label_col is not None else None
            if args.label_driven:
                if label is None:
                    print(f"[WARN] unknown label in {csv_path.name}, skipping: {rel_path}")
                    continue
                class_name = args.real_class_name if label == "real" else args.fake_class_name
            else:
                if len(rel.parts) < 2:
                    continue
                class_name = rel.parts[0]

            abs_path = (dataset_root / rel).resolve()
            by_class.setdefault(class_name, []).append(str(abs_path))

            if label_col is not None and not args.label_driven:
                is_real_class = class_name.lower() in {"original", "real", "authentic", "pristine"}
                if label == "fake" and is_real_class:
                    print(f"[WARN] real class with fake/1 label in {csv_path.name}: {rel_path}")
                elif label == "real" and not is_real_class:
                    print(f"[WARN] fake class with real/0 label in {csv_path.name}: {rel_path}")

    for class_name in list(by_class.keys()):
        seen = set()
        uniq = []
        for p in by_class[class_name]:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        by_class[class_name] = uniq

    classes = sorted(by_class.keys())
    if not classes:
        raise ValueError("No video entries found from CSV files.")

    manifest = {
        "seed": args.seed,
        "splits": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "classes": classes,
        "data": {
            "train": {},
            "val": {},
            "test": {},
        },
    }

    for class_name in classes:
        items = by_class[class_name]
        train, val, test = _split_items(
            items,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            rng,
        )
        manifest["data"]["train"][class_name] = train
        manifest["data"]["val"][class_name] = val
        manifest["data"]["test"][class_name] = test
        print(
            f"{class_name}: total={len(items)} train={len(train)} val={len(val)} test={len(test)}"
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved split manifest to: {output}")


if __name__ == "__main__":
    main()
