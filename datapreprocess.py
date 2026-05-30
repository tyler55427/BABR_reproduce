"""
BRAR Dataset Preprocessing Pipeline
====================================
Reads raw data from datas/ directory and prepares it for train.py.

Input:
  - datas/level_1/*.jpg          (grade 1 panoramic radiographs)
  - datas/level_2/*.jpg          (grade 2 panoramic radiographs)
  - datas/level_3/*.jpg          (grade 3 panoramic radiographs)
  - datas/meta_data.csv          (metadata with 'File name' and 'Level' columns)

Output:
  - datas/images/                (flat directory containing all images)
  - datas/labels.csv             (columns: 新文件名, 等级 — ready for train.py)

Usage:
  python datapreprocess.py
  python datapreprocess.py --data_root ./datas --no-copy
"""

import pandas as pd
import shutil
from pathlib import Path
import argparse
import sys


def preprocess_brar_data(
    data_root="datas",
    meta_csv=None,
    output_labels=None,
    output_images=None,
    copy_images=True,
):
    """
    Preprocess BRAR dataset from raw level_*/ subdirectory format
    into a flat structure ready for train.py.

    Parameters
    ----------
    data_root : str or Path
        Root directory containing level_1/, level_2/, level_3/
        subdirectories (used to locate images and as fallback for
        other paths when they are not explicitly provided).
    meta_csv : str or Path, optional
        Path to the metadata CSV file. Default: {data_root}/meta_data.csv
    output_labels : str or Path, optional
        Path to save the labels CSV. Default: {data_root}/labels.csv
    output_images : str or Path, optional
        Directory to copy images into (flat structure).
        Default: {data_root}/images
    copy_images : bool
        If True, copy all images into the output_images directory.
        If False, only generate the labels CSV.

    Returns
    -------
    labels_df : pd.DataFrame
        DataFrame with columns ['新文件名', '等级'].
    """
    data_path = Path(data_root)

    # ── 1. Resolve paths ──────────────────────────────────────────────
    if meta_csv is None:
        meta_csv = data_path / "meta_data.csv"
    else:
        meta_csv = Path(meta_csv)

    if output_labels is None:
        output_labels = data_path / "labels.csv"
    else:
        output_labels = Path(output_labels)

    if output_images is None:
        output_images = data_path / "images"
    else:
        output_images = Path(output_images)

    # ── 2. Validate input ─────────────────────────────────────────────
    if not meta_csv.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {meta_csv}\n"
            f"Use --meta_csv to specify the correct path."
        )

    # ── 3. Read metadata ──────────────────────────────────────────────
    print(f"[1/4] Reading metadata from {meta_csv} ...")
    meta_df = pd.read_csv(meta_csv)
    print(f"      Loaded {len(meta_df)} records, columns: {list(meta_df.columns)}")

    # ── 4. Build labels dataframe ─────────────────────────────────────
    print("[2/4] Building labels dataframe ...")

    # Map column names: meta_data.csv → train.py expected format
    labels_df = meta_df[["File name", "Level"]].copy()
    labels_df.columns = ["新文件名", "等级"]

    # Keep only valid severity grades (1, 2, 3)
    labels_df = labels_df[labels_df["等级"].isin([1, 2, 3])].copy()
    labels_df["等级"] = labels_df["等级"].astype(int)

    # Verify every label has a corresponding image
    missing_images = []
    for _, row in labels_df.iterrows():
        fname = row["新文件名"]
        found = False
        for level in [1, 2, 3]:
            if (data_path / f"level_{level}" / fname).exists():
                found = True
                break
        if not found:
            missing_images.append(fname)

    if missing_images:
        print(f"      ⚠ Warning: {len(missing_images)} images not found in level_*/ directories")
        if len(missing_images) <= 5:
            for m in missing_images:
                print(f"        - {m}")
    else:
        print(f"      ✓ All {len(labels_df)} labels have matching images")

    # ── 5. Copy images to flat directory ──────────────────────────────
    if copy_images:
        output_images.mkdir(parents=True, exist_ok=True)
        print(f"[3/4] Copying images to {output_images} ...")
        copied_count = 0

        for level in [1, 2, 3]:
            level_dir = data_path / f"level_{level}"
            if not level_dir.exists():
                print(f"      ⚠ Directory not found: {level_dir}, skipping")
                continue

            jpg_files = list(level_dir.glob("*.jpg"))
            for img_file in jpg_files:
                dest = output_images / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                    copied_count += 1

        print(f"      Copied {copied_count} images (already present: "
              f"{len(labels_df) - copied_count})")
    else:
        print("[3/4] Skipping image copy (--no-copy flag set)")

    # ── 6. Save labels CSV ────────────────────────────────────────────
    print(f"[4/4] Saving labels to {output_labels} ...")
    output_labels.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(output_labels, index=False)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Preprocessing complete!")
    print("=" * 55)
    print(f"  Total samples:      {len(labels_df)}")
    print(f"  Grade distribution:")
    for level, count in labels_df["等级"].value_counts().sort_index().items():
        print(f"    等级 {level}: {count} ({count / len(labels_df) * 100:.1f}%)")
    print(f"  Labels saved to:    {output_labels}")
    if copy_images:
        print(f"  Images saved to:    {output_images}")
    print("=" * 55)

    return labels_df


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BRAR dataset for train.py"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="datas",
        help="Root directory containing level_1/, level_2/, level_3/ "
             "subdirectories (default: 'datas')",
    )
    parser.add_argument(
        "--meta_csv",
        type=str,
        default=None,
        help="Path to metadata CSV file (default: {data_root}/meta_data.csv)",
    )
    parser.add_argument(
        "--output_labels",
        type=str,
        default=None,
        help="Path to save labels CSV (default: {data_root}/labels.csv)",
    )
    parser.add_argument(
        "--output_images",
        type=str,
        default=None,
        help="Directory to copy images into, flat structure "
             "(default: {data_root}/images)",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Skip copying images, only generate labels CSV",
    )
    args = parser.parse_args()

    try:
        preprocess_brar_data(
            data_root=args.data_root,
            meta_csv=args.meta_csv,
            output_labels=args.output_labels,
            output_images=args.output_images,
            copy_images=not args.no_copy,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
