"""preprocess.py -- CLW-HFL Part 3, section 3.1.

Loads the UNSW-NB15 training/testing CSVs, MinMax-scales numeric features,
one-hot-encodes categorical features, LabelEncodes `attack_cat` into the
canonical 10-class global label space, and writes:

    <processed_dir>/label_map.json        (single source of truth, Part 1 1.2)
    <processed_dir>/feature_columns.json   (final column order, for reproducibility)
    <processed_dir>/train.npz              (X, y_global)
    <processed_dir>/val.npz                (X, y_global)
    <processed_dir>/test.npz               (X, y_global)

IMPORTANT -- there is no official third UNSW-NB15 validation file. The
public release only ships UNSW_NB15_training-set.csv and
UNSW_NB15_testing-set.csv. So:
  - `val.npz` is a stratified (by attack_cat) carve-out FROM the official
    training-set.csv, controlled by --val_fraction (default 0.15). This is
    what each layer script uses for its own per-round validation subset
    (Part 3, section 3.2) to track convergence -- it is held out of client
    training data.
  - `test.npz` is the ENTIRE official testing-set.csv, completely untouched
    and never used for any training-time decision. This is the held-out
    global test set referenced in Part 6, step 2.
  - The MinMax scaler and one-hot encoder are fit ONLY on the post-carve
    training portion (not on val, not on test), to avoid leaking
    validation/test statistics into preprocessing -- the same standard you'd
    want for the labels/model themselves.

Hard dataset-identity guard (Part 1, section 1.1 / Part 2 item 2): after
encoding, the resulting label set is asserted against the confirmed ground
truth from Part 1 section 1.2. If it does not match exactly, this script
raises loudly instead of silently writing a wrong/placeholder mapping --
this is precisely the failure mode that let a CSE-CIC-IDS2018 label file
leak into a previous run of this pipeline.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.label_utils import CANONICAL_LABEL_MAP, assert_canonical_label_map

# Columns that are never model features: 'id' is a row index, 'label' is the
# binary attack/normal flag that is redundant with (and leaks) attack_cat,
# and 'attack_cat' is the target itself.
NON_FEATURE_COLUMNS = {"id", "label", "attack_cat"}

# Real-world UNSW-NB15 quirk: the public training/testing CSVs are not
# perfectly consistent in how they spell/capitalize attack_cat values
# (e.g. stray leading/trailing whitespace, "Backdoors" vs "Backdoor" across
# releases). We normalize defensively BEFORE label-encoding so the encoder
# does not split one true class into two by accident -- silently fragmenting
# a class like this is exactly the kind of bug this rebuild is trying to
# eliminate.
ATTACK_CAT_NORMALIZATION = {
    "backdoors": "Backdoor",
    "backdoor": "Backdoor",
}


def normalize_attack_cat(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    lowered = cleaned.str.lower()
    normalized = lowered.map(lambda v: ATTACK_CAT_NORMALIZATION.get(v, None))
    # Where we didn't have an explicit override, title-case-normalize so
    # casing differences (e.g. "normal" vs "Normal") don't fragment classes.
    fallback = cleaned.where(cleaned.str.lower() != "normal", "Normal")
    out = normalized.fillna(fallback)
    return out


def assert_is_unsw_nb15(df: pd.DataFrame):
    """Defensive check (Part 1, section 1.1): confirm the loaded CSV looks
    like UNSW-NB15 and not e.g. a CSE-CIC-IDS2018 export. UNSW-NB15 has a
    known, distinctive column set; CSE-CIC-IDS2018 does not have
    'attack_cat' and uses entirely different column names (e.g. 'Label',
    'Flow Duration', 'Dst Port', ...).
    """
    expected_unsw_columns = {"attack_cat", "proto", "service", "state", "label"}
    missing = expected_unsw_columns - set(df.columns)
    if missing:
        raise ValueError(
            "This does not look like a UNSW-NB15 CSV -- missing expected "
            f"column(s) {missing}. CLW-HFL requires UNSW-NB15 "
            "(UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv), NOT "
            "CSE-CIC-IDS2018 or any other IDS dataset. Check --train_csv / "
            "--test_csv point at the right files."
        )


def split_feature_columns(df: pd.DataFrame):
    """Numeric vs categorical split, based on actual dtype rather than a
    naive `dtype == object` check. Newer pandas (>=2.x with the string
    dtype backend, and pandas 3.x by default) stores text columns under a
    dedicated StringDtype rather than legacy `object`, so checking
    `dtype == object` alone silently misses categorical columns like
    proto/service/state. Using pandas' own is_numeric_dtype is robust to
    that and to any future dtype-backend changes.
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", default="UNSW_NB15/UNSW_NB15_training-set.csv")
    parser.add_argument("--test_csv", default="UNSW_NB15/UNSW_NB15_testing-set.csv")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--subsample", type=float, default=None,
                         help="Fraction (0-1] of each split to keep, for fast "
                              "iteration on the target laptop hardware "
                              "(Part 1, section 0). Stratified by attack_cat.")
    parser.add_argument("--val_fraction", type=float, default=0.15,
                         help="Fraction of the official training-set.csv to "
                              "carve out as a stratified validation subset "
                              "(Part 3, section 3.2). UNSW-NB15 ships no "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                              "official validation file, so this is taken "
                              "from training data, never from testing-set.csv.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.train_csv) or not os.path.exists(args.test_csv):
        raise FileNotFoundError(
            f"Expected UNSW-NB15 CSVs at '{args.train_csv}' and '{args.test_csv}'. "
            "Download UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv "
            "from the official UNSW-NB15 release and place them under data/, "
            "or pass --train_csv/--test_csv explicitly."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[preprocess] loading {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    print(f"[preprocess] loading {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)

    assert_is_unsw_nb15(train_df)
    assert_is_unsw_nb15(test_df)

    train_df["attack_cat"] = normalize_attack_cat(train_df["attack_cat"])
    test_df["attack_cat"] = normalize_attack_cat(test_df["attack_cat"])

    if args.subsample is not None:
        assert 0 < args.subsample <= 1.0, "--subsample must be in (0, 1]"
        if args.subsample < 1.0:
            train_keep_idx, _ = train_test_split(
                np.arange(len(train_df)),
                train_size=args.subsample,
                random_state=args.seed,
                stratify=train_df["attack_cat"],
            )
            test_keep_idx, _ = train_test_split(
                np.arange(len(test_df)),
                train_size=args.subsample,
                random_state=args.seed,
                stratify=test_df["attack_cat"],
            )
            train_df = train_df.iloc[train_keep_idx].reset_index(drop=True)
            test_df = test_df.iloc[test_keep_idx].reset_index(drop=True)
        print(f"[preprocess] --subsample={args.subsample} -> "
              f"train={len(train_df)} rows, test={len(test_df)} rows")

    # --- Label encoding (Part 1, section 1.2) ---------------------------
    # Fit on the UNION of train+test attack_cat strings so both splits get
    # an identical encoding even if a class happens to be rare/missing in
    # one split. sklearn's LabelEncoder sorts classes_ alphabetically.
    all_classes = pd.concat([train_df["attack_cat"], test_df["attack_cat"]]).unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(all_classes))

    label_map = {str(cls): int(idx) for idx, cls in enumerate(label_encoder.classes_)}
    print(f"[preprocess] generated label_map: {label_map}")
    assert_canonical_label_map(label_map)  # raises loudly if this isn't UNSW-NB15's 10 classes
    print("[preprocess] label_map matches confirmed UNSW-NB15 encoding (Part 1, 1.2). OK.")

    y_train_full = label_encoder.transform(train_df["attack_cat"]).astype(np.int64)
    y_test = label_encoder.transform(test_df["attack_cat"]).astype(np.int64)

    # --- Train/validation split (carved from training-set.csv only) -----
    # No official UNSW-NB15 validation file exists, so we carve one out of
    # the training portion here, stratified by attack_cat so every class
    # (including rare ones like Worms) is represented in both splits.
    # test_df / y_test are NEVER touched by this split.
    assert 0 < args.val_fraction < 1.0, "--val_fraction must be in (0, 1)"
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=y_train_full,
    )
    train_df_final = train_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    y_train = y_train_full[train_idx]
    y_val = y_train_full[val_idx]
    print(f"[preprocess] train/val split: {len(train_df_final)} train rows, "
          f"{len(val_df)} val rows (val_fraction={args.val_fraction}, "
          f"stratified by attack_cat, seed={args.seed})")

    # --- Feature engineering --------------------------------------------
    # Scaler/encoder are fit ONLY on train_df_final (post-carve), so neither
    # the validation subset nor the test set leak their statistics into
    # preprocessing -- they are treated as held-out from this point on.
    numeric_cols, categorical_cols = split_feature_columns(train_df_final)
    print(f"[preprocess] {len(numeric_cols)} numeric cols, "
          f"{len(categorical_cols)} categorical cols: {categorical_cols}")

    scaler = MinMaxScaler()
    x_train_num = scaler.fit_transform(train_df_final[numeric_cols].astype(np.float64))
    x_val_num = scaler.transform(val_df[numeric_cols].astype(np.float64))
    x_test_num = scaler.transform(test_df[numeric_cols].astype(np.float64))

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_train_cat = encoder.fit_transform(train_df_final[categorical_cols].astype(str))
    x_val_cat = encoder.transform(val_df[categorical_cols].astype(str))
    x_test_cat = encoder.transform(test_df[categorical_cols].astype(str))

    x_train = np.concatenate([x_train_num, x_train_cat], axis=1).astype(np.float32)
    x_val = np.concatenate([x_val_num, x_val_cat], axis=1).astype(np.float32)
    x_test = np.concatenate([x_test_num, x_test_cat], axis=1).astype(np.float32)

    feature_names = list(numeric_cols) + list(encoder.get_feature_names_out(categorical_cols))

    print(f"[preprocess] final feature dimension: {x_train.shape[1]} "
          f"(Part 1 1.4 backbone expects 196)")
    if x_train.shape[1] != 196:
        print(
            "[preprocess] WARNING: produced feature dimension "
            f"{x_train.shape[1]} != 196. This depends on exactly which "
            "UNSW-NB15 release/columns you used. Either (a) update "
            "model.input_dim in config.yaml to match the real value above, "
            "or (b) adjust feature engineering until it matches 196 if you "
            "need to keep the architecture in section 1.4 byte-for-byte as "
            "specified. Not auto-correcting this silently.",
            file=sys.stderr,
        )

    # --- Persist ----------------------------------------------------------
    label_map_path = os.path.join(args.output_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)

    with open(os.path.join(args.output_dir, "feature_columns.json"), "w") as f:
        json.dump({
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "final_feature_names": feature_names,
            "final_feature_dim": int(x_train.shape[1]),
        }, f, indent=2)

    np.savez_compressed(os.path.join(args.output_dir, "train.npz"), X=x_train, y=y_train)
    np.savez_compressed(os.path.join(args.output_dir, "val.npz"), X=x_val, y=y_val)
    np.savez_compressed(os.path.join(args.output_dir, "test.npz"), X=x_test, y=y_test)

    print(f"[preprocess] wrote label_map.json -> {label_map_path}")
    print(f"[preprocess] wrote train.npz ({x_train.shape}) / val.npz ({x_val.shape}) / "
          f"test.npz ({x_test.shape}) to {args.output_dir}")
    print("[preprocess] done.")


if __name__ == "__main__":
    main()