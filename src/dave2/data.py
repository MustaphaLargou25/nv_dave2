"""Phase 1: Udacity simulator ingestion and dataset engineering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover - fallback used only when sklearn is unavailable.
    train_test_split = None


EXPECTED_COLUMNS = [
    "center",
    "left",
    "right",
    "steering",
    "throttle",
    "reverse",
    "speed",
]


def _clean_path(raw_path: str) -> str:
    return str(raw_path).strip().strip('"').replace("\\", "/")


def resolve_image_path(raw_data_dir: Path, image_reference: str) -> Path:
    """Resolve simulator image references into absolute filesystem paths."""

    cleaned = _clean_path(image_reference)
    candidate_path = Path(cleaned)
    candidates = []

    if candidate_path.is_absolute():
        candidates.append(candidate_path)

    candidates.append(raw_data_dir / cleaned)
    candidates.append(raw_data_dir / "IMG" / candidate_path.name)
    candidates.append(raw_data_dir / candidate_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Unable to resolve image path '{image_reference}' under '{raw_data_dir}'."
    )


def load_driving_log(raw_data_dir: Path) -> pd.DataFrame:
    """Load and validate the Udacity `driving_log.csv` manifest."""

    csv_path = raw_data_dir / "driving_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find driving log at '{csv_path}'.")

    frame = pd.read_csv(csv_path)
    frame.columns = [column.strip().lower() for column in frame.columns]

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "driving_log.csv is missing required columns: "
            + ", ".join(missing_columns)
        )

    frame = frame[EXPECTED_COLUMNS].copy()
    numeric_columns = ["steering", "throttle", "reverse", "speed"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def build_behavioral_cloning_dataframe(
    raw_data_dir: Path,
    steering_correction: float = 0.2,
) -> pd.DataFrame:
    """Expand center, left, and right images into a single training dataframe."""

    driving_log = load_driving_log(raw_data_dir)
    camera_offsets = {
        "center": 0.0,
        "left": steering_correction,
        "right": -steering_correction,
    }

    records: list[dict[str, Any]] = []
    for row in driving_log.itertuples(index=False):
        for camera, offset in camera_offsets.items():
            source_path = getattr(row, camera)
            adjusted_steering = float(row.steering) + offset
            records.append(
                {
                    "image_path": str(resolve_image_path(raw_data_dir, source_path)),
                    "camera": camera,
                    "steering": adjusted_steering,
                    "base_steering": float(row.steering),
                    "steering_offset": offset,
                    "throttle": float(row.throttle),
                    "reverse": float(row.reverse),
                    "speed": float(row.speed),
                }
            )

    dataframe = pd.DataFrame.from_records(records)
    if dataframe.empty:
        raise ValueError("No samples were generated from driving_log.csv.")

    return dataframe


def _fallback_train_test_split(
    dataframe: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure-pandas split used only when scikit-learn is unavailable."""

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(dataframe))
    test_count = int(round(len(indices) * test_size))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    train_frame = dataframe.iloc[train_indices].reset_index(drop=True)
    test_frame = dataframe.iloc[test_indices].reset_index(drop=True)
    return train_frame, test_frame


def split_behavioral_cloning_dataframe(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform the Phase 1 80/20 train/validation split."""

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    if train_test_split is None:
        return _fallback_train_test_split(dataframe, test_size, random_state)

    train_frame, test_frame = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def summarize_behavioral_cloning_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return compact EDA statistics for the expanded behavioral cloning dataset."""

    return {
        "num_samples": int(len(dataframe)),
        "num_unique_images": int(dataframe["image_path"].nunique()),
        "camera_counts": dataframe["camera"].value_counts().to_dict(),
        "missing_values": dataframe.isna().sum().to_dict(),
        "steering": {
            "mean": float(dataframe["steering"].mean()),
            "std": float(dataframe["steering"].std()),
            "min": float(dataframe["steering"].min()),
            "max": float(dataframe["steering"].max()),
        },
    }


def save_dataset_split_manifests(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    train_output_path: Path,
    validation_output_path: Path,
) -> None:
    """Persist split manifests for reproducibility and debugging."""

    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_output_path.parent.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(train_output_path, index=False)
    validation_frame.to_csv(validation_output_path, index=False)

