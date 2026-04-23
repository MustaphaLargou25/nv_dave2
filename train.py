"""Train the NVIDIA DAVE-2 behavioral cloning model on Udacity simulator data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dave2.config import DataConfig, PreprocessingConfig, TrainingConfig
from dave2.data import (
    build_behavioral_cloning_dataframe,
    save_dataset_split_manifests,
    split_behavioral_cloning_dataframe,
    summarize_behavioral_cloning_dataframe,
)
from dave2.dataset import build_dataloaders
from dave2.model import build_model, count_parameters, describe_variant, get_dave2_vit_config
from dave2.training import fit, seed_everything, select_device


DEFAULT_CHECKPOINT_PATH = Path("artifacts/best_dave2.pth")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a ViT-based NVIDIA DAVE-2 behavioral cloning network in PyTorch.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        required=True,
        help="Directory that contains driving_log.csv and IMG/.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["mini", "medium", "large"],
        default="mini",
        help="ViT capacity preset for the DAVE-2 steering model.",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="Print the available ViT presets and exit.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--steering-correction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--train-manifest-path",
        type=Path,
        default=Path("artifacts/train_manifest.csv"),
    )
    parser.add_argument(
        "--validation-manifest-path",
        type=Path,
        default=Path("artifacts/validation_manifest.csv"),
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    if args.list_variants:
        for variant_name in ("mini", "medium", "large"):
            print(json.dumps(describe_variant(variant_name), indent=2))
        return

    data_config = DataConfig(
        raw_data_dir=args.raw_data_dir,
        steering_correction=args.steering_correction,
        test_size=args.test_size,
        random_state=args.seed,
    )
    preprocessing_config = PreprocessingConfig()
    checkpoint_path = args.checkpoint_path
    if checkpoint_path == DEFAULT_CHECKPOINT_PATH:
        checkpoint_path = Path(f"artifacts/best_dave2_vit_{args.model_size}.pth")
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        checkpoint_path=checkpoint_path,
    )

    seed_everything(args.seed)

    full_dataframe = build_behavioral_cloning_dataframe(
        raw_data_dir=data_config.raw_data_dir,
        steering_correction=data_config.steering_correction,
    )
    dataset_summary = summarize_behavioral_cloning_dataframe(full_dataframe)
    print("Expanded dataset summary:")
    print(json.dumps(dataset_summary, indent=2))

    train_frame, validation_frame = split_behavioral_cloning_dataframe(
        full_dataframe,
        test_size=data_config.test_size,
        random_state=data_config.random_state,
    )
    print(
        f"Train/validation split: {len(train_frame)} train samples | "
        f"{len(validation_frame)} validation samples"
    )

    save_dataset_split_manifests(
        train_frame=train_frame,
        validation_frame=validation_frame,
        train_output_path=args.train_manifest_path,
        validation_output_path=args.validation_manifest_path,
    )

    train_loader, validation_loader = build_dataloaders(
        train_frame=train_frame,
        validation_frame=validation_frame,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        preprocessing=preprocessing_config,
    )

    device = select_device()
    print(f"Using device: {device}")

    model_config = get_dave2_vit_config(args.model_size)
    model = build_model(variant=args.model_size).to(device)
    print(
        "Model configuration:"
        f" variant={model_config.variant}"
        f", patch_size={model_config.patch_size}"
        f", embed_dim={model_config.embed_dim}"
        f", depth={model_config.depth}"
        f", heads={model_config.num_heads}"
        f", parameters={count_parameters(model):,}"
    )
    result = fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=training_config.epochs,
        learning_rate=training_config.learning_rate,
        checkpoint_path=training_config.checkpoint_path,
        device=device,
    )
    print(
        f"Best validation loss: {result.best_validation_loss:.6f} | "
        f"checkpoint: {result.checkpoint_path}"
    )


if __name__ == "__main__":
    main()
