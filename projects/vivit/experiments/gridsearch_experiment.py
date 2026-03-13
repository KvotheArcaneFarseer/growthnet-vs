import warnings
warnings.filterwarnings("ignore")

import os
import sys

import argparse
import ast
import torch
from accelerate import Accelerator
from monai.utils import set_determinism

from src.data.temporal_loader import load_temporal_splits_from_json
from src.networks.t_unetr import TemporalUNETR
from src.train.grid_search import grid_search

SEED = 42
N_SPLITS = 3

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="grid search experiments",
        description="Runs a series of grid search experiments"
    )

    parser.add_argument(
        "file_name",
        metavar="N",
        type=str,
        help="Choose the file name"
    )

    parser.add_argument(
        "target_spacing",
        default=(1.0, 1.0, 2.0),
        metavar="S",
        type=ast.literal_eval,
        help="Choose the spatial size to resize to"
    )
    parser.add_argument(
        "img_size",
        default=(128, 128, 64),
        metavar="I",
        type=ast.literal_eval,
        help="Choose the input image size"
    )
    parser.add_argument(
        "pos_neg",
        default=[3.0, 1.0],
        metavar="P",
        type=ast.literal_eval,
        help="Choose the ratio of positive to negative samples"
    )

    parser.add_argument(
        "temporal_depth",
        default=8,
        metavar="TD",
        type=int,
        help="Choose the temporal depth"
    )

    parser.add_argument(
        "initial_lr",
        default=1e-4,
        metavar="LR",
        type=float,
        help="Choose the initial learning rate"
    )

    parser.add_argument(
        "lam",
        default=1.0,
        metavar="LM",
        type=float,
        help="Choose the lambda for CE"
    )

    # Parse the args
    args = parser.parse_args()
    file_name = args.file_name
    target_spacing = args.target_spacing
    img_size = args.img_size
    pos_neg = args.pos_neg
    temporal_depth = args.temporal_depth
    initial_lr = args.initial_lr
    lam = args.lam

    print("Loading data.")

    # Get the data
    root = "/standard/gam_ai_group/new T1_split"
    splits = load_temporal_splits_from_json(root)
    dataset = splits["train"]

    # Grid search parameters
    model_params = {
        "in_channels": [1],
        "out_channels": [1],
        "img_size": [img_size],
        "patch_size": [16],
        "temporal_depth": [temporal_depth],
        "use_temporal_encoder": [None, "mlp", "position", "dual"],
        "aggregation_method": ["last", "mean", "max", "cat"],
        "vit_from_pretrained": ["vit_pretrain_weights_epochs100.pth"]
    }
    batch_sizes = [4]
    train_params = {
        "max_epochs": [20],
        "loss_function": ["tverskyce"],
        "initial_lr": [initial_lr],
        "val_interval": [1],
        "scheduler": ["warmup"],
        "lam": [lam],
        "patience": [15]
    }

    # Set the file path
    fn = (
        f"{file_name}_grid_search_results_{target_spacing}_{img_size}_{pos_neg}.csv"
    )
    file_path = os.path.join(os.getcwd(), fn)

    # Initialize accelerator
    accelerator = Accelerator()

    # Conduct grid search
    df = grid_search(
        model=TemporalUNETR,
        dataset=dataset,
        model_params=model_params,
        train_params=train_params,
        batch_sizes=batch_sizes,
        n_splits=N_SPLITS,
        target_spacing=target_spacing,
        img_size=img_size,
        pos_neg=pos_neg,
        accelerator=accelerator,
        file_path=file_path
    )
    print("Grid search complete.")

if __name__ == "__main__":
    main()