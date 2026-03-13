import warnings
warnings.filterwarnings("ignore")

import torch
import monai
import matplotlib.pyplot as plt
import numpy as np
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism

from src.data.temporal_loader import load_temporal_splits_from_json
from src.data.transforms import build_train_transform, build_eval_transform
from src.data.utils import pad_sequence_collate_fn, TransformSequence

from src.data.samplers import LengthShuffledBucketBatchSampler
from src.networks.t_unetr import TemporalUNETR
from src.train.train_ops import train

def main():
    # Set seed
    seed = 42
    set_determinism(seed)

    # Assemble per-patient sequences
    root = "/standard/gam_ai_group/new T1_split"
    splits = load_temporal_splits_from_json(root)
    X_train, X_val = splits["train"], splits["val"]

    target_spacing = (0.8, 0.8, 2.0)
    img_size = (128, 128, 64)

    # Create transform
    # Build transforms (prefers your utils.py)
    train_tf = build_train_transform(target_spacing, roi_size=img_size, pos_neg=[20.0,1.0])
    eval_tf  = build_eval_transform(target_spacing, img_size)

    # Create MONAI dataset
    transforms = TransformSequence(keys=["images", "labels", "dates"], spatial_transforms=train_tf)
    train_dataset = Dataset(data=X_train, transform=transforms)
    transforms = TransformSequence(keys=["images", "labels", "dates"], spatial_transforms=eval_tf)
    val_dataset = Dataset(data=X_val, transform=transforms)

    batch_size = 4
    train_sampler = LengthShuffledBucketBatchSampler(train_dataset, batch_size, seed=seed)
    val_sampler = LengthShuffledBucketBatchSampler(val_dataset, batch_size, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=pad_sequence_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=pad_sequence_collate_fn,
    )

    net = TemporalUNETR(
        in_channels=1,
        out_channels=1,
        img_size=img_size,
        patch_size=16,
        temporal_depth=4,
        use_temporal_encoder="position",
        aggregation_method="mean"
    )

    net, _ = train(
        net, 
        train_loader, 
        val_loader, 
        max_epochs=200, 
        initial_lr=1e-4, 
        loss_function="dicefocal",
        scheduler="onecycle",
        lam=1.0, 
        patience=20, 
        log_file="epochs200_td4_position_lr1e4_dicefocal_onecycle_lam10_128x128x64_target08"
    )

    path = "epochs200_td4_position_lr1e4_dicefocal_onecycle_lam10_128x128x64_target08_embedding.pth"
    net.save_embedding(path)

main()