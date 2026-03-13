import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from accelerate import Accelerator

from monai.utils import set_determinism

from src.data.temporal_loader import load_temporal_splits_from_json

from src.train.grid_search import kfold_cross_val
from src.networks.t_unetr import TemporalUNETR

def main():
    # Set seed
    seed = 42
    set_determinism(seed)

    # Assemble per-patient sequences
    root = "/scratch/ejf9db/new T1_split"
    splits = load_temporal_splits_from_json(root)
    X_train = splits["train"]

    target_spacing = (1.0, 1.0, 2.0)
    img_size = (128, 128, 64)
    pos_neg = [20.0, 1.0]
    batch_size = 4
    model_params = {
        "in_channels": 1,
        "out_channels": 1,
        "img_size": img_size,
        "patch_size": 16,
        "temporal_depth": 4,
        "use_temporal_encoder": "dual",
        "aggregation_method": "cat",
        "vit_from_pretrained": "vit_pretrain_weights_epochs100.pth"
    }
    train_params = {
        "max_epochs": 100,
        "loss_function": "tverskyce",
        "initial_lr": 1e-3,
        "val_interval": 1,
        "scheduler": "warmup",
        "lam": 1.0,
        "patience": 100
    }

    accelerator = Accelerator()

    kf_out = kfold_cross_val(
        model=TemporalUNETR,
        dataset=X_train,
        accelerator=accelerator,
        batch_size=batch_size,
        model_params=model_params,
        train_params=train_params,
        target_spacing=target_spacing,
        img_size=img_size,
        pos_neg=pos_neg,
        n_splits=3,
        random_state=seed
    )

    df = pd.DataFrame(kf_out)

    df.to_csv("cv_test_tv.csv")

if __name__ == "__main__":
    main()