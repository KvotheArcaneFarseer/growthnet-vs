import warnings
warnings.filterwarnings("ignore")

from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
from multiprocessing import cpu_count
import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from src.data.temporal_loader import load_temporal_splits_from_json
from src.data.transforms import build_pretrain_transform, build_preval_transform

from src.data.utils import images_only_dataset
from src.networks.vitautoenc import ViTAutoEnc
from src.train.pretrain_ops import pretrain

def main():
    img_size = (128, 128, 64)

    # Create transform
    train_transform = build_pretrain_transform()
    val_transform = build_preval_transform()

    # Set seed
    seed = 42
    set_determinism(seed)

    # Assemble per-patient sequences
    #root = "/standard/gam_ai_group/new T1_split"
    root = "/scratch/ejf9db/new T1_split"
    splits = load_temporal_splits_from_json(root)
    X_train, X_val, _ = splits["train"], splits["val"], splits["test"]

    # Create MONAI dataset
    train_image_data = images_only_dataset(X_train)
    val_image_data = images_only_dataset(X_val)
    train_dataset = Dataset(data=train_image_data, transform=train_transform)
    val_dataset = Dataset(data=val_image_data, transform=val_transform)

    batch_size = 8
    num_workers = cpu_count() - 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
    )

    net = ViTAutoEnc(
        in_channels=1,
        img_size=img_size,
        patch_size=16,
        dropout_rate=0.1
    )

    net, some_output = pretrain(
        net,
        train_loader,
        val_loader,
        max_epochs=100,
        batch_size=batch_size
    )

    print("Saving network.")
    path = "vit_pretrain_weights_epochs100.pth"
    net.save_vit_weights(path)
    print("Network saved!")

main()