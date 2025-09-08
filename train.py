import warnings
# Generic: silence that urllib3 LibreSSL warning
warnings.filterwarnings("ignore", category=Warning, module=r"urllib3(\.|$)")
# Specific: silence the pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
import argparse, torch
from torch.utils.data import DataLoader, random_split
from data.dataset import build_dataset
from models import create_model
from training.loops import (
    train_epoch_pure, eval_epoch_pure,
    train_epoch_torch, eval_epoch_torch,
)
from training.train_model import TrainModel
from training.checkpoints import save_pure_json, save_torch
# If you use torchvision pretrained transforms for ResNet18:
# from torchvision.models import ResNet18_Weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pure", "resnet18", "r3d_18"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()

    # 1) Build your (pairs, labels) here. For example:
    trainer = TrainModel()

    # trainer.extract_all_keypoints()
    # trainer.generate_all_embeddings()
    pairs, labels = trainer.generate_pairs()   # for keypoints


    # For images/videos, build lists of (left_path, right_path) and labels.
    # raise NotImplementedError("Replace this with your actual (pairs, labels) loading")

    # dataset_kind must match your backend
    dataset_kind = args.backend
    # Optional: image_transform = ResNet18_Weights.DEFAULT.transforms()
    dataset = build_dataset(dataset_kind, pair_list=pairs, pair_labels=labels)

    # 2) Split into train/val
    num_val = max(1, int(0.2 * len(dataset)))
    num_train = len(dataset) - num_val
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [num_train, num_val], generator=generator)

    loader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    loader_val   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 3) Create and train the model
    if args.backend == "pure":
        model = create_model("pure", input_dim=51, hidden_dim=64, embed_dim=32, seed=7)
        for epoch_index in range(1, args.epochs + 1):
            train_loss = train_epoch_pure(model, loader_train, learning_rate=args.lr, margin=args.margin)
            val_acc    = eval_epoch_pure(model, loader_val, margin=args.margin)
            print(f"epoch {epoch_index:02d} | train_loss {train_loss:.4f} | val_acc@τ={args.margin:.2f} {val_acc:.4f}")
        save_pure_json(model, "checkpoints/pure_siamese.json")

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.backend == "resnet18":
            model = create_model("resnet18", embed_dim=128, use_pretrained=True, freeze_backbone=args.freeze_backbone)
        else:
            model = create_model("r3d_18",  embed_dim=128, use_pretrained=True, freeze_backbone=args.freeze_backbone)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for epoch_index in range(1, args.epochs + 1):
            train_loss = train_epoch_torch(model, loader_train, optimizer, margin=args.margin, device=device)
            val_acc    = eval_epoch_torch(model, loader_val, margin=args.margin, device=device)
            print(f"epoch {epoch_index:02d} | train_loss {train_loss:.4f} | val_acc@τ={args.margin:.2f} {val_acc:.4f}")

        out_path = "checkpoints/resnet18_siamese.pth" if args.backend == "resnet18" else "checkpoints/r3d18_siamese.pth"
        save_torch(model, out_path)

if __name__ == "__main__":
    main()


#  python train.py --backend pure --epochs 30 --batch_size 128 --lr 5e-4 --margin 1.0