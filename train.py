"""
Tennis Action Similarity — Unified Training Script

Purpose
-------
Train and calibrate a Siamese-style similarity model using one of several backends:
- "pure": small MLP over keypoint/pose embeddings (NumPy-only model)
- "pose_attn": ViTPose keypoints + body-part/temporal attention (PyTorch)
- "r3d_18": 3D CNN over short video clips (PyTorch)

Key Ideas
---------
1) Train on pairwise samples (left,right,label) where label=0 means "similar", 1 means "dissimilar".
2) Calibrate a decision threshold τ (tau) on the validation set. Two options:
   - τ_acc : threshold maximising accuracy on val
   - τ_fpr : threshold that caps the false-positive rate (e.g., ≤ 5%) for negatives
3) Persist the trained weights AND the calibration (τ, scale) for deployment.

Usage Examples
--------------
Pure (pose embeddings):
    python train.py --backend pure --epochs 50 --batch_size 32 --lr 5e-4 --margin 1.0 --embed_dim 64 --use_bn --use_dropout

Pose + Body-Part Attention:
    python train.py --backend pose_attn --epochs 25 --batch_size 16 --lr 1e-3 --margin 0.75 --embed_dim 128

Video (R3D-18):
    python train.py --backend r3d_18 --videos_root dataset/VIDEO_RGB/forehand_openstands --epochs 10 --batch_size 2 --lr 1e-3 --margin 1.0 --freeze_backbone
"""

import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse, torch
from torch.utils.data import DataLoader, random_split
from data.dataset import build_dataset
from models import create_model
from training.loops import (
    train_epoch_pure, eval_epoch_pure,
    train_epoch_torch, eval_epoch_torch,
)
from training.checkpoints import save_pure_json, save_torch

# -----------------------------
# Warnings
# -----------------------------
os.environ.setdefault("MMENGINE_LOG_LEVEL", "ERROR")
warnings.filterwarnings("ignore", category=Warning, module=r"urllib3(\.|$)")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", message=r"Failed to search registry with scope .*", category=UserWarning)


@dataclass
class CalibrationResult:
    """Holds threshold and visualization scale for the UI."""
    tau: float      # Decision threshold for "similar"
    scale: float    # Display scale for similarity mapping (for the frontend)

# -----------------------------
# Helper Functions
# -----------------------------
def gather_val_distances_and_labels_pure(model, loader):
    distances, labels = [], []
    for left_vec, right_vec, label_tensor in loader:
        left = left_vec.numpy().astype(float).tolist()
        right = right_vec.numpy().astype(float).tolist()
        batch_distances = model.distances(left, right)  # List[float]
        distances.extend(batch_distances)
        labels.extend(label_tensor.numpy().astype(int).tolist())
    return np.array(distances, float), np.array(labels, int)


def best_tau_by_accuracy(distances, labels, steps=200):
    if len(distances) == 0:
        return 1.0, 0.0

    low, high = float(np.percentile(distances, 2)), float(np.percentile(distances, 98))
    best_tau, best_acc = 1.0, 0.0
    for i in range(steps):
        tau = low + i * (high - low) / max(1, steps - 1)
        pred_sim = distances < tau
        acc = float(np.mean((labels == 0) == pred_sim))
        if acc > best_acc:
            best_tau, best_acc = tau, acc
    return best_tau, best_acc


def estimate_scale(distances, labels, tau):
    pos = distances[labels == 0]
    neg = distances[labels == 1]
    if len(pos) == 0 or len(neg) == 0:
        return 0.1

    p5, n95 = np.percentile(pos, 5), np.percentile(neg, 95)
    width = max(1e-6, n95 - p5)
    return float(width / 6.0)


def pick_tau_for_fpr(distances, labels, target_fpr=0.05):
    """
    Pick τ such that at most target_fpr of negatives (lab=1) fall below τ.
    Lab: 0=positive (similar), 1=negative (dissimilar)
    """
    neg = distances[labels == 1]
    if len(neg) == 0:
        return 1.0  # fallback if no negatives
    neg_sorted = np.sort(neg)
    idx = int(np.floor(target_fpr * len(neg_sorted)))
    idx = max(0, min(len(neg_sorted) - 1, idx))
    return float(neg_sorted[idx])


def plot_training_curves(train_losses, val_accs, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_accs, label="Val Acc (τ=margin)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Training Curves")
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()


def plot_distances(dist, lab, tau, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.hist(dist[lab == 0], bins=50, alpha=0.5, label="Positives")
    plt.hist(dist[lab == 1], bins=50, alpha=0.5, label="Negatives")
    plt.axvline(tau, color="red", linestyle="--", label=f"τ={tau:.2f}")
    plt.legend()
    plt.title("Distance Distributions")
    plt.savefig(os.path.join(save_dir, "distances.png"))
    plt.close()


def plot_roc(dist, lab, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(lab, -dist, pos_label=0)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "roc.png"))
    plt.close()


def collect_val_dists_labels_torch(model, loader, device="cpu"):
    model.eval()
    all_distances, all_labels = [], []
    for batch in loader:
        if len(batch) != 3:
            raise RuntimeError("Expected batches as (left, right, label).")
        left, right, labels = batch

        left = left.to(device)
        right = right.to(device)

        output = model(left, right)
        # Accept either a single tensor (B,) or a tuple/list where the last item is the distance.
        distances = output[-1] if isinstance(output, (tuple, list)) else output
        if distances.ndim > 1:
            distances = distances.norm(dim=1)

        all_distances.append(distances.detach().float().cpu())
        all_labels.append(labels.detach().long().cpu())

    all_distances = torch.cat(all_distances).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return all_distances, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pure", "resnet18", "r3d_18", "pose_attn"], required=True)
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--margin", type=float, default=1.0, help="Contrastive loss margin")
    parser.add_argument("--videos_root", type=str, default="dataset/VIDEO_RGB/forehand_openstands")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--embed_dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--use_bn", action="store_true", help="Use BatchNorm layers")
    parser.add_argument("--use_dropout", action="store_true", help="Use Dropout layers")
    args = parser.parse_args()

    # 1) Build pairwise dataset
    if args.backend == "pure":
        from training.train_model import KeypointExtract
        trainer = KeypointExtract()
        pairs, labels = trainer.generate_pairs()
        dataset = build_dataset("pure", pair_list=pairs, pair_labels=labels)

    elif args.backend == "pose_attn":
        from training.train_model import KeypointExtract
        kp = KeypointExtract()
        pairs, labels = kp.generate_pairs()
        dataset = build_dataset("pose_attn", pair_list=pairs, pair_labels=labels)

    elif args.backend == "r3d_18":
        from training.pair_builder import make_video_pairs_from_folders
        pairs, labels = make_video_pairs_from_folders(args.videos_root)
        dataset = build_dataset("r3d_18", pair_list=pairs, pair_labels=labels,
                                clip_length_frames=8, spatial_size=96)
    else:
        raise NotImplementedError("backend must be one of: pure, pose_attn, r3d_18")

    # 2) Train/Val split
    num_val = max(1, int(0.2 * len(dataset)))
    num_train = len(dataset) - num_val
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [num_train, num_val], generator=generator)

    tain_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    # 3) Create model + training loop per backend
    if args.backend == "pure":
        model = create_model("pure", input_dim=51, hidden_dim=128, embed_dim=args.embed_dim, seed=7, use_bn=args.use_bn,
                             use_dropout=args.use_dropout)
        train_losses, val_accuracies = [], []
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch_pure(model, tain_loader, learning_rate=args.lr, margin=args.margin)
            val_acc = eval_epoch_pure(model, val_loader, margin=args.margin)
            print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc@τ(args.margin={args.margin:.2f})={val_acc:.4f}")
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)

        # Calibration
        distances, labels = gather_val_distances_and_labels_pure(model, val_loader)
        tau_acc, acc_at_tau_acc = best_tau_by_accuracy(distances, labels, steps=200)
        tau_fpr = pick_tau_for_fpr(distances, labels, target_fpr=0.05)
        chose_tau = tau_fpr
        calibrated_val_acc = eval_epoch_pure(model, val_loader, margin=chose_tau)
        scale = estimate_scale(distances, labels, chose_tau)

        # Logging
        print(f"[calibration] tau_acc={tau_acc:.6f} (acc≈{acc_at_tau_acc:.3f}), tau_fpr5={tau_fpr:.6f}")
        print(f"[calibration] USING chose_tau={chose_tau:.6f}, scale={scale:.6f}")
        acc_at_arg_margin = float(((labels == 0) == (distances < args.margin)).mean())
        print(
            f"[val] acc@chose_tau(calibrated)={calibrated_val_acc:.4f} | acc@chose_tau(args.margin={args.margin:.2f})={acc_at_arg_margin:.4f}")

        # 6) Save model + calibration
        os.makedirs("checkpoints", exist_ok=True)
        save_pure_json(model, "checkpoints/pure_siamese.json")
        with open("checkpoints/pure_siamese_calibration.json", "w") as f:
            json.dump({"chose_tau": float(chose_tau), "scale": float(scale)}, f, indent=2)

        # 7) Plots
        save_dir = os.path.join("results", args.backend)
        plot_training_curves(train_losses, val_accuracies, save_dir)
        plot_distances(distances, labels, chose_tau, save_dir)  # note: chose_tau (calibrated), not args.margin
        plot_roc(distances, labels, save_dir)

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.backend == "r3d_18":
            model = create_model("r3d_18", embed_dim=128, use_pretrained=True, freeze_backbone=args.freeze_backbone)
            ckpt_out = "checkpoints/r3d18_siamese.pth"
            calib_name = "r3d18_calibration.json"

        elif args.backend == "pose_attn":
            from models.pose_attn.body_parts import VITPOSE_COCO17
            model = create_model(
                "pose_attn",
                body_parts=VITPOSE_COCO17,
                joint_in_dim=3,  # (x,y,conf)
                joint_hidden=128,
                joint_out_dim=128,
                part_heads=4,
                temporal_layers=2,
                temporal_heads=4,
                emb_dim=args.embed_dim,
                dropout=0.1,
                margin=args.margin,
                return_attn=False,
            )
            ckpt_out = "checkpoints/pose_attn_siamese.pth"
            calib_name = "pose_attn_calibration.json"

        else:
            raise ValueError("Invalid backend")

        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)\

        for epoch_index in range(1, args.epochs + 1):
            train_loss = train_epoch_torch(model, tain_loader, optimizer, margin=args.margin, device=device)
            val_acc = eval_epoch_torch(model, val_loader, margin=args.margin, device=device)
            print(f"epoch {epoch_index:02d} | train_loss {train_loss:.4f} | val_acc@τ={args.margin:.2f} {val_acc:.4f}")

        # --- Calibration on validation set ---
        distances, labels = collect_val_dists_labels_torch(model, val_loader, device=device)
        tau_acc, acc_at_tau_acc = best_tau_by_accuracy(distances, labels, steps=200)
        tau_fpr = pick_tau_for_fpr(distances, labels, target_fpr=0.05)
        chosen_tau = tau_acc if (not np.isfinite(tau_fpr) or tau_fpr <= 1e-6) else tau_fpr
        scale = estimate_scale(distances, labels, chosen_tau)

        print(
            f"[calibration] τ_acc={tau_acc:.4f} (acc≈{acc_at_tau_acc:.3f}), τ_fpr5={tau_fpr:.4f} → using τ={chosen_tau:.4f}, scale={scale:.4f}")

        # Save calibration JSON (name matches what your backend expects)
        # Persist weights + calibration
        os.makedirs("checkpoints", exist_ok=True)
        save_torch(model, ckpt_out)
        with open(os.path.join("checkpoints", calib_name), "w") as f:
            json.dump(CalibrationResult(tau=float(chosen_tau), scale=float(scale)).__dict__, f, indent=2)


if __name__ == "__main__":
    main()