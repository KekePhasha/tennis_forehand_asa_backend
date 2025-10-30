import json
import os
import warnings
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

os.environ.setdefault("MMENGINE_LOG_LEVEL", "ERROR")

warnings.filterwarnings("ignore", category=Warning, module=r"urllib3(\.|$)")
# Specific: silence the pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", message=r"Failed to search registry with scope .*", category=UserWarning)

# -----------------------------
# Helpers
# -----------------------------
def collect_val_dists_labels_pure(model, loader):
    D, Y = [], []
    for left_vec, right_vec, label_tensor in loader:
        L = left_vec.numpy().astype(float).tolist()
        R = right_vec.numpy().astype(float).tolist()
        d = model.distances(L, R)  # List[float]
        D.extend(d)
        Y.extend(label_tensor.numpy().astype(int).tolist())
    return np.array(D, float), np.array(Y, int)


def choose_best_tau(dist, lab, steps=200):
    if len(dist) == 0:
        return 1.0, 0.0
    lo, hi = float(np.percentile(dist, 2)), float(np.percentile(dist, 98))
    best_tau, best_acc = 1.0, 0.0
    for i in range(steps):
        tau = lo + i * (hi - lo) / max(1, steps - 1)
        pred_sim = dist < tau
        acc = float(np.mean((lab == 0) == pred_sim))
        if acc > best_acc:
            best_tau, best_acc = tau, acc
    return best_tau, best_acc


def estimate_scale(dist, lab, tau):
    pos = dist[lab == 0]
    neg = dist[lab == 1]
    if len(pos) == 0 or len(neg) == 0:
        return 0.1
    p5, n95 = np.percentile(pos, 5), np.percentile(neg, 95)
    width = max(1e-6, n95 - p5)
    return float(width / 6.0)

def pick_tau_for_fpr(dist, lab, target_fpr=0.05):
    """
    Pick τ such that at most target_fpr of negatives (lab=1) fall below τ.
    Lab: 0=positive (similar), 1=negative (dissimilar)
    """
    neg = dist[lab == 1]
    if len(neg) == 0:
        return 1.0  # fallback if no negatives
    neg_sorted = np.sort(neg)
    idx = int(np.floor(target_fpr * len(neg_sorted)))
    idx = max(0, min(len(neg_sorted)-1, idx))
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
    D, Y = [], []
    for batch in loader:
        # Expect (left, right, label) — adapt names if needed
        if len(batch) == 3:
            left, right, lab = batch
        else:
            raise RuntimeError("Expected (left, right, label) from dataset.")

        left  = left.to(device)
        right = right.to(device)

        out = model(left, right)
        # Handle both "dist" or "(dist, (attn1, attn2))"
        dist = out[0] if isinstance(out, (tuple, list)) else out  # shape (B,)
        D.append(dist.detach().float().cpu())
        Y.append(lab.detach().long().cpu())

    D = torch.cat(D).numpy()
    Y = torch.cat(Y).numpy()
    return D, Y

@torch.no_grad()
def debug_label_polarity_pure(model, loader):
    import numpy as np
    D, Y = [], []
    for left_vec, right_vec, label_tensor in loader:
        L = left_vec.numpy().astype(float).tolist()
        R = right_vec.numpy().astype(float).tolist()
        d = model.distances(L, R)  # list of floats
        D.extend(d)
        Y.extend(label_tensor.numpy().astype(int).tolist())
    D = np.asarray(D); Y = np.asarray(Y)
    pos = D[Y == 0].mean() if (Y == 0).any() else float('nan')
    neg = D[Y == 1].mean() if (Y == 1).any() else float('nan')
    print(f"[PURE] mean_pos_dist(0)={pos:.4f}   mean_neg_dist(1)={neg:.4f}")




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

    # 1) Create a dataset of pairs
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


    # 2) Split into train/val
    num_val = max(1, int(0.2 * len(dataset)))
    num_train = len(dataset) - num_val
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [num_train, num_val], generator=generator)

    loader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=False)
    loader_val   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    # 3) Create and train the model
    if args.backend == "pure":
        model = create_model("pure", input_dim=51, hidden_dim=128, embed_dim=args.embed_dim, seed=7,  use_bn=args.use_bn,
        use_dropout=args.use_dropout)
        train_losses, val_accs = [], []
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch_pure(model, loader_train, learning_rate=args.lr, margin=args.margin)
            val_acc    = eval_epoch_pure(model, loader_val, margin=args.margin)
            print(f"epoch {epoch:02d} | train_loss {train_loss:.4f} | val_acc= {val_acc:.4f}")
            debug_label_polarity_pure(model, loader_val)
            train_losses.append(train_loss)
            val_accs.append(val_acc)

        # Calibration
        dist, lab = collect_val_dists_labels_pure(model, loader_val)

        # 1) Pick τ by accuracy and by a target FPR=5%
        tau_acc, acc_acc = choose_best_tau(dist, lab, steps=200)
        tau_fpr = pick_tau_for_fpr(dist, lab, target_fpr=0.05)

        # 2) Choose which τ to use (use the stricter FPR one, or switch to tau_acc)
        tau = tau_fpr  # or: tau = tau_acc

        # 3) Recompute validation accuracy USING the calibrated τ
        val_acc_cal = eval_epoch_pure(model, loader_val, margin=tau)

        # 4) Estimate display scale for your similarity mapping
        scale = estimate_scale(dist, lab, tau)

        # 5) Log everything clearly
        print(f"[calibration] tau_acc={tau_acc:.6f} (acc≈{acc_acc:.3f}), tau_fpr5={tau_fpr:.6f}")
        print(f"[calibration] USING tau={tau:.6f}, scale={scale:.6f}")
        # Optional: also show what accuracy would have been at args.margin for comparison
        acc_at_arg_margin = float(((lab == 0) == (dist < args.margin)).mean())
        print(
            f"[val] acc@tau(calibrated)={val_acc_cal:.4f} | acc@tau(args.margin={args.margin:.2f})={acc_at_arg_margin:.4f}")

        # 6) Save model + calibration
        os.makedirs("checkpoints", exist_ok=True)
        save_pure_json(model, "checkpoints/pure_siamese.json")
        with open("checkpoints/pure_siamese_calibration.json", "w") as f:
            json.dump({"tau": float(tau), "scale": float(scale)}, f, indent=2)

        # 7) Plots (use calibrated τ in the distance histogram)
        save_dir = os.path.join("results", args.backend)
        plot_training_curves(train_losses, val_accs, save_dir)
        plot_distances(dist, lab, tau, save_dir)  # note: tau (calibrated), not args.margin
        plot_roc(dist, lab, save_dir)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.backend == "resnet18":
            model = create_model("resnet18", embed_dim=128, use_pretrained=True, freeze_backbone=args.freeze_backbone)
        elif args.backend == "r3d_18":
            model = create_model("r3d_18",  embed_dim=128, use_pretrained=True, freeze_backbone=args.freeze_backbone)
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
        else:
            raise ValueError("Invalid backend")

        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for epoch_index in range(1, args.epochs + 1):
            train_loss = train_epoch_torch(model, loader_train, optimizer, margin=args.margin, device=device)
            val_acc    = eval_epoch_torch(model, loader_val, margin=args.margin, device=device)
            print(f"epoch {epoch_index:02d} | train_loss {train_loss:.4f} | val_acc@τ={args.margin:.2f} {val_acc:.4f}")

        # --- Calibration on validation set ---
        dist, lab = collect_val_dists_labels_torch(model, loader_val, device=device)

        # Choose τ. You can use accuracy-optimal or FPR-targeted:
        tau_acc, acc = choose_best_tau(dist, lab, steps=200)
        tau_fpr = pick_tau_for_fpr(dist, lab, target_fpr=0.05)
        if not np.isfinite(tau_fpr) or tau_fpr <= 1e-6:
            tau = tau_acc
        else:
            tau = tau_fpr

        scale = estimate_scale(dist, lab, tau)

        print(
            f"[pose_attn calibration] τ_acc={tau_acc:.4f} (acc≈{acc:.3f}), τ_fpr5={tau_fpr:.4f}  → using τ={tau:.4f}, scale={scale:.4f}")

        if args.backend == "resnet18":
            out_path = "checkpoints/resnet18_siamese.pth"
        elif args.backend == "r3d_18":
            out_path = "checkpoints/r3d18_siamese.pth"
        else:
            out_path = "checkpoints/pose_attn_siamese.pth"
        save_torch(model, out_path)

        # Save calibration JSON (name matches what your backend expects)
        calib_name = {
            "pose_attn": "pose_attn_calibration.json",
            "resnet18": "resnet18_calibration.json",
            "r3d_18": "r3d18_calibration.json",
        }[args.backend]
        os.makedirs("checkpoints", exist_ok=True)
        with open(os.path.join("checkpoints", calib_name), "w") as f:
            json.dump({"tau": float(tau), "scale": float(scale)}, f, indent=2)

if __name__ == "__main__":
    main()


"""
Train Pose + Body-Part Attention (expects keypoint sequences)
    python train.py --backend pose_attn --epochs 25 --batch_size 16 --lr 1e-3 --margin 0.75 --embed_dim 128
"""

"""
Train Pure Python Siamese Network with Keypoint Embeddings
    python train.py   --backend pure  --epochs 50  --batch_size 32 --lr 5e-4 --margin 1.0 --embed_dim 64 --use_bn --use_dropout
"""

"""
Train R3D-18 Siamese Network (expects video clips)
    python train.py --backend r3d_18  --videos_root dataset/VIDEO_RGB/forehand_openstands --epochs 10 --batch_size 2 --lr 1e-3 --margin 1.0 --freeze_backbone true
"""
