# # --- put these lines at the VERY TOP of main.py (line 1) ---
# import warnings
#
# from utils.save_model import save_model_json
#
# # Generic: silence that urllib3 LibreSSL warning
# warnings.filterwarnings("ignore", category=Warning, module=r"urllib3(\.|$)")
# # Specific: silence the pkg_resources deprecation warning
# warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
#
# from models.linear.siamese_trainable import SiameseModelTrainable
# import random
# import numpy as np
# from torch.utils.data import DataLoader
# from data.dataset import KeypointsPairDataset
# from training.train_model import TrainModel
#
# def _silence_warnings_in_worker(_):
#     import warnings
#     from urllib3.exceptions import NotOpenSSLWarning
#     warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
#     warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
#
#
# def make_batches(pairs, labels, batch_size=64):
#     # pairs: list of (x1:list[float], x2:list[float]) with len == input_dim
#     # labels: list[int] 0/1
#     idx = list(range(len(pairs)))
#     random.shuffle(idx)
#     for s in range(0, len(idx), batch_size):
#         sl = idx[s:s+batch_size]
#         x1 = [pairs[i][0] for i in sl]
#         x2 = [pairs[i][1] for i in sl]
#         y  = [labels[i]    for i in sl]
#         yield x1, x2, y
#
#
#
# def coerce_pairs_numpy(pairs, labels, expected_dim=51):
#     clean_pairs = []
#     for a, b in pairs:
#         # Convert to NumPy arrays and flatten
#         fa = np.array(a, dtype=float).reshape(-1)
#         fb = np.array(b, dtype=float).reshape(-1)
#         if expected_dim is not None:
#             if fa.shape[0] != expected_dim or fb.shape[0] != expected_dim:
#                 raise ValueError(
#                     f"Expected {expected_dim} features, "
#                     f"got {fa.shape[0]} and {fb.shape[0]}"
#                 )
#         clean_pairs.append((fa.tolist(), fb.tolist()))
#     return clean_pairs, [int(l) for l in labels]
#
# def l2norm_rows(X):
#     out = []
#     for row in X:
#         s = sum(v*v for v in row) ** 0.5
#         out.append([v/s for v in row] if s > 0 else row[:])
#     return out
#
# if __name__ == '__main__':
#     trainer = TrainModel()
#     # trainer.extract_all_keypoints()
#     # trainer.generate_all_embeddings()
#     pairs, labels = trainer.generate_pairs()
#     dataset = KeypointsPairDataset(pairs, labels)
#     loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=_silence_warnings_in_worker)
#
#     # pairs, labels = coerce_pairs_numpy(pairs, labels, expected_dim=51)
#
#     # model
#     # model = SiameseModel(input_dim=51, layers=True, seed=7)
#     model = SiameseModelTrainable(input_dim=51,h=64, embed=32, seed=7)
#
#     epochs = 20
#     lr = 1e-4
#     margin = 1.0
#
#     for epoch in range(epochs):
#         running, steps = 0.0, 0
#         for emb1, emb2, lab in loader:
#             # tensors -> python lists
#             x1 = l2norm_rows(emb1.numpy().astype(float).tolist())
#             x2 = l2norm_rows(emb2.numpy().astype(float).tolist())
#             y = lab.numpy().astype(int).tolist()
#
#             loss = model.train_batch(x1, x2, y, lr=lr, margin=margin)
#             running += loss
#             steps += 1
#
#         print(f"epoch {epoch + 1}: loss ~ {running / max(1, steps):.4f}")
#
#     # torch.save(model.state_dict(), 'models/siamese_model.pth')
#     # print("Training completed. Model saved to 'models/siamese_model.pth'")
#     save_model_json(model, "models/siamese_trainable_epoch20.json",
#                     meta={"input_dim": 51, "h": 64, "embed": 32, "epochs": 20})
#

