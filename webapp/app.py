import json
import math
import warnings
from pathlib import Path

# Generic: silence that urllib3 LibreSSL warning
warnings.filterwarnings("ignore", category=Warning, module=r"urllib3(\.|$)")
# Specific: silence the pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import os
import tempfile
import traceback

import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from embedding.pose_embedding import PoseEmbedding
from models import SiameseModelTrainable
from models.siamese_model import SiameseModel
from pose_estimation.vitpose_extractor import ViTPoseEstimator
from training.checkpoints import load_pure_json
from utils.file_utils import FileSaver

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

pose_estimator = ViTPoseEstimator(FileSaver())
pose_embedding = PoseEmbedding(confidence_threshold=0.6)

CKPT_PATH = (
    Path(__file__).resolve()           # .../webapp/app.py
    .parent                            # .../webapp
    .parent / "checkpoints" / "pure_siamese.json"   # .. / checkpoints / pure_siamese.json
).resolve()

CALIB_PATH = (
    Path(__file__).resolve()
    .parent
    .parent / "checkpoints" / "pure_siamese_calibration.json"
).resolve()

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

pure_model = SiameseModelTrainable(input_dim=51, hidden_dim=128, embed_dim=32, seed=7)
load_pure_json(pure_model, str(CKPT_PATH))
print("First linear W:", pure_model.net.layers[0].W[0][:5])
print("First linear b:", pure_model.net.layers[0].b[:5])

print("Second linear W:", pure_model.net.layers[4].W[0][:5])
print("Second linear b:", pure_model.net.layers[4].b[:5])

if CALIB_PATH.exists():
    with open(CALIB_PATH, "r") as f:
        _cal = json.load(f)
    TAU = float(_cal.get("tau", 1.0))
    SCALE = float(_cal.get("scale", 0.1))
else:
    TAU, SCALE = 1.0, 0.1  # fallback; consider logging a warning

def prob_similar(distance: float, tau: float = None, scale: float = None) -> float:
    """Convert an L2 distance to calibrated probability of 'similar'."""
    tau = TAU if tau is None else tau
    scale = SCALE if scale is None else scale
    z = (tau - float(distance)) / max(1e-6, float(scale))
    return 1.0 / (1.0 + math.exp(-z))

def _to_list51(vec: np.ndarray) -> list[list[float]]:
    """Ensure 1x51 float list-of-lists for the pure model API."""
    v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    if v.shape[1] != 51:
        raise ValueError(f"Expected 51-dim embedding, got {v.shape[1]}")
    # (optional) L2-normalize like training
    n = np.linalg.norm(v, axis=1, keepdims=True)
    v = np.divide(v, np.maximum(n, 1e-12))
    return v.tolist()


@app.route('/analyse', methods=['POST'])
def analyze():
    """
    Endpoint to analyze uploaded video files.
    :return: JSON response with analysis results or error message.
    """
    # Check if the request contains the required files
    if 'sample' not in request.files or 'ref' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    sample_file = request.files['sample']
    ref_file = request.files['ref']

    # Create temporary files to save the uploaded videos
    with tempfile.NamedTemporaryFile(delete=False, suffix='.avi') as sample_temp, tempfile.NamedTemporaryFile(
            delete=False, suffix='.mp4') as ref_temp:

        sample_path = sample_temp.name
        ref_path = ref_temp.name
        sample_file.save(sample_path)
        ref_file.save(ref_path)

    try:
        #print statements to debug
        print("Extracting keypoints")
        sample_keypoints, _ = pose_estimator.extract_keypoints(sample_path, save_name='sample')
        ref_keypoints, _ = pose_estimator.extract_keypoints(ref_path, save_name='ref')

        print("Generating embeddings")
        sample_embed = pose_embedding.generate_embedding(sample_keypoints)
        ref_embed = pose_embedding.generate_embedding(ref_keypoints)


        left = _to_list51(sample_embed)
        right = _to_list51(ref_embed)
        distance = pure_model.distances(left, right, train=False)[0]

        print("Sample embed (raw):", sample_embed[:10])  # first 10 values
        print("Ref embed (raw):", ref_embed[:10])
        print("Left forward:", pure_model.forward_once(left)[0][:10])
        print("Right forward:", pure_model.forward_once(right)[0][:10])
        print("Distance:", distance)

        # similarity_score = 1.0 / (1.0 + float(distance))

        similarity_score = prob_similar(distance)  # 0..1
        is_similar = bool(distance < TAU)

        return jsonify({
            'similarity_score': similarity_score,
            'message': 'Comparison successful',
            'is_similar': is_similar,
            'distance': float(distance)
        })
    except Exception as e:
        print("Error occurred:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(sample_path)
        os.remove(ref_path)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
