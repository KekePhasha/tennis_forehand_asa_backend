import os
import tempfile

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from embedding.pose_embedding import PoseEmbedding
from models.siamese_model import SiameseModel
from pose_estimation.vitpose_extractor import ViTPoseEstimator

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

pose_estimator = ViTPoseEstimator()
pose_embedding = PoseEmbedding(confidence_threshold=0.6)
model = SiameseModel()
model.load_state_dict(torch.load("../models/siamese_model.pth", map_location=torch.device('cpu')))
model.eval()


@app.route("/")
def hello():
    return "Hello, World!"


@app.route('/analyse', methods=['POST'])
def analyze():
    """
    Endpoint to analyze uploaded video files.
    :return:
        JSON response with analysis results or error message.
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
        sample_keypoints = pose_estimator.extract_keypoints(sample_path)
        ref_keypoints = pose_estimator.extract_keypoints(ref_path)

        sample_embedding = pose_embedding.generate_embedding(sample_keypoints)
        ref_embedding = pose_embedding.generate_embedding(ref_keypoints)

        sample_tensor = torch.from_numpy(sample_embedding).unsqueeze(0).float()
        ref_tensor = torch.from_numpy(ref_embedding).unsqueeze(0).float()

        with torch.no_grad():
            distance = model.forward(sample_tensor, ref_tensor).item()
            similarity_score = 1 / (1 + distance)

        return jsonify({
            'similarity_score': similarity_score,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(sample_path)
        os.remove(ref_path)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
