import os, tempfile, traceback, logging, warnings
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from registry import build_backend
from webapp.config import CLEAN_CORS

# Quiet noisy libs (optional)
os.environ.setdefault("MMENGINE_LOG_LEVEL", "ERROR")
logging.basicConfig(level=logging.ERROR, force=True)
for name in ["mmengine","mmengine.fileio","mmcv","mmpose","mmdet"]:
    logging.getLogger(name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"Failed to search registry with scope .*")

app = Flask(__name__)
CORS(app, origins=CLEAN_CORS)

def _save_temp(upload, suffix):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    upload.save(f.name); return f.name

@app.route("/analyse", methods=["POST"])
def analyse():
    """
    Multipart form:
      - sample: file
      - ref:    file (optional for some backends)
      - backend: pure | pose_attn | r3d18 (default: pure)
    Response JSON:

    """
    sample_path = None
    ref_path = None
    try:
        backend_name = (request.form.get("backend") or request.args.get("backend") or "pose_attn").lower()
        if "sample" not in request.files:
            return jsonify({"error":"Missing 'sample' file"}), 400
        sample_path = _save_temp(request.files["sample"], ".mp4")

        ref_path = None
        if "ref" in request.files:
            ref_path = _save_temp(request.files["ref"], ".mp4")

        print("Backend Name: ", backend_name)
        backend = build_backend(backend_name)
        data = backend.preprocess(sample_path, ref_path or sample_path)  # pass sample when ref unused
        out  = backend.infer(data)

        resp = {
            "backend": backend_name,
            "distance": out["distance"],
            "similarity_score": out["similarity_score"],
            "is_similar": out["is_similar"],
            "extras": out.get("extras", {})
        }
        return jsonify(resp), 200
    except Exception as e:
        print("ERROR:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        # clean-up temps
        try:
            if 'sample_path' in locals() and Path(sample_path).exists(): os.remove(sample_path)
            if 'ref_path' in locals() and ref_path and Path(ref_path).exists(): os.remove(ref_path)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
