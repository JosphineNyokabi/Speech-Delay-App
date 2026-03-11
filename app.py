import os
import sys
import pickle
import traceback
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")

sys.path.insert(0, BASE_DIR)

try:
    with open(os.path.join(ARTIFACTS_DIR, "trained_model.pkl"),  "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "fitted_imputer.pkl"), "rb") as f:
        imputer = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "feature_names.pkl"),  "rb") as f:
        feature_names = pickle.load(f)
    print("✓ Model artifacts loaded successfully")
except FileNotFoundError as e:
    print(f"ERROR: Model artifact not found — {e}")
    print("Run Model_Project.py first to generate the pkl files in model_artifacts/")
    model         = None
    imputer       = None
    feature_names = None

# Import the prediction function from the model script
try:
    from Model_Project import predict_new_child
    print("✓ predict_new_child imported successfully")
except ImportError as e:
    print(f"ERROR: Could not import predict_new_child — {e}")
    predict_new_child = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the caregiver questionnaire form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Check model is loaded
    if model is None or imputer is None or feature_names is None:
        return jsonify({
            "error": "Model artifacts not loaded. Run Model_Project.py first."
        }), 500

    if predict_new_child is None:
        return jsonify({
            "error": "predict_new_child function could not be imported."
        }), 500

    # Parse incoming JSON
    try:
        child_data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": f"Could not parse request body: {str(e)}"}), 400

    if not child_data:
        return jsonify({"error": "Empty request body received."}), 400

    # Run prediction
    try:
        result = predict_new_child(model, imputer, feature_names, child_data)
    except Exception as e:
        print("Prediction error:")
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Convert any numpy types to plain Python so jsonify works
    def make_serialisable(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    result_clean = make_serialisable(result)

    return jsonify(result_clean)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)