from flask import Flask, request, jsonify, render_template
import pickle
import sys
import os   
# point to the model file
sys.path.insert(0, os.path.dirname(__file__))
from Model_Project import predict_new_child
# creating the flask app
app = Flask(__name__)   
# load the model
print("Loading model...")
with open("model_artifacts/trained_model.pkl",  "rb") as f:
    MODEL = pickle.load(f)
with open("model_artifacts/fitted_imputer.pkl", "rb") as f:
    IMPUTER = pickle.load(f)
with open("model_artifacts/feature_names.pkl",  "rb") as f:
    FEATURES = pickle.load(f)
print("Model ready.")
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    child_data = request.get_json()
    result     = predict_new_child(MODEL, IMPUTER, FEATURES, child_data)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    