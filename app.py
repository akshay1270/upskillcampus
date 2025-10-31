# app.py
import os
import json
from flask import Flask, render_template, request, jsonify
import pandas as pd
from joblib import load
import numpy as np

app = Flask(__name__)

MODELS_DIR = "models"
DATA_DIR = "data"

# Load model & metadata
model_path = os.path.join(MODELS_DIR, "model.pkl")
meta_path = os.path.join(MODELS_DIR, "meta.json")

if not os.path.exists(model_path) or not os.path.exists(meta_path):
    raise SystemExit("Model files not found. Run train_model.py first to create models/model.pkl and models/meta.json")

model = load(model_path)
with open(meta_path, "r") as f:
    meta = json.load(f)

FEATURES = meta["features"]
ENCODERS = meta.get("encoders", {})  # mapping col -> classes list
TARGET_COLUMN = meta["target_column"]

# Helper to load raw data for dropdown values
def load_raw_data():
    csvs = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    dfs = []
    for c in csvs:
        try:
            dfs.append(pd.read_csv(c, low_memory=False))
        except:
            pass
    if dfs:
        return pd.concat(dfs, ignore_index=True, sort=False)
    return pd.DataFrame()

RAW_DF = load_raw_data()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_options")
def get_options():
    resp = {}
    # For each feature, try to provide unique sorted list
    for col in FEATURES:
        if col in RAW_DF.columns:
            vals = RAW_DF[col].dropna().astype(str).unique().tolist()
            vals = sorted(vals)[:500]  # cap length
            resp[col] = vals
        else:
            # if encoder classes exist, return them instead
            if col in ENCODERS:
                resp[col] = ENCODERS[col]
            else:
                resp[col] = []
    return jsonify(resp)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    # Build a feature vector in the same order as FEATURES
    x = []
    for col in FEATURES:
        val = data.get(col, "")
        if col in ENCODERS:
            classes = ENCODERS[col]
            try:
                idx = classes.index(str(val))
            except ValueError:
                # if unseen class, append new index at end
                idx = len(classes)
        else:
            # numeric expected
            try:
                idx = float(val)
            except:
                idx = 0.0
        x.append(idx)
    arr = np.array(x).reshape(1, -1)
    pred = model.predict(arr)[0]
    # return predicted value and target column name
    return jsonify({"prediction": float(pred), "target": TARGET_COLUMN})

if __name__ == "__main__":
    app.run(debug=True)
