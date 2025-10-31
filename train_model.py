# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import json

DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Read all CSVs in data/
csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
if not csv_files:
    raise SystemExit(f"No CSV files found in {DATA_DIR}. Put your CSV files there and run this script.")

print("Found CSV files:", csv_files)

dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f, low_memory=False)
        df["__source_file__"] = os.path.basename(f)
        dfs.append(df)
    except Exception as e:
        print(f"Failed to read {f}: {e}")

if not dfs:
    raise SystemExit("No dataframes loaded.")

df = pd.concat(dfs, ignore_index=True, sort=False)
print("Combined dataframe shape:", df.shape)
print("Columns:", df.columns.tolist()[:50])

# Attempt to find a production numeric column
possible_targets = [
    "Production", "production", "PRODUCTION", "production_tonnes",
    "Production (Tonnes)", "Yield", "yield", "Yield (kg/ha)"
]
target = None
for t in possible_targets:
    if t in df.columns and pd.api.types.is_numeric_dtype(df[t]):
        target = t
        break

# If not found, try any numeric column other than year/area
if target is None:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Heuristic: choose the numeric column with highest variance and not 'Year' if present
    if numeric_cols:
        candidates = [c for c in numeric_cols if 'year' not in c.lower()]
        if candidates:
            target = max(candidates, key=lambda c: df[c].var() if df[c].notnull().sum()>1 else -1)
        else:
            target = numeric_cols[0]

if target is None:
    raise SystemExit("Could not find any numeric column to predict. Please ensure your CSVs have a numeric production/yield column.")

print("Using target column:", target)

# Select features: choose common categorical features if present:
candidate_features = ["state", "State", "crop", "Crop", "Variety", "variety", "year", "Year", "area", "Area", "season"]
# Keep whichever exist
features = []
for c in candidate_features:
    if c in df.columns:
        features.append(c)
# If still not enough features, use first few columns excluding target
if not features:
    features = [c for c in df.columns if c != target][:4]

# Normalize column names (lowercase) mapping
col_map = {c: c for c in df.columns}
# Use features actually present
features = [f for f in features if f in df.columns]
print("Features used:", features)

# Drop rows where target is missing
df = df.dropna(subset=[target])
print("After dropping missing target:", df.shape)

# Basic preprocessing
X = df[features].copy()
y = df[target].astype(float).copy()

# Fill numeric missing
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna(X[col].median())

# Label encode categorical features
encoders = {}
X_enc = pd.DataFrame(index=X.index)
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X_enc[col] = X[col].astype(float)
    else:
        le = LabelEncoder()
        vals = X[col].astype(str).fillna("NA")
        X_enc[col] = le.fit_transform(vals)
        encoders[col] = list(le.classes_)
        print(f"Encoded {col}: {len(encoders[col])} classes")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.15, random_state=42)
print("Train size:", X_train.shape, "Test size:", X_test.shape)

# Model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Model R^2 on test set: {score:.4f}")

# Save model and encoders and metadata
dump(model, os.path.join(MODELS_DIR, "model.pkl"))
meta = {
    "target_column": target,
    "features": list(X_enc.columns),
    "encoders": encoders
}
with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved model to {MODELS_DIR}/model.pkl and metadata to {MODELS_DIR}/meta.json")
print("Training complete.")
