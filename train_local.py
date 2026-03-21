"""
train_local.py — FIFA 22 Player Value Prediction
=================================================
รัน script นี้บนเครื่องของเพื่อนเพื่อเทรนโมเดลใหม่
และ save .pkl ด้วย sklearn version เดียวกับที่ติดตั้งอยู่

วิธีใช้:
    1. วางไฟล์นี้ใน folder เดียวกับ players_22.csv
    2. เปิด CMD แล้วรัน:  python train_local.py
    3. ได้ไฟล์: player_value_model.pkl, model_metadata.json, requirements.txt
"""

import sys, warnings
import pandas as pd
import numpy as np
import json
import joblib
import sklearn
warnings.filterwarnings("ignore")

print(f"Python  version: {sys.version.split()[0]}")
print(f"sklearn version: {sklearn.__version__}")
print("-" * 45)

# ─────────────────────────────────────────────
# 1. โหลดข้อมูล
# ─────────────────────────────────────────────
CSV_PATH = "players_22.csv"
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

# ─────────────────────────────────────────────
# 2. Feature Engineering & Cleaning
# ─────────────────────────────────────────────

# สร้าง is_gk feature
df["is_gk"] = df["player_positions"].str.contains("GK", na=False).astype(int)

# เติม 0 สำหรับ GK (outfield stats ไม่มีความหมายสำหรับ GK)
outfield_stats = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
df[outfield_stats] = df[outfield_stats].fillna(0)

# Drop แถวที่ target หรือ wage_eur หาย
df = df.dropna(subset=["value_eur", "wage_eur"])

# Drop แถวที่ value_eur = 0 (ไม่มีมูลค่า = ไม่มีข้อมูล)
df = df[df["value_eur"] > 0]

# จัดการ preferred_foot missing
df["preferred_foot"] = df["preferred_foot"].fillna("Right")

print(f"After cleaning: {len(df):,} rows")

# ─────────────────────────────────────────────
# 3. กำหนด Features / Target
# ─────────────────────────────────────────────
numeric_features = [
    "overall", "potential", "age", "height_cm", "weight_kg",
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys",
    "movement_reactions", "mentality_vision", "mentality_composure",
    "weak_foot", "skill_moves", "international_reputation", "wage_eur",
    "is_gk",
]

categorical_features = ["preferred_foot"]
target = "value_eur"

# ตรวจสอบว่าทุก column มีอยู่จริง
all_features = numeric_features + categorical_features
missing_cols = [c for c in all_features if c not in df.columns]
if missing_cols:
    print(f"⚠️  columns ไม่พบใน CSV: {missing_cols}")
    numeric_features  = [c for c in numeric_features  if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

# Drop แถวที่ features หลักหาย
df_model = df[numeric_features + categorical_features + [target]].dropna()
print(f"Rows used for model: {len(df_model):,}")

X = df_model[numeric_features + categorical_features]
y = np.log1p(df_model[target])

# ─────────────────────────────────────────────
# 4. Train / Test split
# ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────
# 5. Pipeline
# ─────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

numeric_transformer     = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,     numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ─────────────────────────────────────────────
# 6. GridSearchCV
# ─────────────────────────────────────────────
print("\nRunning GridSearchCV (อาจใช้เวลา 3–7 นาที)...")

gb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(random_state=42)),
])

param_grid = {
    "model__n_estimators":    [100, 200],
    "model__max_depth":       [3, 5],
    "model__learning_rate":   [0.05, 0.1],
    "model__min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    gb_pipeline, param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="r2", n_jobs=-1, verbose=1,
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV R²:  {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────────
# 7. Evaluate on Test Set
# ─────────────────────────────────────────────
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred_log  = best_model.predict(X_test)
y_pred      = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

mae  = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2   = r2_score(y_test_orig, y_pred)
mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100

print(f"\nTest R²:   {r2:.4f}")
print(f"Test MAE:  €{mae:,.0f}")
print(f"Test RMSE: €{rmse:,.0f}")
print(f"Test MAPE: {mape:.2f}%")

# ─────────────────────────────────────────────
# 8. Save model + metadata + requirements.txt
# ─────────────────────────────────────────────
joblib.dump(best_model, "player_value_model.pkl")
print("\n✅ Saved: player_value_model.pkl")

metadata = {
    "numeric_features":    numeric_features,
    "categorical_features": categorical_features,
    "target":              target,
    "sklearn_version":     sklearn.__version__,
    "best_params":         grid_search.best_params_,
    "test_metrics": {
        "r2":   round(r2,   4),
        "mae":  round(mae,  0),
        "rmse": round(rmse, 0),
        "mape": round(mape, 2),
    },
}

with open("model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print("✅ Saved: model_metadata.json")

req = f"""streamlit>=1.28.0
scikit-learn=={sklearn.__version__}
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
"""
with open("requirements.txt", "w") as f:
    f.write(req)
print(f"✅ Saved: requirements.txt  (pinned scikit-learn=={sklearn.__version__})")

print("\n" + "=" * 45)
print("🎉 เทรนเสร็จแล้ว! ไฟล์ที่ได้:")
print("   player_value_model.pkl")
print("   model_metadata.json")
print("   requirements.txt")
print("\nขั้นตอนต่อไป:")
print("  streamlit run app.py")
