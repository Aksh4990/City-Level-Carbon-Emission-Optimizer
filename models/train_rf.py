import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "carbon_emission_data.csv")
MODELS_DIR = os.path.join(ROOT, "models")

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "month_index",
    "temperature_c",
    "gdp_index",
    "transport_activity",
    "industry_activity",
    "renewable_share_pct",
]
TARGET = "total_emission"

X = df[FEATURES].values
y = df[TARGET].values

# ── Train / test split (80/20, no shuffle — keeps time order) ──────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ── Scale features ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train Random Forest ────────────────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train_sc, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test_sc)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("=" * 40)
print("   Random Forest Baseline Results")
print("=" * 40)
print(f"  MAE  : {mae:.4f} million tons")
print(f"  RMSE : {rmse:.4f} million tons")
print(f"  R²   : {r2:.4f}")
print("=" * 40)

print("\nSample predictions vs actual:")
print(f"{'Actual':>10}  {'Predicted':>10}  {'Error':>10}")
for actual, predicted in zip(y_test[:8], y_pred[:8]):
    print(f"{actual:>10.4f}  {predicted:>10.4f}  {abs(actual-predicted):>10.4f}")

# ── Feature importance ─────────────────────────────────────────────────────────
print("\nFeature importances:")
for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {feat:<25} {imp:.3f}  {bar}")

# ── Save model and scaler ──────────────────────────────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(rf,     os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "rf_scaler.pkl"))
print(f"\nModel saved to {os.path.join(MODELS_DIR, 'rf_model.pkl')}")
print(f"Scaler saved to {os.path.join(MODELS_DIR, 'rf_scaler.pkl')}")
