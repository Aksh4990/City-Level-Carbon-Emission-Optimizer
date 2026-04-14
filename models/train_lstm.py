import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    "total_emission",   # included so LSTM sees past emission values too
]

data = df[FEATURES].values  # shape: (60, 7)

# ── Normalize ──────────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
data_sc = scaler.fit_transform(data)

# ── Create sliding window sequences ───────────────────────────────────────────
# Input : last SEQ_LEN months of all features
# Target: next month's total_emission (last column)
SEQ_LEN = 12

X_seq, y_seq = [], []
for i in range(SEQ_LEN, len(data_sc)):
    X_seq.append(data_sc[i - SEQ_LEN:i])        # (12, 7)
    y_seq.append(data_sc[i, -1])                 # total_emission index = 6

X_seq = np.array(X_seq)   # (48, 12, 7)
y_seq = np.array(y_seq)   # (48,)

# ── Train / test split (80/20, time order preserved) ──────────────────────────
split = int(len(X_seq) * 0.8)
X_train = torch.tensor(X_seq[:split],  dtype=torch.float32)
y_train = torch.tensor(y_seq[:split],  dtype=torch.float32)
X_test  = torch.tensor(X_seq[split:],  dtype=torch.float32)
y_test  = torch.tensor(y_seq[split:],  dtype=torch.float32)

# ── LSTM Model ─────────────────────────────────────────────────────────────────
class EmissionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()   # last time step → scalar

INPUT_SIZE  = len(FEATURES)   # 7
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
EPOCHS      = 150
LR          = 0.001

model     = EmissionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── Training loop ──────────────────────────────────────────────────────────────
print("Training LSTM...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    pred  = model(X_train)
    loss  = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 25 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test)
            val_loss = criterion(val_pred, y_test)
        print(f"  Epoch {epoch:>3}/{EPOCHS}  |  Train Loss: {loss.item():.5f}  |  Val Loss: {val_loss.item():.5f}")

# ── Evaluate ───────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred_sc = model(X_test).numpy()

# Inverse-transform predictions back to original scale
# We need to reconstruct a full row to inverse_transform properly
def inverse_emission(scaled_vals):
    dummy = np.zeros((len(scaled_vals), len(FEATURES)))
    dummy[:, -1] = scaled_vals
    return scaler.inverse_transform(dummy)[:, -1]

y_pred_real = inverse_emission(y_pred_sc)
y_test_real = inverse_emission(y_test.numpy())

mae  = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
r2   = r2_score(y_test_real, y_pred_real)

print("\n" + "=" * 40)
print("       LSTM Forecast Results")
print("=" * 40)
print(f"  MAE  : {mae:.4f} million tons")
print(f"  RMSE : {rmse:.4f} million tons")
print(f"  R²   : {r2:.4f}")
print("=" * 40)

print("\nSample predictions vs actual:")
print(f"{'Actual':>10}  {'Predicted':>10}  {'Error':>10}")
for actual, predicted in zip(y_test_real, y_pred_real):
    print(f"{actual:>10.4f}  {predicted:>10.4f}  {abs(actual-predicted):>10.4f}")

# ── Forecast next 6 months ─────────────────────────────────────────────────────
print("\nForecasting next 6 months (Jan–Jun 2026)...")

last_sequence = data_sc[-SEQ_LEN:].copy()   # last 12 months
future_preds  = []

model.eval()
with torch.no_grad():
    seq = last_sequence.copy()
    for step in range(6):
        inp  = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32)
        pred = model(inp).item()
        future_preds.append(pred)

        # Roll the window: drop oldest, append new row
        new_row        = seq[-1].copy()
        new_row[-1]    = pred                  # update emission with prediction
        new_row[0]    += 1                     # increment month_index
        seq            = np.vstack([seq[1:], new_row])

future_real = inverse_emission(np.array(future_preds))

future_months = ["Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026", "May 2026", "Jun 2026"]
print(f"\n{'Month':<12}  {'Predicted Emission':>20}")
print("-" * 36)
for month, val in zip(future_months, future_real):
    print(f"{month:<12}  {val:>18.4f} M tons")

# ── Save model and scaler ──────────────────────────────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "lstm_model.pth"))
joblib.dump(scaler,            os.path.join(MODELS_DIR, "lstm_scaler.pkl"))
joblib.dump(FEATURES,          os.path.join(MODELS_DIR, "lstm_features.pkl"))
print(f"\nModel saved to {os.path.join(MODELS_DIR, 'lstm_model.pth')}")
print(f"Scaler saved to {os.path.join(MODELS_DIR, 'lstm_scaler.pkl')}")
