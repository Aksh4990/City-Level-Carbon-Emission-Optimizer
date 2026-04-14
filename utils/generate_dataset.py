import numpy as np
import pandas as pd
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
DATA_PATH = os.path.join(DATA_DIR, "carbon_emission_data.csv")

np.random.seed(42)
months = 60  # 5 years of monthly data (Jan 2020 – Dec 2024)

dates = pd.date_range(start="2020-01-01", periods=months, freq="MS")

temperature = 25 + 10 * np.sin(2 * np.pi * np.arange(months) / 12) + np.random.normal(0, 1.5, months)

gdp_index = 100 + np.linspace(0, 20, months) + np.random.normal(0, 2, months)

transport_activity = (
    60 + 0.2 * np.arange(months)
    + 5 * np.sin(2 * np.pi * np.arange(months) / 12)
    + np.random.normal(0, 3, months)
)

industry_activity = (
    55 + 0.15 * np.arange(months)
    + np.random.normal(0, 4, months)
)

renewable_share = 15 + 0.3 * np.arange(months) + np.random.normal(0, 2, months)
renewable_share = np.clip(renewable_share, 5, 60)

emission = (
    0.012 * transport_activity
    + 0.010 * industry_activity
    + 0.004 * temperature
    - 0.008 * renewable_share
    + 0.002 * gdp_index
    + np.random.normal(0, 0.05, months)
    + 0.5
)

transport_emission = 0.40 * emission + np.random.normal(0, 0.02, months)
industry_emission  = 0.35 * emission + np.random.normal(0, 0.02, months)
energy_emission    = emission - transport_emission - industry_emission

df = pd.DataFrame({
    "date":                dates,
    "month_index":         np.arange(months),
    "temperature_c":       temperature.round(2),
    "gdp_index":           gdp_index.round(2),
    "transport_activity":  transport_activity.round(2),
    "industry_activity":   industry_activity.round(2),
    "renewable_share_pct": renewable_share.round(2),
    "transport_emission":  transport_emission.round(4),
    "industry_emission":   industry_emission.round(4),
    "energy_emission":     energy_emission.round(4),
    "total_emission":      emission.round(4),
})

os.makedirs(DATA_DIR, exist_ok=True)
df.to_csv(DATA_PATH, index=False)

print(f"Dataset saved to {DATA_PATH}")
print(f"Shape: {df.shape}")
print(f"Emission range: {emission.min():.3f} – {emission.max():.3f} million tons")
print(f"Renewable share: {renewable_share.min():.1f}% – {renewable_share.max():.1f}%")
print("\nFirst 5 rows:")
print(df.head())
print("\nCorrelation with total_emission:")
corr_cols = ["temperature_c", "transport_activity", "industry_activity",
             "renewable_share_pct", "total_emission"]
print(df[corr_cols].corr()["total_emission"].round(3))
