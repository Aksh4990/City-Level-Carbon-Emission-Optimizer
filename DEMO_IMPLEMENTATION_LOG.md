# Demo implementation log

Changelog for the **CarbonAI** dashboard hardening (single source of truth, honest labels, GA trace). *Session: April 3, 2026.*

## Summary

The Streamlit app now drives **Overview, Forecast, Comparison, and executive metrics** from the same computed 6‑month LSTM fan and RF baseline trajectory stored in `st.session_state`, instead of hardcoded June values and percentages. The sector pie and correlation bar use the **latest CSV row** and **full-series correlations**. The genetic algorithm writes a **real convergence series** to `models/ga_convergence.json` for the Optimization tab.

---

## 1. `dashboard/app.py`

| Change | Why |
|--------|-----|
| `ROOT` + shared `compute_lstm_forecast()` / `compute_rf_horizon_forecast()` | One code path for LSTM and RF horizons; avoids duplicated logic between tabs. |
| Session keys `fc_lstm`, `fc_rf`, `fc_months` | Forecasts computed once per session; **Recompute** button clears them after retrain/data edits. |
| Executive snapshot strip + expander “About this demo” | Sets expectations: synthetic data, LSTM vs RF roles, GA optimizes RF only, convergence file source. |
| Overview metrics | 12‑month trend and Jun LSTM figures come from **data + models**; removed fixed `+8.2%` / `2.181 Mt`. |
| Sector pie | Uses `transport_emission`, `industry_emission`, `energy_emission` from latest row (fallback if invalid). |
| Correlation bar | Pearson vs `total_emission` for temperature, transport, industry, renewable over the full dataframe. |
| Forecast tab | Always shows chart + table; **LSTM + RF** on same plot; band renamed **Illustrative ±0.06 Mt**; **Download forecast CSV**. |
| Optimization tab | Plots **`ga_convergence.json`** when present; copy mentions **RF evaluations** and **bounded levers**; placeholder curve only if JSON missing. |
| Comparison tab | “No action” path = **LSTM fan**; June numbers and table rows match `no_action_jun_mt`; savings = LSTM Jun − optimized Mt. |
| Removed unused imports | `plotly.express`, `MinMaxScaler`. |

---

## 2. `models/ga_optimizer.py`

| Change | Why |
|--------|-----|
| `convergence_history` appended each generation after `hof.update` | Records true **best-so-far RF emission (Mt)** per generation. |
| Write `models/ga_convergence.json` | Dashboard plots this instead of a synthetic exponential + noise. |
| Extra fields in `ga_result.json` | `population_size`, `generations`, `fitness_evaluations` for accurate copy (`8,000` = 100×80). |

---

## 3. Artefacts (regenerated locally)

After `pip install deap` (if needed), from repo root:

`python models/ga_optimizer.py`

Produces / updates:

- `models/ga_result.json`
- `models/ga_convergence.json` (80 points)

---

## 4. How to demo

1. From **`SB_Project` root**: `streamlit run dashboard/app.py` (so `models/` and `data/` resolve).
2. Open **Overview** → check executive snapshot matches **Forecast** / **Comparison**.
3. **Forecast** → show LSTM vs RF and CSV download.
4. **Optimization** → show **recorded** convergence if `ga_convergence.json` exists.
5. Expand **About this demo** if asked about data realism or why GA does not use LSTM.

---

## 5. Optional follow-ups (not done here)

- Log per-generation convergence without re-running full GA (resume / checkpoint).
- Joint GA objective using LSTM horizon (heavier, needs a clear fitness definition).
- Model paths relative to `ROOT` so `streamlit run` works from any cwd.
