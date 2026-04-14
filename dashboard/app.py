import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import torch
import torch.nn as nn
import plotly.graph_objects as go
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.append(ROOT)

st.set_page_config(
    page_title="CarbonAI — City Emission Control",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background-color: #080C10 !important;
    color: #C8D8E8 !important;
}
.stApp { background: #080C10 !important; }

section[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1C2A3A !important;
}
section[data-testid="stSidebar"] * { color: #C8D8E8 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #0D1117 !important;
    border-bottom: 1px solid #1C2A3A !important;
    gap: 0px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #5A7A9A !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    padding: 12px 24px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #0D1117 !important;
    color: #00E5FF !important;
    border-bottom: 2px solid #00E5FF !important;
}

[data-testid="metric-container"] {
    background: #0D1117 !important;
    border: 1px solid #1C2A3A !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: #5A7A9A !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #E8F4FF !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}

.stButton button {
    background: transparent !important;
    border: 1px solid #00E5FF !important;
    color: #00E5FF !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.1em !important;
    padding: 10px 28px !important;
    border-radius: 4px !important;
}
.stButton button:hover {
    background: #00E5FF18 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #1C2A3A !important;
    border-radius: 8px !important;
}
hr { border-color: #1C2A3A !important; }
.stAlert {
    background: #0D1117 !important;
    border: 1px solid #1C2A3A !important;
    border-radius: 8px !important;
    color: #C8D8E8 !important;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: #E8F4FF !important;
}
</style>
""", unsafe_allow_html=True)

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0D1117",
    font=dict(family="DM Mono, monospace", color="#7A9AB8", size=11),
    margin=dict(t=30, b=30, l=10, r=10),
)
AXIS_STYLE = dict(gridcolor="#1C2A3A", linecolor="#1C2A3A",
                  zerolinecolor="#1C2A3A")

ACCENT = "#00E5FF"
DANGER = "#FF4B6E"
SUCCESS = "#00E676"
WARNING = "#FFB300"


class EmissionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


RF_FEATURES = [
    "month_index",
    "temperature_c",
    "gdp_index",
    "transport_activity",
    "industry_activity",
    "renewable_share_pct",
]

# Synthetic data linear term in utils/generate_dataset.py: emission -= 0.008 * renewable_share
RENEWABLE_PHYSICS_MT_PER_PCT = -0.008
RF_RENEW_DEAD_THRESHOLD = 1e-4


def rf_predict_whatif(rf, rf_sc, sim_state, latest, renewable_inc, feature_order):
    """
    RF prediction for the what-if panel. Shallow forests can assign the latest city-state
    to leaves whose path never splits on `renewable_share_pct`, so the model output stays
    flat when only that slider moves. If so, apply the dataset's linear renewable sensitivity.
    """
    X = np.array([[sim_state[f] for f in feature_order]])
    pred = float(rf.predict(rf_sc.transform(X))[0])
    if renewable_inc <= 0:
        return pred
    baseline_renew = float(latest["renewable_share_pct"])
    state0 = dict(sim_state)
    state0["renewable_share_pct"] = baseline_renew
    X0 = np.array([[state0[f] for f in feature_order]])
    pred0 = float(rf.predict(rf_sc.transform(X0))[0])
    if abs(pred - pred0) < RF_RENEW_DEAD_THRESHOLD:
        pred += RENEWABLE_PHYSICS_MT_PER_PCT * float(renewable_inc)
    return pred


def _whatif_base_state(latest):
    return {
        "month_index":         float(latest["month_index"]),
        "temperature_c":       float(latest["temperature_c"]),
        "gdp_index":           float(latest["gdp_index"]),
        "transport_activity":  float(latest["transport_activity"]),
        "industry_activity":   float(latest["industry_activity"]),
        "renewable_share_pct": float(latest["renewable_share_pct"]),
    }


def whatif_example_nudges(rf, rf_sc, latest, feature_order):
    """
    Fixed-size nudges vs the latest row for hint text (Δ Mt = scenario − baseline).
    Negative Δ means lower modeled CO₂.
    """
    base = _whatif_base_state(latest)
    p0 = rf_predict_whatif(rf, rf_sc, base, latest, 0, feature_order)

    def pair(label, state, r_inc):
        p = rf_predict_whatif(rf, rf_sc, state, latest, r_inc, feature_order)
        return label, p - p0

    rows = [
        pair(
            "Transport −10%",
            {**base, "transport_activity": base["transport_activity"] * 0.9},
            0,
        ),
        pair(
            "Renewable +10 pp",
            {
                **base,
                "renewable_share_pct": base["renewable_share_pct"] + 10,
            },
            10,
        ),
        pair(
            "Industry −10%",
            {**base, "industry_activity": base["industry_activity"] * 0.9},
            0,
        ),
        pair(
            "Temperature +2 °C",
            {**base, "temperature_c": base["temperature_c"] + 2},
            0,
        ),
    ]
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return p0, rows


def solve_policy_for_target(rf, rf_sc, latest, feature_order, target_mt, temp_adj, mode):
    """
    Search the displayed What-If slider space and return the lowest-friction
    policy mix that reaches the requested emission target.
    """
    candidates = []
    rows = []
    rows_without_renew = []

    for transport_red in range(0, 31):
        for renewable_inc in range(0, 31):
            for industry_eff in range(0, 16):
                transport_activity = latest["transport_activity"] * (1 - transport_red / 100)
                industry_activity = latest["industry_activity"] * (1 - industry_eff / 100)
                renewable_share = latest["renewable_share_pct"] + renewable_inc
                row = [
                    latest["month_index"],
                    latest["temperature_c"] + temp_adj,
                    latest["gdp_index"],
                    transport_activity,
                    industry_activity,
                    renewable_share,
                ]
                row_without_renew = row.copy()
                row_without_renew[5] = latest["renewable_share_pct"]
                difficulty = (
                    (transport_red / 30) * 40
                    + (renewable_inc / 30) * 25
                    + (industry_eff / 15) * 25
                    + (abs(temp_adj) / 5) * 10
                )
                candidates.append({
                    "transport": transport_red,
                    "renewable": renewable_inc,
                    "industry": industry_eff,
                    "difficulty": float(difficulty),
                })
                rows.append(row)
                rows_without_renew.append(row_without_renew)

    pred = rf.predict(rf_sc.transform(np.array(rows, dtype=float)))
    pred_without_renew = rf.predict(rf_sc.transform(np.array(rows_without_renew, dtype=float)))

    for candidate, emission, emission_without_renew in zip(candidates, pred, pred_without_renew):
        if (
            candidate["renewable"] > 0
            and abs(float(emission) - float(emission_without_renew)) < RF_RENEW_DEAD_THRESHOLD
        ):
            emission = float(emission) + RENEWABLE_PHYSICS_MT_PER_PCT * candidate["renewable"]
        candidate["emission"] = float(emission)

        lever_levels = [
            candidate["transport"] / 30,
            candidate["renewable"] / 30,
            candidate["industry"] / 15,
        ]
        active_levers = sum(level > 0 for level in lever_levels)
        concentration_penalty = max(lever_levels) * 20 - active_levers * 4
        if mode == "Balanced mix":
            candidate["solver_score"] = candidate["difficulty"] + concentration_penalty
        elif mode == "Transport-first":
            candidate["solver_score"] = candidate["difficulty"] - candidate["transport"] * 0.6
        elif mode == "Industry-light":
            candidate["solver_score"] = candidate["difficulty"] + candidate["industry"] * 1.2
        else:
            candidate["solver_score"] = candidate["difficulty"]

    best_overall = min(candidates, key=lambda c: c["emission"])
    meeting = [c for c in candidates if c["emission"] <= target_mt]
    best_meeting = min(
        meeting,
        key=lambda c: (c["solver_score"], c["emission"], c["transport"] + c["renewable"] + c["industry"]),
        default=None,
    )

    return best_meeting, best_overall


def compute_lstm_forecast(df, lstm_model, lstm_sc, lstm_feat, n_steps=6, seq_len=12):
    data = df[lstm_feat].values
    data_sc = lstm_sc.transform(data)
    seq = data_sc[-seq_len:].copy()
    future_preds = []
    lstm_model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            inp = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32)
            pred = lstm_model(inp).item()
            future_preds.append(pred)
            new_row = seq[-1].copy()
            new_row[-1] = pred
            new_row[0] += 1
            seq = np.vstack([seq[1:], new_row])
    dummy = np.zeros((len(future_preds), len(lstm_feat)))
    dummy[:, -1] = np.array(future_preds, dtype=np.float64)
    return lstm_sc.inverse_transform(dummy)[:, -1]


def compute_rf_horizon_forecast(rf, rf_sc, latest, n_steps=6):
    preds = []
    for k in range(n_steps):
        row = np.array([[
            latest["month_index"] + k + 1,
            latest["temperature_c"],
            latest["gdp_index"] + 0.05 * (k + 1),
            latest["transport_activity"],
            latest["industry_activity"],
            latest["renewable_share_pct"],
        ]])
        preds.append(rf.predict(rf_sc.transform(row))[0])
    return np.array(preds)


def load_ga_convergence():
    path = os.path.join(MODELS_DIR, "ga_convergence.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


USE_CASE_SCENARIOS = [
    {
        "id": "council_briefing",
        "title": "Council briefing — H1 2026 package",
        "who": "Chief Sustainability Officer",
        "story": (
            "**January 2026.** You must brief elected officials: where emissions are headed with no new policy, "
            "what a **data-backed** lever mix could achieve, and how that compares to an **ad‑hoc** package similar "
            "to typical city commitments (transport ↓, renewables ↑, industry efficiency)."
        ),
        "try_in_app": [
            ("OVERVIEW", "Anchor the room: current Mt, trend, sector split."),
            ("FORECAST", "Show the **LSTM** path vs **RF** baseline — justify why you care about June."),
            ("OPTIMIZATION", "Present **GA + RF** best bounded package (transparent constraints)."),
            ("COMPARISON", "Quantify **no‑action vs optimized** in one slide."),
            ("WHAT-IF SIMULATOR", "Load the council-style preset, then adjust live for “what if we go gentler on transport?”"),
        ],
        "preset": {
            "whatif_transport": 8,
            "whatif_renewable": 12,
            "whatif_industry": 5,
            "whatif_temp": 0,
        },
    },
    {
        "id": "heat_grid",
        "title": "Heat wave & cooling load",
        "who": "Utility / Disaster risk desk",
        "story": (
            "**Summer stress scenario.** Cooling demand rises with temperature; you want to show how **+3 °C** "
            "against the latest month **shifts modeled CO₂** if operations (transport/industry) are only partly curtailed."
        ),
        "try_in_app": [
            ("OVERVIEW", "Baseline context."),
            ("WHAT-IF SIMULATOR", "Load preset, then nudge **temperature** and transport to discuss demand response."),
            ("FORECAST", "Cite medium-term trajectory if leadership asks “is this structural?”"),
        ],
        "preset": {
            "whatif_transport": 6,
            "whatif_renewable": 10,
            "whatif_industry": 4,
            "whatif_temp": 3,
        },
    },
    {
        "id": "electrify",
        "title": "Electrification & renewable targets",
        "who": "Energy transition task force",
        "story": (
            "**Grid story.** Strong push on renewable share and moderate transport demand management — test how far "
            "the model says Mt moves before council asks for **industry** sacrifices."
        ),
        "try_in_app": [
            ("WHAT-IF SIMULATOR", "Start from preset; increase **Renewable** slider to stress-test."),
            ("OPTIMIZATION", "Compare your narrative to **GA**’s mix (may favour different weights)."),
            ("COMPARISON", "Tie back to June outlook."),
        ],
        "preset": {
            "whatif_transport": 10,
            "whatif_renewable": 22,
            "whatif_industry": 3,
            "whatif_temp": 0,
        },
    },
    {
        "id": "industry_window",
        "title": "Industrial efficiency window",
        "who": "Economic development + environment joint desk",
        "story": (
            "**Growth vs air quality.** A planned efficiency / production-smoothing window: heavier **industry** lever, "
            "lighter transport, moderate renewables — debate **GDP vs emissions** using the what-if readout."
        ),
        "try_in_app": [
            ("WHAT-IF SIMULATOR", "Load preset; slide **Industry** vs **Transport** to show trade-offs."),
            ("COMPARISON", "Show gap to no-action June."),
        ],
        "preset": {
            "whatif_transport": 4,
            "whatif_renewable": 8,
            "whatif_industry": 12,
            "whatif_temp": 0,
        },
    },
]


@st.cache_resource
def load_models():
    rf = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    rf_sc = joblib.load(os.path.join(MODELS_DIR, "rf_scaler.pkl"))
    lstm_sc = joblib.load(os.path.join(MODELS_DIR, "lstm_scaler.pkl"))
    lstm_feat = joblib.load(os.path.join(MODELS_DIR, "lstm_features.pkl"))
    lstm = EmissionLSTM(input_size=7, hidden_size=64,
                        num_layers=2, dropout=0.2)
    lstm.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, "lstm_model.pth"), map_location="cpu"))
    lstm.eval()
    with open(os.path.join(MODELS_DIR, "ga_result.json"), encoding="utf-8") as f:
        ga = json.load(f)
    return rf, rf_sc, lstm, lstm_sc, lstm_feat, ga


@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, "carbon_emission_data.csv"), parse_dates=["date"])


rf, rf_sc, lstm_model, lstm_sc, lstm_feat, ga_result = load_models()
df = load_data()
latest = df.iloc[-1]

if "fc_lstm" not in st.session_state:
    st.session_state.fc_months = pd.date_range(
        "2026-01-01", periods=6, freq="MS")
    st.session_state.fc_lstm = compute_lstm_forecast(
        df, lstm_model, lstm_sc, lstm_feat)
    st.session_state.fc_rf = compute_rf_horizon_forecast(
        rf, rf_sc, latest)

future_months = st.session_state.fc_months
fc_lstm = st.session_state.fc_lstm
fc_rf = st.session_state.fc_rf

for _wk, _wv in [
    ("whatif_transport", 0),
    ("whatif_renewable", 0),
    ("whatif_industry", 0),
    ("whatif_temp", 0),
]:
    if _wk not in st.session_state:
        st.session_state[_wk] = _wv

baseline_mt = float(latest["total_emission"])
if "whatif_target" not in st.session_state:
    st.session_state.whatif_target = round(baseline_mt * 0.94, 3)
if "whatif_solver_mode" not in st.session_state:
    st.session_state.whatif_solver_mode = "Balanced mix"

lstm_jun = float(fc_lstm[-1])
lstm_delta_jun_pct = (lstm_jun - baseline_mt) / baseline_mt * 100
if len(df) >= 12:
    y12 = float(df.iloc[-12]["total_emission"])
    trend_12_pct = (baseline_mt - y12) / y12 * 100
else:
    trend_12_pct = 0.0

ga_conv = load_ga_convergence()

te = float(latest["transport_emission"])
ie = float(latest["industry_emission"])
ee = float(latest["energy_emission"])
pie_sector_vals = [max(te, 0.0), max(ie, 0.0), max(ee, 0.0)]
pie_sum = sum(pie_sector_vals)
if pie_sum <= 0:
    pie_sector_vals = [40.0, 35.0, 25.0]

corr_feature_order = [
    "temperature_c", "transport_activity",
    "industry_activity", "renewable_share_pct",
]

def _corr_safe(col):
    c = df[col].corr(df["total_emission"])
    return float(c) if pd.notna(c) else 0.0

corr_vals = [_corr_safe(c) for c in corr_feature_order]
corr_labels = ["Temperature", "Transport", "Industry", "Renewable"]

ga_evals = int(ga_result.get("fitness_evaluations", 8000))
ga_gens = int(ga_result.get("generations", 80))

# Jun 2026 “no policy” trajectory end point = LSTM fan (single source of truth)
no_action_jun_mt = lstm_jun
no_action_delta_pct = lstm_delta_jun_pct

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 22px; font-weight: 800; color: #E8F4FF;'>
            🌍 CARBON<span style='color:#00E5FF;'>AI</span>
        </div>
        <div style='font-family: DM Mono, monospace; font-size: 10px;
                    color: #3A5A7A; letter-spacing: 0.15em; margin-top: 4px;'>
            CITY EMISSION CONTROL SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A5A7A;letter-spacing:0.12em;margin-bottom:10px;">LIVE CITY STATE — DEC 2024</div>', unsafe_allow_html=True)
    st.metric("Monthly CO₂",       f"{latest['total_emission']:.3f} Mt")
    st.metric("Transport Activity", f"{latest['transport_activity']:.1f}")
    st.metric("Renewable Share",    f"{latest['renewable_share_pct']:.1f}%")
    st.metric("Temperature",        f"{latest['temperature_c']:.1f} °C")
    st.divider()

    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A5A7A;letter-spacing:0.12em;margin-bottom:12px;">AI MODULES</div>', unsafe_allow_html=True)
    for label, color in [("Random Forest", SUCCESS), ("LSTM Network", SUCCESS), ("Genetic Algo.", SUCCESS)]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;padding:8px 0;
                    border-bottom:1px solid #1C2A3A;'>
            <span style='font-family:DM Mono,monospace;font-size:11px;color:#7A9AB8;'>{label}</span>
            <span style='font-family:DM Mono,monospace;font-size:10px;color:{color};'>● READY</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br><div style="font-family:DM Mono,monospace;font-size:10px;color:#2A3A4A;text-align:center;">v1.0 · SB Project 2026</div>', unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 4px 0;'>
    <div style='font-family: Syne, sans-serif; font-size: 32px; font-weight: 800;
                color: #E8F4FF; letter-spacing: -0.03em; line-height: 1.1;'>
        City Carbon Emission<br>
        <span style='color: #00E5FF;'>Intelligence System</span>
    </div>
    <div style='font-family: DM Mono, monospace; font-size: 12px;
                color: #3A6A8A; margin-top: 10px; letter-spacing: 0.05em;'>
        AI-POWERED PREDICTION & OPTIMIZATION DASHBOARD  ·  JANUARY 2026
    </div>
</div>
<hr style='margin: 20px 0; border-color: #1C2A3A;'>
""", unsafe_allow_html=True)

_uc_title = st.session_state.get("active_use_case_title", "")
_uc_html = (
    f"<br><span style='color:{ACCENT};font-size:13px;'>▸ Active demo path: {_uc_title}</span>"
    if _uc_title else ""
)
st.markdown(f"""
<div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px;'>
<div style='flex:1;min-width:220px;padding:16px 18px;background:#0D1117;border:1px solid #1C2A3A;border-radius:8px;border-left:3px solid {ACCENT};'>
<div style="font-family:DM Mono,monospace;font-size:9px;color:#5A7A9A;letter-spacing:0.14em;">EXECUTIVE SNAPSHOT</div>
<div style="font-family:Syne,sans-serif;font-size:15px;color:#E8F4FF;margin-top:10px;line-height:1.55;">
<b>{baseline_mt:.3f} Mt</b> CO₂ today (last CSV month)<br>
<b>{lstm_jun:.3f} Mt</b> LSTM Jun 2026 if no new policy &nbsp;({lstm_delta_jun_pct:+.1f}%)<br>
<b>{ga_result['optimized_emission']:.3f} Mt</b> under GA + RF optimum &nbsp;(−{ga_result['reduction_pct']}% vs today){_uc_html}
</div></div></div>
""", unsafe_allow_html=True)

with st.expander("About this demo (data & model roles)", expanded=False):
    st.markdown(
        """
**Data:** Monthly series is **synthetic** (see `utils/generate_dataset.py`) — a proof-of-concept city, not a live feed.

**Forecasts:** **LSTM** does recursive 6-step horizons from the last 12 months. **Random Forest** adds a **tabular baseline** trajectory (month index + mild GDP drift) for comparison on the Forecast tab.

**Optimization:** The **genetic algorithm** minimizes **RF-predicted** emissions with **bounded** policy levers — not a separate economic cost model; “economic viability” in copy means those bounds.

**GA chart:** When present, `models/ga_convergence.json` is the **recorded** best-so-far trace from the last `ga_optimizer.py` run.
        """
    )

tab_uc, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["USE CASES", "OVERVIEW", "FORECAST", "OPTIMIZATION", "COMPARISON", "WHAT-IF SIMULATOR"])

# ── TAB 0: USE CASES (demo narrative) ─────────────────────────────────────────
with tab_uc:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:6px;">── stakeholder-driven demo paths</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:22px;font-weight:700;color:#E8F4FF;margin-bottom:12px;">Pick a story, then walk the tabs</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Each scenario explains **who** is in the room, **what** decision they need, and **which tabs** to show. "
        "Presets load realistic lever combinations into the What-If simulator."
    )

    if st.session_state.get("active_use_case_title"):
        st.info(f"**Active path:** {st.session_state.active_use_case_title} — open **WHAT-IF SIMULATOR** to see loaded levers.")

    c_rst, _ = st.columns([0.28, 0.72])
    with c_rst:
        if st.button("Reset demo path & levers", key="btn_reset_use_case"):
            st.session_state.active_use_case_title = ""
            st.session_state.whatif_transport = 0
            st.session_state.whatif_renewable = 0
            st.session_state.whatif_industry = 0
            st.session_state.whatif_temp = 0
            st.rerun()

    st.markdown("---")

    for i, sc in enumerate(USE_CASE_SCENARIOS):
        expanded = sc["id"] == "council_briefing"
        with st.expander(f"**{sc['title']}** · *{sc['who']}*", expanded=expanded):
            st.markdown(sc["story"])
            st.markdown("**Suggested flow in this app**")
            for j, (tname, hint) in enumerate(sc["try_in_app"], start=1):
                st.markdown(f"{j}. **{tname}** — {hint}")
            st.caption(
                f"Preset levers: transport −{sc['preset']['whatif_transport']}%, "
                f"renewable +{sc['preset']['whatif_renewable']}%, "
                f"industry −{sc['preset']['whatif_industry']}%, "
                f"temp Δ {sc['preset']['whatif_temp']:+d} °C"
            )
            _btn_kw = {"key": f"btn_uc_{sc['id']}"}
            if sc["id"] == "council_briefing":
                _btn_kw["type"] = "primary"
            if st.button("Load preset into What-If sliders", **_btn_kw):
                for pk, pv in sc["preset"].items():
                    st.session_state[pk] = pv
                st.session_state.active_use_case_title = sc["title"]
                st.rerun()

    st.markdown("---")
    st.markdown(
        f'<div style="font-family:DM Mono,monospace;font-size:11px;color:#5A7A9A;">Tip: start with <b>Council briefing</b> for a full end-to-end pitch ({ACCENT} first expander).</div>',
        unsafe_allow_html=True,
    )

# ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Emission",
                f"{latest['total_emission']:.3f} Mt", "last CSV month")
    col2.metric("12-Month Trend",
                f"{trend_12_pct:+.1f}%", delta_color="inverse")
    col3.metric("LSTM Jun 2026 Forecast", f"{lstm_jun:.3f} Mt",
                f"{lstm_delta_jun_pct:+.1f}%",
                delta_color="inverse" if lstm_delta_jun_pct > 0 else "normal")
    col4.metric("GA Optimized Target",    f"{ga_result['optimized_emission']:.3f} Mt",
                f"-{ga_result['reduction_pct']}%", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:12px;">── 5-YEAR EMISSION TREND (2020–2024)</div>', unsafe_allow_html=True)

    df["rolling"] = df["total_emission"].rolling(3).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["total_emission"],
                             mode="lines", name="Monthly CO₂",
                             line=dict(color=ACCENT, width=2),
                             fill="tozeroy", fillcolor="rgba(0,229,255,0.04)"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["rolling"],
                             mode="lines", name="3-month avg",
                             line=dict(color=WARNING, width=1.5, dash="dot")))
    fig.update_layout(**PLOT_THEME, height=300,
                      yaxis=dict(range=[1.3, 2.7], **AXIS_STYLE),
                      xaxis=dict(**AXIS_STYLE),
                      legend=dict(orientation="h", y=1.12, font=dict(
                          family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                      hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:12px;">── SECTOR BREAKDOWN</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Transport", "Industry", "Energy"], values=pie_sector_vals,
            marker=dict(colors=[ACCENT, DANGER, SUCCESS],
                        line=dict(color="#080C10", width=3)),
            hole=0.6, textfont=dict(family="DM Mono", size=11)))
        fig_pie.update_layout(**PLOT_THEME, height=280,
                              legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)",
                                          orientation="h", y=-0.05))
        st.plotly_chart(fig_pie, width="stretch")

    with col_b:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:12px;">── FEATURE CORRELATION WITH EMISSIONS</div>', unsafe_allow_html=True)
        bar_colors = [ACCENT if c > 0.5 else WARNING if c >
                      0.3 else DANGER for c in corr_vals]
        fig_corr = go.Figure(go.Bar(
            x=corr_vals, y=corr_labels,
            orientation="h", marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{c:.3f}" for c in corr_vals], textposition="outside",
            textfont=dict(family="DM Mono", size=10, color="#7A9AB8")))
        fig_corr.update_layout(**PLOT_THEME, height=280,
                               xaxis=dict(range=[0, 0.85], **AXIS_STYLE),
                               yaxis=dict(**AXIS_STYLE))
        st.plotly_chart(fig_corr, width="stretch")

# ── TAB 2: FORECAST ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:6px;">── LSTM + RANDOM FORECAST · 6-MONTH HORIZON</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:20px;font-weight:700;color:#E8F4FF;margin-bottom:12px;">6-Month Emission Prediction</div>', unsafe_allow_html=True)
    st.caption(
        "Horizon labeled Jan–Jun 2026. LSTM = recurrent fan from the last 12 rows; RF = month-by-month tabular projection (GDP drift +1 step). "
        "Recompute after retraining models or changing the CSV.")

    c_btn, _ = st.columns([0.35, 0.65])
    with c_btn:
        if st.button("↻  Recompute forecasts", help="Clear cached horizon and rerun from current data/models"):
            for k in ("fc_lstm", "fc_rf", "fc_months"):
                st.session_state.pop(k, None)
            st.rerun()

    future_real = fc_lstm
    rf_future = fc_rf

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jan 2026 (LSTM)", f"{future_real[0]:.3f} Mt")
    col2.metric("Mar 2026 (LSTM)", f"{future_real[2]:.3f} Mt")
    col3.metric("May 2026 (LSTM)", f"{future_real[4]:.3f} Mt")
    col4.metric("Jun 2026 (LSTM)", f"{future_real[5]:.3f} Mt",
                f"{((future_real[5]-baseline_mt)/baseline_mt*100):+.1f}%",
                delta_color="inverse" if future_real[5] > baseline_mt else "normal")

    st.markdown("<br>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["total_emission"],
                             mode="lines", name="Historical",
                             line=dict(color="#3A6A8A", width=1.5),
                             fill="tozeroy", fillcolor="rgba(58,106,138,0.08)"))
    fig.add_trace(go.Scatter(x=future_months, y=rf_future,
                             mode="lines+markers", name="RF baseline trajectory",
                             line=dict(color=WARNING, width=2, dash="dash"),
                             marker=dict(size=7, color=WARNING, line=dict(color="#080C10", width=1))))
    fig.add_trace(go.Scatter(x=future_months, y=future_real,
                             mode="lines+markers", name="LSTM forecast",
                             line=dict(color=DANGER, width=2.5),
                             marker=dict(size=9, color=DANGER, line=dict(color="#080C10", width=2))))
    fig.add_trace(go.Scatter(
        x=list(future_months) + list(future_months[::-1]),
        y=list(future_real + 0.06) + list((future_real - 0.06)[::-1]),
        fill="toself", fillcolor="rgba(255,75,110,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Illustrative ±0.06 Mt band", hoverinfo="skip"))
    fig.add_vline(x=pd.Timestamp("2025-12-01").timestamp() * 1000,
                  line_dash="dot", line_color="#3A6A8A",
                  annotation_text="FORECAST →",
                  annotation_font=dict(family="DM Mono", size=10, color="#3A6A8A"))
    fig.update_layout(**PLOT_THEME, height=360,
                      yaxis=dict(range=[1.3, 2.6], **AXIS_STYLE),
                      xaxis=dict(**AXIS_STYLE),
                      legend=dict(orientation="h", y=1.12, font=dict(
                          family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                      hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    def _delta_label(v):
        p = (v - baseline_mt) / baseline_mt * 100
        return f"{p:+.2f}%"

    forecast_df = pd.DataFrame({
        "Month":          [m.strftime("%B %Y") for m in future_months],
        "LSTM (Mt)":     [round(float(v), 4) for v in future_real],
        "RF baseline (Mt)": [round(float(v), 4) for v in rf_future],
        "Δ vs today (LSTM)": [_delta_label(v) for v in future_real],
    })
    st.dataframe(forecast_df, width="stretch", hide_index=True)

    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download forecast table (CSV)",
        data=csv_bytes,
        file_name="carbon_forecast_6m.csv",
        mime="text/csv",
    )

    jun_pct = (future_real[5] - baseline_mt) / baseline_mt * 100
    trend_word = "increase" if jun_pct > 0 else "decrease"
    st.markdown(f"""
    <div style='margin-top:16px;padding:14px 20px;background:#120810;
                border:1px solid {DANGER}40;border-left:3px solid {DANGER};
                border-radius:6px;font-family:DM Mono,monospace;font-size:12px;color:#C8D8E8;'>
        ⚠ &nbsp; LSTM path: June 2026
        <span style='color:{DANGER};font-weight:500;'>{future_real[5]:.3f} Mt</span>
        — <span style='color:{DANGER};'>{abs(jun_pct):.1f}% {trend_word}</span>
        vs today ({baseline_mt:.3f} Mt). RF June:
        <span style='color:{WARNING};font-weight:500;'>{rf_future[5]:.3f} Mt</span>.
    </div>""", unsafe_allow_html=True)

# ── TAB 3: OPTIMIZATION ───────────────────────────────────────────────────────
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:6px;">── GENETIC ALGORITHM — POLICY OPTIMIZER</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:20px;font-weight:700;color:#E8F4FF;margin-bottom:20px;">Optimal Reduction Strategy</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Transport Reduction",
                f"{ga_result['transport_reduction']}%",  "policy lever")
    col2.metric("Renewable Increase",
                f"+{ga_result['renewable_increase']}%",  "energy share")
    col3.metric("Industrial Efficiency",
                f"{ga_result['industry_efficiency']}%",  "output reduction")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.1, 0.9])

    with col_a:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:16px;">── GA CONVERGENCE (RECORDED)</div>', unsafe_allow_html=True)
        base_e = ga_result["baseline_emission"]
        opt_e = ga_result["optimized_emission"]
        fig_conv = go.Figure()
        if ga_conv and "best_emission_mt" in ga_conv:
            gx = ga_conv.get("generation", list(range(1, len(ga_conv["best_emission_mt"]) + 1)))
            gy = ga_conv["best_emission_mt"]
            fig_conv.add_trace(go.Scatter(
                x=gx, y=gy, mode="lines",
                line=dict(color=SUCCESS, width=2),
                fill="tozeroy", fillcolor="rgba(0,230,118,0.05)",
                name="Best-so-far (RF Mt)"))
        else:
            st.caption("Run `python models/ga_optimizer.py` from project root to create `ga_convergence.json`.")
            gens = np.arange(1, ga_gens + 1)
            conv = opt_e + (base_e - opt_e) * np.exp(-0.12 * gens)
            fig_conv.add_trace(go.Scatter(
                x=gens, y=conv, mode="lines",
                line=dict(color="#5A7A7A", width=2, dash="dot"),
                name="Placeholder curve"))
        fig_conv.add_hline(y=opt_e, line_dash="dot", line_color=ACCENT,
                           annotation_text=f"Optimum: {opt_e:.4f}",
                           annotation_font=dict(family="DM Mono", size=10, color=ACCENT))
        fig_conv.update_layout(**PLOT_THEME, height=300,
                               xaxis=dict(title="Generation", **AXIS_STYLE),
                               yaxis=dict(title="Best emission (Mt)", **AXIS_STYLE),
                               showlegend=False)
        st.plotly_chart(fig_conv, width="stretch")

    with col_b:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:16px;">── POLICY LEVER STRENGTH</div>', unsafe_allow_html=True)
        fig_policy = go.Figure(go.Bar(
            x=["Transport", "Renewable", "Industry"],
            y=[ga_result["transport_reduction"],
                ga_result["renewable_increase"], ga_result["industry_efficiency"]],
            marker=dict(color=[ACCENT, SUCCESS, WARNING], line=dict(width=0)),
            text=[f"{v:.1f}%" for v in [ga_result["transport_reduction"],
                                        ga_result["renewable_increase"], ga_result["industry_efficiency"]]],
            textposition="outside",
            textfont=dict(family="DM Mono", size=11, color="#C8D8E8")))
        fig_policy.update_layout(**PLOT_THEME, height=300,
                                 yaxis=dict(range=[0, 28], **AXIS_STYLE),
                                 xaxis=dict(**AXIS_STYLE),
                                 showlegend=False)
        st.plotly_chart(fig_policy, width="stretch")

    st.markdown(f"""
    <div style='padding:16px 20px;background:#081208;border:1px solid {SUCCESS}40;
                border-left:3px solid {SUCCESS};border-radius:6px;
                font-family:DM Mono,monospace;font-size:12px;color:#C8D8E8;'>
        ✓ &nbsp; GA ran <span style='color:{SUCCESS};'>{ga_gens} generations</span>
        ({ga_evals:,} RF evaluations). Objective: minimize <b>RF-predicted</b> Mt under lever bounds.
        Result: <b>{ga_result['baseline_emission']} Mt</b> →
        <span style='color:{SUCCESS};font-weight:500;'>{ga_result['optimized_emission']} Mt</span>
        (−{ga_result['reduction_pct']}% vs that baseline state).
    </div>""", unsafe_allow_html=True)

# ── TAB 4: COMPARISON ─────────────────────────────────────────────────────────
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:6px;">── SCENARIO ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:20px;font-weight:700;color:#E8F4FF;margin-bottom:20px;">No Action vs Optimized Strategy</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Today",
                f"{ga_result['baseline_emission']} Mt", "last CSV month")
    col2.metric("No Action — Jun 2026 (LSTM)",
                f"{no_action_jun_mt:.3f} Mt",
                f"{no_action_delta_pct:+.1f}%",
                delta_color="inverse" if no_action_delta_pct > 0 else "normal")
    opt_vs_baseline = (
        (ga_result["baseline_emission"] - ga_result["optimized_emission"])
        / ga_result["baseline_emission"] * 100)
    col3.metric("Optimized — Jun 2026 (RF+GA)",
                f"{ga_result['optimized_emission']:.3f} Mt",
                f"-{opt_vs_baseline:.1f}% vs today",
                delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)
    no_action = np.array(fc_lstm, dtype=float)
    optimized = np.linspace(
        ga_result["baseline_emission"], ga_result["optimized_emission"], 6)

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(x=future_months, y=no_action,
                                 mode="lines+markers", name="No action (LSTM fan)",
                                 line=dict(color=DANGER, width=2.5),
                                 marker=dict(size=8, color=DANGER, line=dict(color="#080C10", width=2))))
    fig_cmp.add_trace(go.Scatter(x=future_months, y=optimized,
                                 mode="lines+markers", name="Optimized ramp (policy target)",
                                 line=dict(color=SUCCESS, width=2.5),
                                 marker=dict(size=8, color=SUCCESS, line=dict(
                                     color="#080C10", width=2)),
                                 fill="tonexty", fillcolor="rgba(0,230,118,0.06)"))
    fig_cmp.update_layout(**PLOT_THEME, height=360,
                          yaxis=dict(range=[1.7, 2.4], **AXIS_STYLE),
                          xaxis=dict(**AXIS_STYLE),
                          legend=dict(orientation="h", y=1.12, font=dict(
                              family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                          hovermode="x unified")
    st.plotly_chart(fig_cmp, width="stretch")

    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin:20px 0 12px 0;">── DECISION SUMMARY</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        "Scenario":          ["No Action (LSTM)", "Optimized (RF+GA)"],
        "Jun 2026 Emission": [f"{no_action_jun_mt:.4f} Mt",
                             f"{ga_result['optimized_emission']} Mt"],
        "Δ from Today":      [f"{no_action_delta_pct:+.1f}%",
                             f"−{ga_result['reduction_pct']}% (vs GA baseline state)"],
        "Transport Policy":  ["—",        f"↓ {ga_result['transport_reduction']}%"],
        "Renewable Policy":  ["—",        f"↑ {ga_result['renewable_increase']}%"],
        "Industry Policy":   ["—",        f"↓ {ga_result['industry_efficiency']}%"],
        "Economic Impact":   ["N/A",      "Bounded levers only"],
    })
    st.dataframe(summary_df, width="stretch", hide_index=True)

    saved = no_action_jun_mt - float(ga_result["optimized_emission"])
    st.markdown(f"""
    <div style='margin-top:16px;display:flex;gap:12px;'>
        <div style='flex:1;padding:20px;background:#080C10;border:1px solid #1C2A3A;
                    border-radius:8px;text-align:center;'>
            <div style='font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.1em;'>TONS SAVED / MONTH</div>
            <div style='font-family:Syne,sans-serif;font-size:28px;font-weight:800;color:{SUCCESS};margin-top:8px;'>{saved*1e6:,.0f}</div>
        </div>
        <div style='flex:1;padding:20px;background:#080C10;border:1px solid #1C2A3A;
                    border-radius:8px;text-align:center;'>
            <div style='font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.1em;'>REDUCTION %</div>
            <div style='font-family:Syne,sans-serif;font-size:28px;font-weight:800;color:{SUCCESS};margin-top:8px;'>{ga_result["reduction_pct"]}%</div>
        </div>
        <div style='flex:1;padding:20px;background:#080C10;border:1px solid #1C2A3A;
                    border-radius:8px;text-align:center;'>
            <div style='font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.1em;'>ECONOMIC STATUS</div>
            <div style='font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:{ACCENT};margin-top:8px;'>VIABLE</div>
        </div>
    </div>""", unsafe_allow_html=True)

# ── TAB 5: WHAT-IF SIMULATOR ──────────────────────────────────────────────────
with tab5:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:6px;">── REAL-TIME POLICY SIMULATOR</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:20px;font-weight:700;color:#E8F4FF;margin-bottom:4px;">What-If Scenario Builder</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:12px;color:#3A6A8A;margin-bottom:16px;">Drag the sliders to simulate any policy combination. Predictions use the Random Forest; if it does not respond to renewable share for this row (common with shallow trees), a small physics term from the data generator (−0.008 Mt per %-point) is applied so the lever is visible.</div>', unsafe_allow_html=True)

    _p0_hint, _nudge_rows = whatif_example_nudges(rf, rf_sc, latest, RF_FEATURES)
    with st.expander("Lever guide — if you change X, what happens to CO₂?", expanded=True):
        st.markdown(
            """
**Direction (this demo dataset + RF):**

| You move… | Modeled effect on monthly Mt (usually) |
|-----------|----------------------------------------|
| **Transport ↓** | **Total CO₂ ↓** — less activity → less burn; strong driver in training. |
| **Renewable share ↑** | **Total CO₂ ↓** — cleaner grid; RF may be flat on the last row, then the **−0.008 Mt / %-point** fallback matches the synthetic formula. |
| **Industry ↓** | **Total CO₂ ↓** — less industrial throughput in this simplified model. |
| **Temperature ↑** | **Total CO₂ ↑** here — proxy for cooling / stress (see `generate_dataset.py`); **↓** tends the other way. |

**To lower Mt:** combine transport and/or industry reductions with higher renewables before leaning only on temperature.
            """
        )
        st.markdown(
            f"*Fixed test nudges from the **latest CSV month** (baseline **{_p0_hint:.3f}** Mt):*"
        )
        for _lbl, _dm in _nudge_rows:
            _arrow = "↓ Mt" if _dm < 0 else "↑ Mt" if _dm > 0 else "≈ flat"
            st.markdown(
                f"- **{_lbl}** → **{_dm:+.4f}** Mt ({_arrow}) vs that baseline."
            )
        if _nudge_rows:
            _top_l, _top_d = _nudge_rows[0]
            st.success(
                f"Among those examples, **{_top_l}** has the **largest** modeled move "
                f"({_top_d:+.4f} Mt). Your sliders scale beyond these steps."
            )

    col_sliders, col_result = st.columns([1, 1])

    with col_sliders:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:16px;">── POLICY LEVERS</div>', unsafe_allow_html=True)

        c_target, c_solve = st.columns([0.62, 0.38])
        with c_target:
            target_mt = st.number_input(
                "Target emission (Mt)",
                min_value=0.5,
                max_value=3.0,
                step=0.01,
                format="%.3f",
                help="Demo goal for the current what-if scenario. The app will score whether the slider mix reaches it.",
                key="whatif_target",
            )
        with c_solve:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("Find target plan", help="Search slider combinations for a plan that reaches the target."):
                found, best = solve_policy_for_target(
                    rf, rf_sc, latest, RF_FEATURES, target_mt, st.session_state.whatif_temp,
                    st.session_state.whatif_solver_mode)
                solver_pick = found or best
                st.session_state.whatif_transport = solver_pick["transport"]
                st.session_state.whatif_renewable = solver_pick["renewable"]
                st.session_state.whatif_industry = solver_pick["industry"]
                if found:
                    st.session_state.whatif_solver_note = (
                        f"{st.session_state.whatif_solver_mode} solver found a target plan: {found['emission']:.3f} Mt "
                        f"at difficulty {found['difficulty']:.0f}/100."
                    )
                else:
                    st.session_state.whatif_solver_note = (
                        f"Target is outside the current slider bounds. Best found: "
                        f"{best['emission']:.3f} Mt at difficulty {best['difficulty']:.0f}/100."
                    )

        st.selectbox(
            "Solver style",
            ["Balanced mix", "Lowest-friction", "Transport-first", "Industry-light"],
            help=(
                "Balanced mix spreads action across levers. Lowest-friction may lean heavily on renewables "
                "because this demo dataset gives that lever strong impact. Transport-first and Industry-light "
                "bias the search for presentation scenarios."
            ),
            key="whatif_solver_mode",
        )

        if st.session_state.get("whatif_solver_note"):
            st.info(st.session_state.whatif_solver_note)

        transport_red = st.slider(
            "🚗  Transport Activity Reduction (%)",
            min_value=0, max_value=30, step=1,
            help="Reduce transport activity via restrictions, EV adoption, public transit",
            key="whatif_transport",
        )
        renewable_inc = st.slider(
            "☀️  Renewable Energy Increase (%)",
            min_value=0, max_value=30, step=1,
            help="Increase share of renewable energy in the grid",
            key="whatif_renewable",
        )
        industry_eff = st.slider(
            "🏭  Industrial Efficiency Improvement (%)",
            min_value=0, max_value=15, step=1,
            help="Reduce industrial activity through efficiency measures",
            key="whatif_industry",
        )
        temp_adj = st.slider(
            "🌡️  Temperature Adjustment (°C)",
            min_value=-5, max_value=5, step=1,
            help="Simulate seasonal or climate scenario",
            key="whatif_temp",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='padding:12px 16px;background:#0D1117;border:1px solid #1C2A3A;
                    border-radius:8px;font-family:DM Mono,monospace;font-size:11px;
                    color:#5A7A9A;line-height:2;'>
            Transport activity &nbsp;&nbsp; {latest['transport_activity'] * (1 - transport_red/100):.1f}
            &nbsp; (was {latest['transport_activity']:.1f})<br>
            Renewable share &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {latest['renewable_share_pct'] + renewable_inc:.1f}%
            &nbsp; (was {latest['renewable_share_pct']:.1f}%)<br>
            Industry activity &nbsp;&nbsp;&nbsp;&nbsp; {latest['industry_activity'] * (1 - industry_eff/100):.1f}
            &nbsp; (was {latest['industry_activity']:.1f})<br>
            Temperature &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {latest['temperature_c'] + temp_adj:.1f}°C
            &nbsp; (was {latest['temperature_c']:.1f}°C)
        </div>
        """, unsafe_allow_html=True)
        st.caption(
            "Tip: **lower** transport or industry, or **raise** renewables, to push Mt **down**; "
            "**raise** temperature to show a heat-stress case."
        )

    with col_result:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:16px;">── PREDICTED OUTCOME</div>', unsafe_allow_html=True)

        # Build adjusted feature vector and predict
        sim_state = {
            "month_index":         latest["month_index"],
            "temperature_c":       latest["temperature_c"] + temp_adj,
            "gdp_index":           latest["gdp_index"],
            "transport_activity":  latest["transport_activity"] * (1 - transport_red / 100),
            "industry_activity":   latest["industry_activity"] * (1 - industry_eff / 100),
            "renewable_share_pct": latest["renewable_share_pct"] + renewable_inc,
        }

        sim_emission = rf_predict_whatif(
            rf, rf_sc, sim_state, latest, renewable_inc, RF_FEATURES)

        baseline = latest["total_emission"]
        delta = sim_emission - baseline
        delta_pct = (delta / baseline) * 100
        is_reduction = delta < 0
        result_color = SUCCESS if is_reduction else DANGER
        arrow = "↓" if is_reduction else "↑"
        target_gap = sim_emission - target_mt
        target_met = target_gap <= 0

        impact_score = float(np.clip((-delta_pct / 10) * 100, 0, 100))
        difficulty_score = (
            (transport_red / 30) * 40
            + (renewable_inc / 30) * 25
            + (industry_eff / 15) * 25
            + (abs(temp_adj) / 5) * 10
        )
        target_score = 100 if target_met else float(np.clip(100 - (target_gap / baseline) * 500, 0, 100))
        overall_score = 0.55 * impact_score + 0.30 * target_score + 0.15 * (100 - difficulty_score)
        if overall_score >= 85:
            scenario_grade = "A"
            grade_color = SUCCESS
        elif overall_score >= 70:
            scenario_grade = "B"
            grade_color = ACCENT
        elif overall_score >= 55:
            scenario_grade = "C"
            grade_color = WARNING
        else:
            scenario_grade = "D"
            grade_color = DANGER

        feasibility = "LOW FRICTION" if difficulty_score < 35 else "MODERATE" if difficulty_score < 65 else "HIGH INTENSITY"

        st.markdown(f"""
        <div style='padding:28px;background:#0D1117;border:1px solid #1C2A3A;
                    border-radius:12px;text-align:center;margin-bottom:16px;'>
            <div style='font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;
                        letter-spacing:0.15em;margin-bottom:12px;'>PREDICTED EMISSION</div>
            <div style='font-family:Syne,sans-serif;font-size:48px;font-weight:800;
                        color:{result_color};line-height:1;'>{sim_emission:.3f}</div>
            <div style='font-family:DM Mono,monospace;font-size:14px;color:#5A7A9A;
                        margin-top:6px;'>million tons CO₂</div>
            <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:700;
                        color:{result_color};margin-top:16px;'>
                {arrow} {abs(delta_pct):.1f}% {'reduction' if is_reduction else 'increase'}
            </div>
            <div style='font-family:DM Mono,monospace;font-size:11px;color:#3A6A8A;margin-top:6px;'>
                vs baseline of {baseline:.3f} Mt
            </div>
        </div>
        """, unsafe_allow_html=True)

        target_status = "TARGET MET" if target_met else "TARGET GAP"
        target_text = (
            f"{abs(target_gap):.3f} Mt below target"
            if target_met else f"{target_gap:.3f} Mt above target"
        )
        target_color = SUCCESS if target_met else WARNING
        st.markdown(f"""
        <div style='display:flex;gap:12px;margin-bottom:16px;'>
            <div style='flex:1;padding:16px;background:#0D1117;border:1px solid {target_color}60;border-radius:8px;'>
                <div style='font-family:DM Mono,monospace;font-size:10px;color:#5A7A9A;letter-spacing:0.12em;'>{target_status}</div>
                <div style='font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:{target_color};margin-top:6px;'>{target_text}</div>
                <div style='font-family:DM Mono,monospace;font-size:11px;color:#3A6A8A;margin-top:6px;'>Goal: {target_mt:.3f} Mt</div>
            </div>
            <div style='width:120px;padding:16px;background:#0D1117;border:1px solid {grade_color}60;border-radius:8px;text-align:center;'>
                <div style='font-family:DM Mono,monospace;font-size:10px;color:#5A7A9A;letter-spacing:0.12em;'>PLAN GRADE</div>
                <div style='font-family:Syne,sans-serif;font-size:34px;font-weight:800;color:{grade_color};line-height:1;margin-top:6px;'>{scenario_grade}</div>
                <div style='font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;margin-top:6px;'>{overall_score:.0f}/100</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Mini gauge bar
        pct_of_max = min(sim_emission / 2.8, 1.0)
        bar_width = int(pct_of_max * 100)
        st.markdown(f"""
        <div style='font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;
                    letter-spacing:0.12em;margin-bottom:8px;'>EMISSION LEVEL</div>
        <div style='background:#1C2A3A;border-radius:4px;height:10px;overflow:hidden;'>
            <div style='width:{bar_width}%;height:100%;background:{result_color};
                        border-radius:4px;transition:width 0.3s ease;'></div>
        </div>
        <div style='display:flex;justify-content:space-between;
                    font-family:DM Mono,monospace;font-size:10px;color:#3A5A7A;margin-top:4px;'>
            <span>0 Mt</span><span>1.4 Mt</span><span>2.8 Mt</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Breakdown chart: what each lever contributed
        contributions = []
        labels_c = []

        if transport_red > 0:
            t_only = rf.predict(rf_sc.transform(np.array([[
                sim_state["month_index"], sim_state["temperature_c"],
                sim_state["gdp_index"],
                latest["transport_activity"] * (1 - transport_red/100),
                latest["industry_activity"], latest["renewable_share_pct"]
            ]])))[0]
            contributions.append(round(baseline - t_only, 4))
            labels_c.append("Transport")

        if renewable_inc > 0:
            sim_r = {
                "month_index":         latest["month_index"],
                "temperature_c":       latest["temperature_c"],
                "gdp_index":           latest["gdp_index"],
                "transport_activity":  latest["transport_activity"],
                "industry_activity":   latest["industry_activity"],
                "renewable_share_pct": latest["renewable_share_pct"] + renewable_inc,
            }
            r_only = rf_predict_whatif(
                rf, rf_sc, sim_r, latest, renewable_inc, RF_FEATURES)
            contributions.append(round(baseline - r_only, 4))
            labels_c.append("Renewable")

        if industry_eff > 0:
            i_only = rf.predict(rf_sc.transform(np.array([[
                sim_state["month_index"], sim_state["temperature_c"],
                sim_state["gdp_index"], latest["transport_activity"],
                latest["industry_activity"] * (1 - industry_eff/100),
                latest["renewable_share_pct"]
            ]])))[0]
            contributions.append(round(baseline - i_only, 4))
            labels_c.append("Industry")

        if contributions:
            st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:10px;">── PER-LEVER CONTRIBUTION (Mt saved)</div>', unsafe_allow_html=True)
            fig_contrib = go.Figure(go.Bar(
                x=labels_c, y=contributions,
                marker=dict(color=[ACCENT, SUCCESS, WARNING][:len(contributions)],
                            line=dict(width=0)),
                text=[f"{v:.4f}" for v in contributions],
                textposition="outside",
                textfont=dict(family="DM Mono", size=11, color="#C8D8E8"),
            ))
            fig_contrib.update_layout(**PLOT_THEME, height=220,
                                      yaxis=dict(
                                          gridcolor="#1C2A3A", linecolor="#1C2A3A", zerolinecolor="#1C2A3A"),
                                      xaxis=dict(gridcolor="#1C2A3A", linecolor="#1C2A3A",
                                                 zerolinecolor="#1C2A3A"),
                                      showlegend=False)
            st.plotly_chart(fig_contrib, width="stretch")
        else:
            st.markdown("""
            <div style='padding:20px;text-align:center;border:1px dashed #1C2A3A;
                        border-radius:8px;font-family:DM Mono,monospace;font-size:11px;color:#3A5A7A;'>
                Move the sliders to see per-lever contribution
            </div>""", unsafe_allow_html=True)

        if contributions:
            best_idx = int(np.argmax([abs(v) for v in contributions]))
            best_lever = labels_c[best_idx]
            best_value = contributions[best_idx]
            if best_value > 0:
                lever_note = f"{best_lever} is doing the most useful work in this scenario ({best_value:.4f} Mt saved on its own)."
            else:
                lever_note = f"{best_lever} has the largest modeled movement, but it is not improving this scenario by itself."
        else:
            lever_note = "No policy lever is active yet, so this is effectively the current-city baseline."

        if not target_met:
            recommendation = (
                f"Target is still {target_gap:.3f} Mt away. Increase the strongest clean lever first, then tune the others: {lever_note}"
            )
            rec_color = WARNING
        elif difficulty_score >= 65:
            recommendation = (
                f"The target is met, but the package is high intensity. Try easing one lever at a time while keeping emissions under {target_mt:.3f} Mt. {lever_note}"
            )
            rec_color = ACCENT
        elif is_reduction:
            recommendation = (
                f"This is a strong demo plan: target met, emissions are down {abs(delta_pct):.1f}%, and policy intensity is {feasibility.lower()}. {lever_note}"
            )
            rec_color = SUCCESS
        else:
            recommendation = (
                f"The target is met only because the goal is above the current prediction; this is not a reduction scenario yet. {lever_note}"
            )
            rec_color = WARNING

        st.markdown(f"""
        <div style='margin-top:16px;padding:16px 18px;background:#0D1117;
                    border:1px solid {rec_color}60;border-left:3px solid {rec_color};
                    border-radius:8px;font-family:DM Mono,monospace;font-size:12px;
                    color:#C8D8E8;line-height:1.7;'>
            <div style='font-size:10px;color:#5A7A9A;letter-spacing:0.14em;margin-bottom:6px;'>RECOMMENDATION</div>
            {recommendation}<br>
            <span style='color:#5A7A9A;'>Impact score: {impact_score:.0f}/100 · Feasibility: {feasibility} · Difficulty: {difficulty_score:.0f}/100</span>
        </div>
        """, unsafe_allow_html=True)

        top_lever = best_lever if contributions else "None"
        top_lever_impact = best_value if contributions else 0.0
        current_plan = {
            "Target (Mt)": round(float(target_mt), 3),
            "Predicted Emission (Mt)": round(float(sim_emission), 4),
            "Target Gap (Mt)": round(float(target_gap), 4),
            "Target Met": "Yes" if target_met else "No",
            "Grade": scenario_grade,
            "Score": round(float(overall_score), 1),
            "Difficulty": round(float(difficulty_score), 1),
            "Feasibility": feasibility,
            "Transport Reduction (%)": int(transport_red),
            "Renewable Increase (%)": int(renewable_inc),
            "Industry Efficiency (%)": int(industry_eff),
            "Temperature Adjustment (C)": int(temp_adj),
            "Top Lever": top_lever,
            "Top Lever Impact (Mt)": round(float(top_lever_impact), 4),
            "Recommendation": recommendation,
        }

        st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin:20px 0 10px 0;">── DECISION TOOLS</div>', unsafe_allow_html=True)
        c_plan_a, c_plan_b, c_report = st.columns([0.28, 0.28, 0.44])
        with c_plan_a:
            if st.button("Save as Plan A", key="save_plan_a"):
                st.session_state.whatif_plan_a = dict(current_plan)
                st.success("Plan A saved.")
        with c_plan_b:
            if st.button("Save as Plan B", key="save_plan_b"):
                st.session_state.whatif_plan_b = dict(current_plan)
                st.success("Plan B saved.")

        contribution_lines = (
            "\n".join(f"- {label}: {value:+.4f} Mt saved" for label, value in zip(labels_c, contributions))
            if contributions else "- No active levers."
        )
        report_text = f"""# CarbonAI What-If Decision Report

Baseline emission: {baseline:.4f} Mt
Target emission: {target_mt:.4f} Mt
Predicted scenario emission: {sim_emission:.4f} Mt
Target status: {'Met' if target_met else 'Missed'} ({target_gap:+.4f} Mt)
Plan grade: {scenario_grade} ({overall_score:.1f}/100)
Feasibility: {feasibility}
Difficulty: {difficulty_score:.1f}/100

## Policy Levers

- Transport reduction: {transport_red}%
- Renewable increase: {renewable_inc}%
- Industry efficiency: {industry_eff}%
- Temperature adjustment: {temp_adj:+d} C

## Per-Lever Contribution

{contribution_lines}

## Recommendation

{recommendation}
"""
        with c_report:
            st.download_button(
                "Download current report",
                data=report_text.encode("utf-8"),
                file_name="carbonai_whatif_report.md",
                mime="text/markdown",
                key="download_whatif_report",
            )

        saved_plans = []
        for label, key in [("Plan A", "whatif_plan_a"), ("Plan B", "whatif_plan_b")]:
            plan = st.session_state.get(key)
            if plan:
                saved_plans.append({"Plan": label, **plan})

        if saved_plans:
            compare_df = pd.DataFrame(saved_plans)[[
                "Plan",
                "Predicted Emission (Mt)",
                "Target Gap (Mt)",
                "Target Met",
                "Grade",
                "Score",
                "Difficulty",
                "Feasibility",
                "Transport Reduction (%)",
                "Renewable Increase (%)",
                "Industry Efficiency (%)",
                "Temperature Adjustment (C)",
                "Top Lever",
            ]]
            st.dataframe(compare_df, width="stretch", hide_index=True)

            fig_plan_cmp = go.Figure(go.Bar(
                x=compare_df["Plan"],
                y=compare_df["Predicted Emission (Mt)"],
                marker=dict(color=[ACCENT, SUCCESS][:len(compare_df)], line=dict(width=0)),
                text=[f"{v:.3f} Mt" for v in compare_df["Predicted Emission (Mt)"]],
                textposition="outside",
                textfont=dict(family="DM Mono", size=11, color="#C8D8E8"),
            ))
            fig_plan_cmp.add_hline(
                y=target_mt,
                line_dash="dot",
                line_color=WARNING,
                annotation_text=f"Target {target_mt:.3f} Mt",
                annotation_font=dict(family="DM Mono", size=10, color=WARNING),
            )
            fig_plan_cmp.update_layout(
                **PLOT_THEME,
                height=220,
                yaxis=dict(title="Predicted emission (Mt)", **AXIS_STYLE),
                xaxis=dict(**AXIS_STYLE),
                showlegend=False,
            )
            st.plotly_chart(fig_plan_cmp, width="stretch")
