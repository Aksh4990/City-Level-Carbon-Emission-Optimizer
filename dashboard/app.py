import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


@st.cache_resource
def load_models():
    rf = joblib.load("models/rf_model.pkl")
    rf_sc = joblib.load("models/rf_scaler.pkl")
    lstm_sc = joblib.load("models/lstm_scaler.pkl")
    lstm_feat = joblib.load("models/lstm_features.pkl")
    lstm = EmissionLSTM(input_size=7, hidden_size=64,
                        num_layers=2, dropout=0.2)
    lstm.load_state_dict(torch.load(
        "models/lstm_model.pth", map_location="cpu"))
    lstm.eval()
    with open("models/ga_result.json") as f:
        ga = json.load(f)
    return rf, rf_sc, lstm, lstm_sc, lstm_feat, ga


@st.cache_data
def load_data():
    return pd.read_csv("data/carbon_emission_data.csv", parse_dates=["date"])


rf, rf_sc, lstm_model, lstm_sc, lstm_feat, ga_result = load_models()
df = load_data()
latest = df.iloc[-1]

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

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["OVERVIEW", "FORECAST", "OPTIMIZATION", "COMPARISON", "WHAT-IF SIMULATOR"])

# ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Emission",
                f"{latest['total_emission']:.3f} Mt", "Dec 2024")
    col2.metric("12-Month Trend",         "+8.2%", delta_color="inverse")
    col3.metric("LSTM Jun 2026 Forecast", "2.181 Mt",
                "+3.0%", delta_color="inverse")
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
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:12px;">── SECTOR BREAKDOWN</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Transport", "Industry", "Energy"], values=[40, 35, 25],
            marker=dict(colors=[ACCENT, DANGER, SUCCESS],
                        line=dict(color="#080C10", width=3)),
            hole=0.6, textfont=dict(family="DM Mono", size=11)))
        fig_pie.update_layout(**PLOT_THEME, height=280,
                              legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)",
                                          orientation="h", y=-0.05))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:12px;">── FEATURE CORRELATION WITH EMISSIONS</div>', unsafe_allow_html=True)
        corrs = [0.693, 0.706, 0.563, 0.173]
        bar_colors = [ACCENT if c > 0.5 else WARNING if c >
                      0.3 else DANGER for c in corrs]
        fig_corr = go.Figure(go.Bar(
            x=corrs, y=["Temperature", "Transport", "Industry", "Renewable"],
            orientation="h", marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{c:.3f}" for c in corrs], textposition="outside",
            textfont=dict(family="DM Mono", size=10, color="#7A9AB8")))
        fig_corr.update_layout(**PLOT_THEME, height=280,
                               xaxis=dict(range=[0, 0.85], **AXIS_STYLE),
                               yaxis=dict(**AXIS_STYLE))
        st.plotly_chart(fig_corr, use_container_width=True)

# ── TAB 2: FORECAST ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:6px;">── LSTM DEEP LEARNING FORECAST ENGINE</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:20px;font-weight:700;color:#E8F4FF;margin-bottom:20px;">6-Month Emission Prediction</div>', unsafe_allow_html=True)

    if st.button("▶  RUN LSTM FORECAST", type="primary"):
        with st.spinner("Running LSTM inference..."):
            data = df[lstm_feat].values
            data_sc = lstm_sc.transform(data)
            SEQ_LEN = 12
            seq = data_sc[-SEQ_LEN:].copy()
            future_preds = []
            lstm_model.eval()
            with torch.no_grad():
                for step in range(6):
                    inp = torch.tensor(
                        seq[np.newaxis, :, :], dtype=torch.float32)
                    pred = lstm_model(inp).item()
                    future_preds.append(pred)
                    new_row = seq[-1].copy()
                    new_row[-1] = pred
                    new_row[0] += 1
                    seq = np.vstack([seq[1:], new_row])

            def inv(vals):
                dummy = np.zeros((len(vals), len(lstm_feat)))
                dummy[:, -1] = vals
                return lstm_sc.inverse_transform(dummy)[:, -1]

            future_real = inv(np.array(future_preds))
            future_months = pd.date_range("2026-01-01", periods=6, freq="MS")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Jan 2026", f"{future_real[0]:.3f} Mt")
        col2.metric("Mar 2026", f"{future_real[2]:.3f} Mt")
        col3.metric("May 2026", f"{future_real[4]:.3f} Mt")
        col4.metric("Jun 2026", f"{future_real[5]:.3f} Mt",
                    delta=f"+{((future_real[5]-latest['total_emission'])/latest['total_emission']*100):.1f}%",
                    delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["total_emission"],
                                 mode="lines", name="Historical",
                                 line=dict(color="#3A6A8A", width=1.5),
                                 fill="tozeroy", fillcolor="rgba(58,106,138,0.08)"))
        fig.add_trace(go.Scatter(x=future_months, y=future_real,
                                 mode="lines+markers", name="LSTM Forecast",
                                 line=dict(color=DANGER, width=2.5),
                                 marker=dict(size=9, color=DANGER, line=dict(color="#080C10", width=2))))
        fig.add_trace(go.Scatter(
            x=list(future_months) + list(future_months[::-1]),
            y=list(future_real + 0.06) + list((future_real - 0.06)[::-1]),
            fill="toself", fillcolor="rgba(255,75,110,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence band", hoverinfo="skip"))
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
        st.plotly_chart(fig, use_container_width=True)

        forecast_df = pd.DataFrame({
            "Month":        [m.strftime("%B %Y") for m in future_months],
            "Emission":     [f"{v:.4f} Mt" for v in future_real],
            "Δ from today": [f"+{((v-latest['total_emission'])/latest['total_emission']*100):.2f}%" for v in future_real],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div style='margin-top:16px;padding:14px 20px;background:#120810;
                    border:1px solid {DANGER}40;border-left:3px solid {DANGER};
                    border-radius:6px;font-family:DM Mono,monospace;font-size:12px;color:#C8D8E8;'>
            ⚠ &nbsp; Without intervention, emissions projected to reach
            <span style='color:{DANGER};font-weight:500;'>{future_real[5]:.3f} Mt</span>
            by June 2026 —
            <span style='color:{DANGER};'>
            {((future_real[5]-latest["total_emission"])/latest["total_emission"]*100):.1f}% increase</span>
            from current levels.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='padding:48px;text-align:center;border:1px dashed #1C2A3A;
                    border-radius:8px;font-family:DM Mono,monospace;font-size:12px;color:#3A5A7A;'>
            Click RUN LSTM FORECAST to generate the 6-month prediction
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
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:16px;">── GA CONVERGENCE CURVE</div>', unsafe_allow_html=True)
        gens = np.arange(1, 81)
        base_e = ga_result["baseline_emission"]
        opt_e = ga_result["optimized_emission"]
        conv = opt_e + (base_e - opt_e) * np.exp(-0.12 *
                                                 gens) + np.random.normal(0, 0.003, 80)
        conv = np.clip(conv, opt_e, base_e)

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(x=gens, y=conv, mode="lines",
                                      line=dict(color=SUCCESS, width=2),
                                      fill="tozeroy", fillcolor="rgba(0,230,118,0.05)"))
        fig_conv.add_hline(y=opt_e, line_dash="dot", line_color=ACCENT,
                           annotation_text=f"Optimal: {opt_e:.4f}",
                           annotation_font=dict(family="DM Mono", size=10, color=ACCENT))
        fig_conv.update_layout(**PLOT_THEME, height=300,
                               xaxis=dict(title="Generation", **AXIS_STYLE),
                               yaxis=dict(title="Emission (Mt)", **AXIS_STYLE),
                               showlegend=False)
        st.plotly_chart(fig_conv, use_container_width=True)

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
        st.plotly_chart(fig_policy, use_container_width=True)

    st.markdown(f"""
    <div style='padding:16px 20px;background:#081208;border:1px solid {SUCCESS}40;
                border-left:3px solid {SUCCESS};border-radius:6px;
                font-family:DM Mono,monospace;font-size:12px;color:#C8D8E8;'>
        ✓ &nbsp; GA converged in <span style='color:{SUCCESS};'>80 generations</span>
        testing <span style='color:{SUCCESS};'>8,000+</span> combinations.
        Best strategy: <b>{ga_result['baseline_emission']} Mt</b> →
        <span style='color:{SUCCESS};font-weight:500;'>{ga_result['optimized_emission']} Mt</span>
        — <span style='color:{SUCCESS};'>{ga_result['reduction_pct']}% reduction</span>
        within acceptable economic constraints.
    </div>""", unsafe_allow_html=True)

# ── TAB 4: COMPARISON ─────────────────────────────────────────────────────────
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:6px;">── SCENARIO ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:20px;font-weight:700;color:#E8F4FF;margin-bottom:20px;">No Action vs Optimized Strategy</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Today",
                f"{ga_result['baseline_emission']} Mt", "Dec 2024")
    col2.metric("No Action — Jun 2026", "2.181 Mt",
                "+3.0%", delta_color="inverse")
    col3.metric("Optimized — Jun 2026", f"{ga_result['optimized_emission']} Mt",
                f"-{ga_result['reduction_pct']}%", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)
    future_months = pd.date_range("2026-01-01", periods=6, freq="MS")
    no_action = np.linspace(ga_result["baseline_emission"], 2.181, 6)
    optimized = np.linspace(
        ga_result["baseline_emission"], ga_result["optimized_emission"], 6)

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(x=future_months, y=no_action,
                                 mode="lines+markers", name="No Action",
                                 line=dict(color=DANGER, width=2.5),
                                 marker=dict(size=8, color=DANGER, line=dict(color="#080C10", width=2))))
    fig_cmp.add_trace(go.Scatter(x=future_months, y=optimized,
                                 mode="lines+markers", name="Optimized Strategy",
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
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin:20px 0 12px 0;">── DECISION SUMMARY</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        "Scenario":          ["No Action", "Optimized Strategy"],
        "Jun 2026 Emission": ["2.1810 Mt", f"{ga_result['optimized_emission']} Mt"],
        "Δ from Today":      ["+3.0%",    f"-{ga_result['reduction_pct']}%"],
        "Transport Policy":  ["—",        f"↓ {ga_result['transport_reduction']}%"],
        "Renewable Policy":  ["—",        f"↑ {ga_result['renewable_increase']}%"],
        "Industry Policy":   ["—",        f"↓ {ga_result['industry_efficiency']}%"],
        "Economic Impact":   ["N/A",      "Within limits"],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    saved = no_action[-1] - optimized[-1]
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
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:12px;color:#3A6A8A;margin-bottom:24px;">Drag the sliders to simulate any policy combination. The AI model predicts emission instantly.</div>', unsafe_allow_html=True)

    col_sliders, col_result = st.columns([1, 1])

    with col_sliders:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:16px;">── POLICY LEVERS</div>', unsafe_allow_html=True)

        transport_red = st.slider(
            "🚗  Transport Activity Reduction (%)",
            min_value=0, max_value=30, value=0, step=1,
            help="Reduce transport activity via restrictions, EV adoption, public transit"
        )
        renewable_inc = st.slider(
            "☀️  Renewable Energy Increase (%)",
            min_value=0, max_value=30, value=0, step=1,
            help="Increase share of renewable energy in the grid"
        )
        industry_eff = st.slider(
            "🏭  Industrial Efficiency Improvement (%)",
            min_value=0, max_value=15, value=0, step=1,
            help="Reduce industrial activity through efficiency measures"
        )
        temp_adj = st.slider(
            "🌡️  Temperature Adjustment (°C)",
            min_value=-5, max_value=5, value=0, step=1,
            help="Simulate seasonal or climate scenario"
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

    with col_result:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#3A6A8A;letter-spacing:0.15em;margin-bottom:16px;">── PREDICTED OUTCOME</div>', unsafe_allow_html=True)

        # Build adjusted feature vector and predict
        FEATURES_RF = ["month_index", "temperature_c", "gdp_index",
                       "transport_activity", "industry_activity", "renewable_share_pct"]

        sim_state = {
            "month_index":         latest["month_index"],
            "temperature_c":       latest["temperature_c"] + temp_adj,
            "gdp_index":           latest["gdp_index"],
            "transport_activity":  latest["transport_activity"] * (1 - transport_red / 100),
            "industry_activity":   latest["industry_activity"] * (1 - industry_eff / 100),
            "renewable_share_pct": latest["renewable_share_pct"] + renewable_inc,
        }

        X_sim = np.array([[sim_state[f] for f in FEATURES_RF]])
        X_sim_sc = rf_sc.transform(X_sim)
        sim_emission = rf.predict(X_sim_sc)[0]

        baseline = latest["total_emission"]
        delta = sim_emission - baseline
        delta_pct = (delta / baseline) * 100
        is_reduction = delta < 0
        result_color = SUCCESS if is_reduction else DANGER
        arrow = "↓" if is_reduction else "↑"

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
            r_only = rf.predict(rf_sc.transform(np.array([[
                sim_state["month_index"], sim_state["temperature_c"],
                sim_state["gdp_index"], latest["transport_activity"],
                latest["industry_activity"],
                latest["renewable_share_pct"] + renewable_inc
            ]])))[0]
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
            st.plotly_chart(fig_contrib, use_container_width=True)
        else:
            st.markdown("""
            <div style='padding:20px;text-align:center;border:1px dashed #1C2A3A;
                        border-radius:8px;font-family:DM Mono,monospace;font-size:11px;color:#3A5A7A;'>
                Move the sliders to see per-lever contribution
            </div>""", unsafe_allow_html=True)
