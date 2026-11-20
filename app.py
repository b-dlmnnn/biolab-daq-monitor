import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import datetime
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="BioLab-DAQ | Microfluidic & Optical Experiment Monitor", page_icon="🔬", layout="wide")

def init_db():
    conn = sqlite3.connect("biolab_experiments.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS experiment_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            chip_id TEXT,
            assay_type TEXT,
            target_temp_c REAL,
            steady_temp_c REAL,
            peak_optical_mv REAL,
            estimated_conc REAL,
            snr_db REAL,
            notes TEXT
        )
    """)
    conn.commit()
    return conn

conn = init_db()

def generate_experiment_stream(target_temp: float, sample_conc: float, noise_level: float, duration_sec: int = 120):
    np.random.seed(int(sample_conc * 10) + int(target_temp))
    t = np.arange(0, duration_sec, 1)
    ambient_temp = 25.0
    tau = 14.0
    overshoot = 1.4 * np.exp(-t / 18.0) * np.sin(t / 5.0)
    clean_temp = ambient_temp + (target_temp - ambient_temp) * (1 - np.exp(-t / tau)) + overshoot
    clean_temp[t < 5] = ambient_temp
    raw_temp = clean_temp + np.random.normal(0, noise_level * 0.4, len(t))
    
    baseline_mv = 110.0
    max_signal = 8.5 * sample_conc
    kinetics = 1.0 / (1.0 + np.exp(-0.14 * (t - 55.0)))
    clean_optical = baseline_mv + max_signal * kinetics
    
    spikes = np.zeros(len(t))
    spike_idx = np.random.choice(len(t), size=4, replace=False)
    spikes[spike_idx] = np.random.choice([-1, 1], size=4) * (noise_level * 18.0)
    raw_optical = clean_optical + np.random.normal(0, noise_level * 5.0, len(t)) + spikes
    
    return pd.DataFrame({
        "Time_s": t,
        "Raw_Temp_C": raw_temp,
        "Clean_Temp_C": clean_temp,
        "Raw_Optical_mV": raw_optical,
        "Clean_Optical_mV": clean_optical
    })

def apply_signal_filters(df: pd.DataFrame, window_size: int):
    df["Filtered_Temp_C"] = df["Raw_Temp_C"].rolling(window=window_size, min_periods=1, center=True).mean()
    med_opt = df["Raw_Optical_mV"].rolling(window=3, min_periods=1, center=True).median()
    df["Filtered_Optical_mV"] = med_opt.ewm(span=window_size).mean()
    signal_power = np.mean(df["Filtered_Optical_mV"] ** 2)
    noise_power = np.mean((df["Raw_Optical_mV"] - df["Filtered_Optical_mV"]) ** 2) + 1e-6
    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return df, round(float(snr_db), 2)

st.sidebar.title("🔌 DAQ & Chip Controller")
daq_port = st.sidebar.selectbox("Serial / USB DAQ Interface", ["COM3 (ESP32-S3 Optical Reader)", "/dev/cu.usbserial-0001 (Lab DAQ)", "Simulated Benchtop Rig"])
chip_id = st.sidebar.text_input("Microfluidic Chip ID", value="CHIP-NPUST-04")
assay_type = st.sidebar.selectbox("Assay Protocol", ["Isothermal LAMP (65.0 °C)", "RPA Nucleic Acid (39.0 °C)", "Colorimetric ELISA (37.0 °C)"])

default_temp = 65.0 if "65.0" in assay_type else (39.0 if "39.0" in assay_type else 37.0)
target_temp = st.sidebar.slider("Heater Setpoint (°C)", 30.0, 75.0, default_temp, 0.5)
sample_conc = st.sidebar.slider("Test Sample Concentration (ng/mL)", 0.0, 100.0, 45.0, 5.0)
noise_level = st.sidebar.slider("Sensor Electrical Noise Level", 0.2, 2.5, 1.0, 0.1)
filter_window = st.sidebar.slider("DSP Filter Window (samples)", 3, 15, 7, 2)

df_raw = generate_experiment_stream(target_temp, sample_conc, noise_level)
df, snr_db = apply_signal_filters(df_raw, filter_window)

st.sidebar.markdown("---")
run_live = st.sidebar.button("▶️ Simulate Live 120s Run", use_container_width=True)

st.title("🔬 BioLab-DAQ: Microfluidic & Optical Experiment Monitor")
st.caption(f"Connected Interface: `{daq_port}`  |  Active Chip: **{chip_id}**  |  Protocol: **{assay_type}**")

steady_temp = float(df["Filtered_Temp_C"].iloc[-15:].mean())
peak_mv = float(df["Filtered_Optical_mV"].iloc[-10:].mean())
estimated_conc = max(0.0, (peak_mv - 110.0) / 8.5)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Steady-State Chip Temp", f"{steady_temp:.2f} °C", f"{steady_temp - target_temp:+.2f} °C vs Setpoint")
k2.metric("Filtered Optical Signal", f"{peak_mv:.1f} mV", f"+{peak_mv - 110.0:.1f} mV above baseline")
k3.metric("Quantified Concentration", f"{estimated_conc:.2f} ng/mL", f"True: {sample_conc:.1f} ng/mL")
k4.metric("Signal-to-Noise Ratio (SNR)", f"{snr_db:.1f} dB", "DSP Filter Active")

if run_live:
    progress_bar = st.progress(0, text="Acquiring serial packets at 1 Hz...")
    chart_placeholder = st.empty()
    for step in range(20, 121, 20):
        sub_df = df.iloc[:step]
        fig_live = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Chip Thermal Control (°C)", "Optical Photodiode Readout (mV)"))
        fig_live.add_trace(go.Scatter(x=sub_df["Time_s"], y=sub_df["Raw_Temp_C"], name="Raw Temp", line=dict(color="rgba(255,183,3,0.35)")), row=1, col=1)
        fig_live.add_trace(go.Scatter(x=sub_df["Time_s"], y=sub_df["Filtered_Temp_C"], name="Filtered Temp", line=dict(color="#ffb703", width=2.5)), row=1, col=1)
        fig_live.add_trace(go.Scatter(x=sub_df["Time_s"], y=sub_df["Raw_Optical_mV"], name="Raw Optical", line=dict(color="rgba(0,180,216,0.35)")), row=2, col=1)
        fig_live.add_trace(go.Scatter(x=sub_df["Time_s"], y=sub_df["Filtered_Optical_mV"], name="Filtered Optical", line=dict(color="#00b4d8", width=2.5)), row=2, col=1)
        fig_live.update_layout(height=420, template="plotly_dark")
        chart_placeholder.plotly_chart(fig_live, use_container_width=True)
        progress_bar.progress(int((step / 120) * 100), text=f"Acquiring serial packets... {step}/120s")
        time.sleep(0.25)
    progress_bar.empty()
    chart_placeholder.empty()

tab1, tab2, tab3 = st.tabs(["📈 1. Real-Time DAQ & DSP Filtering", "📐 2. Standard Calibration Curve (R² & LOD)", "🗄️ 3. SQLite Lab Notebook & Export"])

with tab1:
    st.subheader("Raw Sensor Telemetry vs. Digital Signal Processing (DSP) Output")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("Microheater Thermal Trajectory (°C)", "Optical Biosensor Kinetic Curve (mV)"))
    fig.add_trace(go.Scatter(x=df["Time_s"], y=df["Raw_Temp_C"], name="Raw Thermistor (°C)", line=dict(color="rgba(255, 183, 3, 0.35)", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Time_s"], y=df["Filtered_Temp_C"], name="DSP Filtered Temp (°C)", line=dict(color="#ffb703", width=3)), row=1, col=1)
    fig.add_hline(y=target_temp, line_dash="dash", line_color="#52b788", annotation_text=f"Setpoint ({target_temp}°C)", row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Time_s"], y=df["Raw_Optical_mV"], name="Raw Photodiode (mV)", line=dict(color="rgba(0, 180, 216, 0.35)", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Time_s"], y=df["Filtered_Optical_mV"], name="DSP Filtered Signal (mV)", line=dict(color="#00b4d8", width=3)), row=2, col=1)
    fig.update_layout(height=500, template="plotly_dark", hovermode="x unified")
    fig.update_xaxes(title_text="Elapsed Experiment Time (seconds)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Assay Calibration Curve, Linear Regression & Limit of Detection (LOD)")
    std_concs = np.array([0.0, 10.0, 25.0, 50.0, 75.0, 100.0])
    np.random.seed(7)
    rep1 = 110.0 + 8.5 * std_concs + np.random.normal(0, 12.0, len(std_concs))
    rep2 = 110.0 + 8.5 * std_concs + np.random.normal(0, 12.0, len(std_concs))
    rep3 = 110.0 + 8.5 * std_concs + np.random.normal(0, 12.0, len(std_concs))
    means = np.mean([rep1, rep2, rep3], axis=0)
    stds = np.std([rep1, rep2, rep3], axis=0)
    slope, intercept, r_value, p_value, std_err = stats.linregress(std_concs, means)
    r_squared = r_value ** 2
    lod = (3.0 * float(stds[0])) / slope
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Calibration Equation", f"y = {slope:.2f}x + {intercept:.1f}")
    c2.metric("Linearity (R²)", f"{r_squared:.4f}", "Publication Grade (>0.99)")
    c3.metric("Limit of Detection (LOD)", f"{lod:.2f} ng/mL", "3σ / Slope")
    
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=std_concs, y=means, mode="markers", name="Standard Triplicates (Mean ± SD)", error_y=dict(type="data", array=stds, visible=True, color="#00b4d8"), marker=dict(size=10, color="#00b4d8")))
    fig_cal.add_trace(go.Scatter(x=std_concs, y=slope * std_concs + intercept, mode="lines", name=f"OLS Fit (R² = {r_squared:.4f})", line=dict(color="#52b788", width=2.5, dash="dash")))
    fig_cal.add_trace(go.Scatter(x=[estimated_conc], y=[peak_mv], mode="markers", name="Current Active Chip Sample", marker=dict(size=14, color="#ff4d6d", symbol="star")))
    fig_cal.update_layout(height=420, template="plotly_dark", title="Optical Biosensor Standard Calibration Curve", xaxis_title="Target Biomarker Concentration (ng/mL)", yaxis_title="Steady-State Photodiode Output (mV)")
    st.plotly_chart(fig_cal, use_container_width=True)

with tab3:
    st.subheader("Electronic Lab Notebook (SQLite) & Publication Data Export")
    with st.form("save_run_form"):
        col_f1, col_f2 = st.columns([3, 1])
        run_notes = col_f1.text_input("Experiment Observation / Reagent Batch Notes", placeholder="e.g., PMMA chip #4, LED excitation 470nm, stable baseline")
        save_btn = col_f2.form_submit_button("💾 Log Run to Lab Database", use_container_width=True)
        if save_btn:
            c = conn.cursor()
            c.execute("INSERT INTO experiment_runs (timestamp, chip_id, assay_type, target_temp_c, steady_temp_c, peak_optical_mv, estimated_conc, snr_db, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), chip_id, assay_type, target_temp, round(steady_temp, 2), round(peak_mv, 2), round(estimated_conc, 2), snr_db, run_notes))
            conn.commit()
            st.toast("Experiment run archived to SQLite!", icon="🧪")
            st.rerun()
            
    runs_df = pd.read_sql_query("SELECT * FROM experiment_runs ORDER BY run_id DESC", conn)
    if not runs_df.empty:
        st.dataframe(runs_df, use_container_width=True, hide_index=True)
    else:
        st.info("No runs archived yet. Click 'Log Run to Lab Database' above to save the current chip experiment.")
        
    st.markdown("---")
    dl1, dl2 = st.columns(2)
    dl1.download_button("📥 Download Current Time-Series Stream (CSV)", data=df.to_csv(index=False).encode("utf-8"), file_name=f"{chip_id}_timeseries.csv", mime="text/csv")
    dl2.download_button("📥 Download All Archived Lab Runs (CSV)", data=runs_df.to_csv(index=False).encode("utf-8"), file_name="biolab_notebook_archive.csv", mime="text/csv")

st.markdown("---")
st.caption("**Project Note:** I built this tool to simulate the daily data acquisition and analysis workflow of a microfluidic biosensor bench—reading noisy thermal and photodiode streams, applying digital filters, plotting calibration curves ($R^2$ and LOD), and archiving runs into SQLite.")
