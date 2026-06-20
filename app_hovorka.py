import streamlit as st
from hovorka_model import HovorkaModel

st.set_page_config(page_title="Hovorka Virtual Patient", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3, h6 { color: #ffffff; }
    h6 { margin-bottom: 0; font-weight: 400; color: #a0a4b0; }
    .stNumberInput label { color: #c8ccd4; }
    .section-header {
        background: linear-gradient(90deg, #1a1d27, #0e1117);
        border-left: 3px solid #4fc3f7;
        padding: 8px 16px;
        margin: 20px 0 10px 0;
        border-radius: 0 4px 4px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; padding-bottom:4px;'>Hovorka Virtual Patient Model</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#a0a4b0;'>Type 1 Diabetes Glucose-Insulin Simulation (EGP6 Extended Version)</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Patient Parameters ──────────────────────────────────────────────────────
st.markdown("<div class='section-header'><h2 style='margin:0;'>Patient Parameters</h2></div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    BW = st.number_input("Body Weight (kg)", min_value=30.0, max_value=150.0, value=70.0, step=1.0)
with c2:
    u_basal = st.number_input("Basal Insulin Infusion Rate (mU/min)", min_value=1.0, max_value=50.0, value=12.9127, step=0.5, format="%.4f")

# ── Scenario Configuration ──────────────────────────────────────────────────
st.markdown("<div class='section-header'><h2 style='margin:0;'>Scenario Configuration</h2></div>", unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    num_meals = st.number_input("Number of Meals", min_value=1, max_value=5, value=5, step=1)
with c4:
    bolus_duration = st.number_input("Bolus Infusion Duration (minutes)", min_value=1.0, max_value=30.0, value=5.0, step=1.0)

# Dynamic input tables for Meals and Insulin Boluses
st.markdown("<h3 style='margin-top:15px; color:#4fc3f7;'>Meal & Bolus Schedules</h3>", unsafe_allow_html=True)

meal_times = []
meal_durations = []
meal_cho = []
bolus_times = []
bolus_values = []

# Exact reference schedules matching Mam's Table 5 and Table 6
default_times = [480.0, 780.0, 960.0, 1140.0, 1380.0]  # 8:00 AM, 1:00 PM, 4:00 PM, 7:00 PM, 11:00 PM
default_durations = [10.0, 20.0, 10.0, 20.0, 10.0]
default_cho = [124.17, 34.5, 34.5, 76.95, 34.5]
default_bolus = [700.0, 250.0, 200.0, 490.0, 210.0]

header_cols = st.columns([1, 1.5, 1.5, 1.5, 1.5])
header_cols[0].markdown("**Meal #**")
header_cols[1].markdown("**Time (min)**")
header_cols[2].markdown("**Duration (min)**")
header_cols[3].markdown("**Carbs (g)**")
header_cols[4].markdown("**Insulin Bolus (mU/min)**")

for i in range(int(num_meals)):
    m_t = default_times[i] if i < len(default_times) else default_times[-1] + 120.0
    m_d = default_durations[i] if i < len(default_durations) else 10.0
    m_c = default_cho[i] if i < len(default_cho) else 34.5
    b_v = default_bolus[i] if i < len(default_bolus) else 200.0
    
    row_cols = st.columns([1, 1.5, 1.5, 1.5, 1.5])
    row_cols[0].markdown(f"<h5 style='margin:10px 0;'>Meal {i+1}</h5>", unsafe_allow_html=True)
    
    meal_times.append(row_cols[1].number_input(f"Time {i}", min_value=0.0, max_value=1440.0, value=m_t, step=10.0, key=f"mt_{i}", label_visibility="collapsed"))
    meal_durations.append(row_cols[2].number_input(f"Duration {i}", min_value=1.0, max_value=120.0, value=m_d, step=5.0, key=f"md_{i}", label_visibility="collapsed"))
    meal_cho.append(row_cols[3].number_input(f"Carbs {i}", min_value=0.0, max_value=200.0, value=m_c, step=5.0, key=f"mc_{i}", label_visibility="collapsed"))
    
    bolus_times.append(meal_times[-1])
    bolus_values.append(row_cols[4].number_input(f"Bolus {i}", min_value=0.0, max_value=50000.0, value=b_v, step=10.0, key=f"bv_{i}", label_visibility="collapsed"))

st.markdown("<br>", unsafe_allow_html=True)

# ── Run Simulation ───────────────────────────────────────────────────────────
if st.button("▶  Run Simulation", use_container_width=True):
    with st.spinner("Simulating 24-hour glucose-insulin dynamics..."):
        try:
            model = HovorkaModel(BW=BW, u_basal=u_basal)
            t, G_mmol, G_mgdl, I = model.simulate(
                meal_times=meal_times,
                meal_durations=meal_durations,
                meal_cho=meal_cho,
                bolus_times=bolus_times,
                bolus_values=bolus_values,
                bolus_duration=float(bolus_duration)
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'><h2 style='margin:0;'>Results</h2></div>", unsafe_allow_html=True)

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Mean BG", f"{G_mgdl.mean():.1f} mg/dL")
            sc2.metric("Peak BG",  f"{G_mgdl.max():.1f} mg/dL")
            sc3.metric("Min BG",   f"{G_mgdl.min():.1f} mg/dL")
            tir = ((G_mgdl >= 70) & (G_mgdl <= 180)).sum() / len(G_mgdl) * 100
            sc4.metric("Time In Range (70-180)", f"{tir:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)
            fig = model.plot(t, G_mmol, G_mgdl, I)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred during simulation: {str(e)}")
