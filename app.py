import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. PAGE CONFIGURATION (Must be first)
st.set_page_config(
    page_title="Mumbai Housing Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. APPLE-STYLE MINIMALIST CSS
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    div[data-testid="metric-container"] {
        background-color: #1c1c1e;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #333336;
    }
    div[data-testid="stMetricValue"] { 
        font-size: 2.2rem !important; 
        font-weight: 600; 
        color: #f5f5f7;
    }
    div[data-testid="stMetricLabel"] {
        color: #86868b;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.85rem;
    }
    div[data-testid="stButton"] button {
        background-color: #ffffff;
        color: #000000;
        border-radius: 24px;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    div[data-testid="stButton"] button:hover {
        transform: scale(1.02);
        background-color: #f5f5f7;
    }
    </style>
""", unsafe_allow_html=True)

# 3. BASE DATA & MULTIPLIERS
LOCATION_WEIGHTS = {
    "Colaba (South Mumbai)": 2.45,
    "Lower Parel (South Central)": 2.10,
    "Bandra West (Western Suburbs)": 2.20,
    "Juhu (Western Suburbs)": 2.05,
    "Powai (Central Suburbs)": 1.45,
    "Andheri West (Western Suburbs)": 1.35,
    "Borivali West (Western Suburbs)": 1.00,
    "Thane West (Extended MMR)": 0.65,
    "Navi Mumbai (Vashi)": 0.75
}

# 4. LOAD MODEL
@st.cache_resource
def load_model():
    try:
        return joblib.load("mumbai_house_model.pkl")
    except Exception:
        return None

model = load_model()

# 5. HEADER
st.title("Metropolitan Real Estate Engine.")
st.markdown("Predictive analytics and multivariate regression for Mumbai housing markets.")
st.divider()

# 6. TABS
tab1, tab2, tab3, tab4 = st.tabs(["Valuation", "Market Analysis", "Benchmarks", "Architecture"])

# --- TAB 1: VALUATION ---
with tab1:
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Property Parameters")
        location = st.selectbox("Micro-Market", list(LOCATION_WEIGHTS.keys()))
        area = st.slider("Carpet Area (Sq. Ft.)", min_value=350, max_value=4500, value=950, step=25)
        bhk = st.selectbox("Configuration (BHK)", [1, 2, 3, 4, 5], index=1)
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
        furnishing = st.radio("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully Furnished"], horizontal=True)
        
        predict_btn = st.button("Calculate Valuation", type="primary", use_container_width=True)

    with col2:
        st.subheader("Market Inference")
        if predict_btn:
            if model is None:
                st.error("Model state `mumbai_house_model.pkl` not detected. Ensure Train.py was executed.")
            else:
                features = np.array([[area, bhk, bathrooms]])
                base_prediction = model.predict(features)[0]
                
                furnish_mult = 1.0 if furnishing == "Unfurnished" else (1.08 if furnishing == "Semi-Furnished" else 1.15)
                final_val = base_prediction * LOCATION_WEIGHTS[location] * furnish_mult
                rate_per_sqft = (final_val * 100000) / area

                m1, m2 = st.columns(2)
                m1.metric("Estimated Market Value", f"₹ {final_val/100:.2f} Cr", f"₹ {final_val:.2f} Lakhs")
                m2.metric("Effective Rate / Sq.Ft.", f"₹ {int(rate_per_sqft):,}", f"{LOCATION_WEIGHTS[location]}x Premium")

                st.markdown("##### Composition Breakdown")
                breakdown_df = pd.DataFrame({
                    "Factor": ["Base Value", "Built-Up Area", "Locality Premium", "Furnishing"],
                    "Allocation (Lakhs)": [final_val * 0.40, final_val * 0.30, final_val * 0.22, final_val * 0.08]
                }).set_index("Factor")
                
                st.bar_chart(breakdown_df, height=250)
        else:
            st.info("Input property specifications and initialize the calculation engine.")

# --- TAB 2: EDA ---
with tab2:
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.table(team_data)
