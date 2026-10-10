import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Mumbai Housing Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        return joblib.load("mumbai_house_model.pkl")
    except Exception:
        return None

model = load_model()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("📍 Property Parameters")
location = st.sidebar.selectbox(
    "Select Suburb / Area",
    ["Andheri West", "Bandra West", "Thane West", "Borivali West", "Colaba", "Juhu", "Powai", "Lower Parel"]
)
area = st.sidebar.slider("Built-up Area (Sq. Ft.)", min_value=300, max_value=5000, value=1000, step=50)
bhk = st.sidebar.selectbox("Configuration (BHK)", [1, 2, 3, 4, 5])
bathrooms = st.sidebar.selectbox("Bathrooms", [1, 2, 3, 4])
property_type = st.sidebar.radio("Property Type", ["Ready to Move", "Under Construction"])

# --- MAIN DASHBOARD ---
st.title("🏙️ Mumbai Real Estate Valuation Engine")
st.markdown("Real-time machine learning inference pipeline designed for Mumbai metropolitan housing trends.")
st.divider()

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("📊 Live Input Summary")
    summary_df = pd.DataFrame({
        "Feature": ["Target Location", "Carpet Area", "Bedrooms", "Bathrooms", "Status"],
        "Value": [location, f"{area} sq.ft.", f"{bhk} BHK", str(bathrooms), property_type]
    })
    st.table(summary_df)
    predict_btn = st.button("🚀 Calculate Estimated Valuation", type="primary", use_container_width=True)

with col2:
    st.subheader("💡 Market Insights")
    st.info("Valuation incorporates spatial footprint, bedroom-to-bathroom ratios, and baseline suburb variance.")
    st.metric(label="Model Engine", value="Random Forest", delta="Active")

# --- PREDICTION EXECUTION ---
if predict_btn:
    st.divider()
    if model is None:
        st.error("Model file 'mumbai_house_model.pkl' not found. Run Train.py first!")
    else:
        try:
            features = np.array([[area, bhk, bathrooms]])
            prediction = model.predict(features)[0]

            st.success("### Valuation Successful")
            m1, m2 = st.columns(2)
            m1.metric(
                label="Estimated Market Price",
                value=f"₹ {prediction:.2f} Lakhs",
                delta=f"₹ {prediction / 100:.2f} Crores"
            )
            m2.metric(
                label="Estimated Rate / Sq.Ft.",
                value=f"₹ {int((prediction * 100000) / area):,}"
            )
        except Exception as e:
            st.error(f"Inference error: {e}")
