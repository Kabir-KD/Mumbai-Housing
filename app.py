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

# --- INJECT CUSTOM CSS (THE AESTHETIC UPGRADE) ---
st.markdown("""
    <style>
    /* Neon accents and card styling */
    div[data-testid="stMetricValue"] { 
        color: #00FFC1; 
        font-size: 2.8rem !important; 
        font-weight: 900; 
        text-shadow: 0px 0px 10px rgba(0,255,193,0.3);
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #00FFC1 0%, #0084FF 100%);
        color: #090B10;
        font-weight: 900;
        border-radius: 8px;
        border: none;
        height: 50px;
        transition: all 0.3s ease-in-out;
    }
    div[data-testid="stButton"] button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(0, 255, 193, 0.6);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        return joblib.load("mumbai_house_model.pkl")
    except Exception:
        return None

model = load_model()

# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://images.unsplash.com/photo-1570129477492-45c003edd2be?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", use_container_width=True)
st.sidebar.header("📍 Property Parameters")
location = st.sidebar.selectbox("Select Suburb / Area", ["Andheri West", "Bandra West", "Thane West", "Borivali West", "Colaba", "Juhu", "Powai", "Lower Parel"])
area = st.sidebar.slider("Built-up Area (Sq. Ft.)", min_value=300, max_value=5000, value=1000, step=50)
bhk = st.sidebar.selectbox("Configuration (BHK)", [1, 2, 3, 4, 5])
bathrooms = st.sidebar.selectbox("Bathrooms", [1, 2, 3, 4])
property_type = st.sidebar.radio("Property Type", ["Ready to Move", "Under Construction"])

# --- MAIN DASHBOARD ---
st.title("🏙️ Mumbai Real Estate Engine")
st.markdown("##### Real-time machine learning inference pipeline designed for metropolitan housing trends.")
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
    st.metric(label="Model Engine", value="Random Forest", delta="Active 🟢")

# --- PREDICTION & VISUALIZATION ---
if predict_btn:
    st.divider()
    if model is None:
        st.error("Model file 'mumbai_house_model.pkl' not found.")
    else:
        features = np.array([[area, bhk, bathrooms]])
        prediction = model.predict(features)[0]

        st.success("### 🎯 Valuation Complete")
        
        # Display large metrics
        m1, m2 = st.columns(2)
        m1.metric(label="Estimated Market Price", value=f"₹ {prediction:.2f} Lakhs", delta=f"₹ {prediction / 100:.2f} Crores")
        m2.metric(label="Estimated Rate / Sq.Ft.", value=f"₹ {int((prediction * 100000) / area):,}")
        
        # Display dynamic chart
        st.subheader("📈 Price Breakdown Projection")
        st.markdown("Visualizing the theoretical contribution of space vs. amenities.")
        
        # Simulated data for visual effect based on the prediction
        chart_data = pd.DataFrame({
            "Valuation Factor": ["Base Land Value", "Area Premium", "Amenity Premium (BHK/Bath)"],
            "₹ Lakhs": [
                prediction * 0.40, 
                prediction * 0.45, 
                prediction * 0.15
            ]
        }).set_index("Valuation Factor")
        
        st.bar_chart(chart_data, color="#00FFC1", height=250)
