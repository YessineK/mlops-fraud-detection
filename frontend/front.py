import streamlit as st
import requests
import pandas as pd
import json
import time

import os

# Configuration - detect Docker environment
# In Docker, use service name; locally use localhost
API_URL = os.getenv("API_URL", "http://backend:8000" if os.path.exists("/.dockerenv") else "http://127.0.0.1:8000")

# Page Config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        color: #000000;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border-color: #45a049;
        color: white;
    }
    .metric-card {
        background-color: white;
        color: #000000;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: inherit;
    }
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, .main p, .main li, .main label {
        color: #000000 !important;
    }
    .main .stMarkdown {
        color: #000000 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .safe {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .fraud {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked--v1.png", width=80)
    st.title("Fraud Detection")
    st.markdown("---")
    
    page = st.radio("Navigation", ["🏠 Home", "🔍 Single Prediction", "📂 Batch Prediction"])
    
    st.markdown("---")
    st.subheader("System Status")
    
    # Check API Health
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            if status == "ok":
                st.success("API Online")
            else:
                st.warning(f"API Status: {status}")
            
            # Model Info
            if data.get("model_loaded"):
                st.info("Model Loaded")
            else:
                st.error("Model Not Loaded")
        else:
            st.error("API Error")
    except:
        st.error("API Offline")
        
    st.markdown("---")
    st.caption("MLOps Project 2025")

# Home Page
if page == "🏠 Home":
    st.title("🛡️ Banking Fraud Detection System")
    st.markdown("### Welcome to the MLOps Fraud Detection Dashboard")
    
    st.markdown("""
    This system uses advanced machine learning models to detect fraudulent transactions in real-time.
    
    **Features:**
    - **Real-time Analysis**: Instant fraud detection for individual transactions.
    - **Batch Processing**: Upload CSV files for bulk analysis.
    - **Explainable AI**: Understand why a transaction was flagged.
    """)
    
    # Fetch Model Info
    try:
        response = requests.get(f"{API_URL}/model-info")
        if response.status_code == 200:
            info = response.json()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Model</h3>
                    <p style="font-size: 1.2em;">{info.get('name', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Version/Source</h3>
                    <p style="font-size: 1.2em;">{info.get('version', info.get('source', 'Unknown'))}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Last Loaded</h3>
                    <p style="font-size: 1.2em;">{info.get('loaded_at', 'Unknown').split('T')[0]}</p>
                </div>
                """, unsafe_allow_html=True)
                
            if 'metadata' in info and info['metadata']:
                st.markdown("### Performance Metrics")
                metrics = info['metadata'].get('metrics', {})
                if metrics:
                    m_cols = st.columns(len(metrics))
                    for i, (k, v) in enumerate(metrics.items()):
                        with m_cols[i]:
                            st.metric(label=k.upper(), value=f"{v:.4f}")
    except Exception as e:
        st.warning("Could not fetch model details. Ensure the API is running.")

# Single Prediction Page
elif page == "🔍 Single Prediction":
    st.title("🔍 Analyze Transaction")
    st.markdown("Enter transaction details below to check for fraud probability.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amt = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=10.0)
            category = st.selectbox("Category", [
                "grocery_pos", "entertainment", "shopping_pos", "misc_pos", "shopping_net",
                "gas_transport", "misc_net", "grocery_net", "food_dining", "health_fitness",
                "kids_pets", "home", "personal_care", "travel"
            ])
            gender = st.selectbox("Gender", ["M", "F"])
            state = st.text_input("State", value="NY")
            zip_code = st.number_input("ZIP Code", min_value=0, value=10001, step=1)
            city_pop = st.number_input("City Population", min_value=0, value=50000, step=1000)
            job = st.text_input("Job", value="Developer")
            
        with col2:
            trans_date = st.date_input("Transaction Date")
            trans_time = st.time_input("Transaction Time")
            dob = st.date_input("Date of Birth", value=pd.to_datetime("1990-01-01"))
            
            # Location details (simplified for UI, could be map picker)
            st.markdown("###### Location Coordinates")
            c1, c2 = st.columns(2)
            lat = c1.number_input("Lat", value=40.7128, format="%.4f")
            long = c2.number_input("Long", value=-74.0060, format="%.4f")
            
            st.markdown("###### Merchant Coordinates")
            c3, c4 = st.columns(2)
            merch_lat = c3.number_input("Merch Lat", value=40.7200, format="%.4f")
            merch_long = c4.number_input("Merch Long", value=-74.0100, format="%.4f")
            
        submit = st.form_submit_button("Analyze Transaction")
        
        if submit:
            # Prepare payload
            dt_str = f"{trans_date} {trans_time}"
            payload = {
                "trans_date_trans_time": str(dt_str),
                "amt": amt,
                "category": category,
                "gender": gender,
                "state": state,
                "zip": zip_code,
                "city_pop": city_pop,
                "job": job,
                "dob": str(dob),
                "lat": lat,
                "long": long,
                "merch_lat": merch_lat,
                "merch_long": merch_long
            }
            
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result["predictions"][0]
                        
                        if prediction == 1:
                            st.markdown("""
                            <div class="prediction-card fraud">
                                ⚠️ FRAUD DETECTED
                            </div>
                            """, unsafe_allow_html=True)
                            st.error("This transaction has been flagged as potentially fraudulent.")
                        else:
                            st.markdown("""
                            <div class="prediction-card safe">
                                ✅ LEGITIMATE TRANSACTION
                            </div>
                            """, unsafe_allow_html=True)
                            st.success("This transaction appears to be safe.")
                            
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

# Batch Prediction Page
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Analysis")
    st.markdown("Upload a CSV file containing multiple transactions for bulk analysis.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.info("File uploaded successfully. Click below to process.")
        
        # Preview
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        if st.button("Process Batch"):
            with st.spinner("Processing batch..."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                    
                    response = requests.post(f"{API_URL}/predictCSV", files=files)
                    
                    if response.status_code == 200:
                        st.success("Processing Complete!")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=response.content,
                            file_name=f"predictions_{uploaded_file.name}",
                            mime="text/csv"
                        )
                        
                        # Show preview of results
                        # We need to read the content back into a DF to show it
                        from io import BytesIO
                        result_df = pd.read_csv(BytesIO(response.content))
                        
                        st.subheader("Results Preview")
                        
                        # Highlight frauds
                        def highlight_fraud(val):
                            color = '#ffcdd2' if val == 1 else '#c8e6c9'
                            return f'background-color: {color}'
                            
                        st.dataframe(result_df.style.map(highlight_fraud, subset=['predict']))
                        
                        # Stats
                        n_fraud = result_df['predict'].sum()
                        st.metric("Frauds Detected", f"{n_fraud} / {len(result_df)}")
                        
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")