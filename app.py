import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
import hashlib
import streamlit.components.v1 as components
import random
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Transformer Health Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utility: CSS & Styling ---
def load_custom_css():
    st.markdown("""
        <style>
        .main-header {font-size: 2.5rem; color: #1E88E5; font-weight: 700;}
        .sub-header {font-size: 1.5rem; color: #424242; font-weight: 600;}
        .card {background-color: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);}
        .stButton>button {width: 100%; border-radius: 5px; height: 3em;}
        </style>
    """, unsafe_allow_html=True)

# --- Logic Class: Transformer Engineering Calculations ---
class TransformerLogic:
    
    @staticmethod
    def calculate_score(value, thresholds):
        """Generic scoring function based on thresholds."""
        for threshold, score in thresholds:
            if value <= threshold:
                return score
        return 1 # Default lowest score (High risk)

    @staticmethod
    def get_abnormalities(dga_data, thresholds):
        abnormalities = []
        checks = [
            (dga_data['C2H2'] > thresholds['acetylene'], "Acetylene above threshold"),
            (dga_data['C2H4'] > thresholds['ethylene'], "Ethylene above threshold"),
            (dga_data['Acidity'] >= thresholds['acidity'], "Acidity above threshold"),
            (dga_data['BDV'] < thresholds['bdv'], "BDV below threshold"),
            (dga_data['Water'] >= thresholds['waterContent'], "Water content above threshold"),
            (dga_data['wrd'] > thresholds['windingResDeviation'], "Winding resistance deviation > threshold"),
            (dga_data['ir'] < thresholds['irValue'], "IR value below threshold"),
            (dga_data['olk'] >= thresholds['oilLeakage'], "Oil leakage above threshold"),
        ]
        
        for condition, message in checks:
            if condition:
                abnormalities.append(message)
                
        # Complex age logic
        if dga_data['age'] >= thresholds['age'] and dga_data['Furan'] >= thresholds['furan']:
            abnormalities.append("Critical Age & Furan levels")
        if dga_data['age'] < thresholds['age'] and dga_data['Furan'] >= 6000:
            abnormalities.append("Premature aging (High Furan for Age)")
            
        return abnormalities

    @staticmethod
    def evaluate_dga_status(dga):
        """Key Gas Method Evaluation."""
        try:
            ratios = {
                "H2": dga["H2"] / 60,
                "CH4": dga["CH4"] / 40,
                "C2H6": dga["C2H6"] / 50,
                "C2H4": dga["C2H4"] / 60,
                "C2H2": dga["C2H2"] / 1
            }
            max_gas = max(ratios, key=ratios.get)
            max_val = ratios[max_gas]

            if max_val == 0: return "No Fault detected (Zero Gas)"
            
            mapping = {
                "H2": "Partial Discharges in voids",
                "CH4": "Sparking < 150°C",
                "C2H6": "Local Overheating 150°C - 300°C",
                "C2H4": "Severe Overheating 300°C - 700°C",
                "C2H2": "Arcing > 700°C"
            }
            return mapping.get(max_gas, "Unknown condition")
        except ZeroDivisionError:
            return "Invalid Data"

    @staticmethod
    def duval_triangle_python(dga):
        """Python implementation of Duval Triangle Logic."""
        total = dga["CH4"] + dga["C2H2"] + dga["C2H4"]
        if total == 0: return "N/A (Total Gas 0)"

        ch4_p = (dga["CH4"] / total) * 100
        c2h2_p = (dga["C2H2"] / total) * 100
        c2h4_p = (dga["C2H4"] / total) * 100

        if c2h2_p > 29: return "D1: Discharges of high energy (arcing)"
        if c2h4_p > 48: return "T3: Thermal fault > 700°C"
        if ch4_p > 98: return "PD: Partial Discharge"
        if 13 < c2h2_p <= 29 and 10 < c2h4_p <= 48: return "D2: Discharges of low energy (sparking)"
        if 87 < ch4_p <= 98 and c2h2_p <= 13 and c2h4_p <= 10: return "PD: Partial Discharge"
        if 4 < c2h2_p <= 13 and 24 < c2h4_p <= 48 and 33 < ch4_p <= 87: return "T2: Thermal fault 300°C - 700°C"
        if 10 < c2h4_p <= 24 and 33 < ch4_p <= 98 and c2h2_p <= 4: return "T1: Thermal fault < 300°C"
        
        return "Undetermined Fault"

    @staticmethod
    def interpolate_furan_life(fal_ppb, current_age):
        # Table for interpretation
        aging_table = [
            {'fal_ppb': 0, 'pct_life': 100}, {'fal_ppb': 130, 'pct_life': 90},
            {'fal_ppb': 292, 'pct_life': 79}, {'fal_ppb': 654, 'pct_life': 66},
            {'fal_ppb': 1464, 'pct_life': 50}, {'fal_ppb': 1720, 'pct_life': 46},
            {'fal_ppb': 2021, 'pct_life': 42}, {'fal_ppb': 2374, 'pct_life': 38},
            {'fal_ppb': 2789, 'pct_life': 33}, {'fal_ppb': 3277, 'pct_life': 29},
            {'fal_ppb': 3851, 'pct_life': 24}, {'fal_ppb': 4524, 'pct_life': 19},
            {'fal_ppb': 5315, 'pct_life': 13}, {'fal_ppb': 6245, 'pct_life': 7},
            {'fal_ppb': 7377, 'pct_life': 0}
        ]
        
        # Logic
        for i in range(len(aging_table) - 1):
            lower = aging_table[i]
            upper = aging_table[i+1]
            
            if lower['fal_ppb'] <= fal_ppb <= upper['fal_ppb']:
                # Linear interpolation
                ratio = (fal_ppb - lower['fal_ppb']) / (upper['fal_ppb'] - lower['fal_ppb'])
                pct_life = lower['pct_life'] + ratio * (upper['pct_life'] - lower['pct_life'])
                
                estimated_rul_years = (pct_life / 100) * current_age if current_age > 0 else 0
                return pct_life, estimated_rul_years, "Calculated"
                
        if fal_ppb > 7377:
            return 0, 0, "End of Life"
        return 100, current_age, "New"

# --- Auth Manager ---
class AuthManager:
    @staticmethod
    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def check_login(username, password):
        # In a real app, store these in st.secrets
        STORED_USER = "Harsh"
        STORED_PASS_HASH = "37d09321078a5990612495835e5d9566111001a21a804b7b68259b92015a2430" # sha256 of summerintern24
        return username == STORED_USER and AuthManager.hash_password(password) == STORED_PASS_HASH

# --- UI Components ---
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🔐 Login to Analytics Platform")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            # Captcha Logic
            if "captcha" not in st.session_state:
                st.session_state.captcha = {
                    "n1": random.randint(1, 10),
                    "n2": random.randint(1, 10)
                }
            
            captcha_ans = st.text_input(f"Solve: {st.session_state.captcha['n1']} + {st.session_state.captcha['n2']} = ?")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                try:
                    if int(captcha_ans) != (st.session_state.captcha['n1'] + st.session_state.captcha['n2']):
                        st.error("Incorrect Math Captcha.")
                        st.session_state.captcha = {"n1": random.randint(1, 10), "n2": random.randint(1, 10)}
                        return

                    if AuthManager.check_login(username, password):
                        st.session_state["logged_in"] = True
                        st.rerun()
                    else:
                        st.error("Invalid Credentials.")
                except ValueError:
                    st.error("Captcha must be a number.")

# --- Page: Health Index Calculator ---
def page_health_index():
    st.markdown("<h1 class='main-header'>Transformer Health Index</h1>", unsafe_allow_html=True)
    
    # Threshold Definitions (Compact)
    thresholds_db = {
        "HT (≥ 132KV)": {'acetylene': 2.5, 'ethylene': 100, 'acidity': 0.2, 'bdv': 40, 'waterContent': 30, 'windingResDeviation': 8, 'irValue': 500, 'oilLeakage': 3, 'age': 50, 'furan': 3000},
        "LF": {'acetylene': 1.5, 'ethylene': 60, 'acidity': 0.2, 'bdv': 35, 'waterContent': 35, 'windingResDeviation': 2, 'irValue': 100, 'oilLeakage': 3, 'age': 50, 'furan': 3000},
        "LT (< 132KV)": {'acetylene': 3.5, 'ethylene': 150, 'acidity': 0.3, 'bdv': 30, 'waterContent': 50, 'windingResDeviation': 10, 'irValue': 100, 'oilLeakage': 4, 'age': 50, 'furan': 3000}
    }

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("Enter the DGA and Physical Parameters below.")
        trafo_type = st.selectbox("Transformer Type", list(thresholds_db.keys()))
        
        with st.expander("Gas Concentraions (ppm)", expanded=True):
            h2 = st.number_input("H2", 0.0)
            ch4 = st.number_input("CH4", 0.0)
            c2h6 = st.number_input("C2H6", 0.0)
            c2h4 = st.number_input("C2H4", 0.0)
            c2h2 = st.number_input("C2H2", 0.0)
            co = st.number_input("CO", 0.0)
            co2 = st.number_input("CO2", 0.0)

        with st.expander("Oil & Physical Parameters", expanded=False):
            acidity = st.number_input("Acidity (mgKOH/g)", 0.0)
            bdv = st.number_input("BDV (KV)", 0.0, value=50.0)
            water = st.number_input("Water (ppm)", 0.0)
            resistance = st.number_input("Sp. Resistance", 0.0)
            tand = st.number_input("Tan Delta", 0.0)
            furan = st.number_input("Furan", 0.0)
            age = st.number_input("Age (Years)", 0.0, value=10.0)
            ir = st.number_input("IR Value", 0.0, value=1000.0)
            olk = st.number_input("Oil Leakage Scale (1-5)", 0.0)
            wrd = st.number_input("Winding Res Dev", 0.0)

    # Calculation Button
    if col1.button("Calculate Health Index"):
        # Data dictionary
        dga_data = {
            'H2': h2, 'CH4': ch4, 'C2H6': c2h6, 'C2H4': c2h4, 'C2H2': c2h2, 
            'CO': co, 'CO2': co2, 'Acidity': acidity, 'Resistance': resistance, 
            'BDV': bdv, 'Water': water, 'tand': tand, 'Furan': furan, 
            'age': age, 'ir': ir, 'olk': olk, 'wrd': wrd
        }
        
        # Scoring Logic (Condensed for readability)
        tl = TransformerLogic()
        scores = [
            tl.calculate_score(c2h2, [(0, 6), (2, 5), (5, 4), (10, 3), (15, 2)]),
            tl.calculate_score(c2h4, [(5, 6), (20, 5), (40, 4), (60, 3), (75, 2)]),
            tl.calculate_score(h2, [(40, 6), (60, 5), (70, 4), (75, 3), (80, 2)]),
            tl.calculate_score(ch4, [(10, 6), (25, 5), (40, 4), (55, 3), (60, 2)]),
            tl.calculate_score(c2h6, [(15, 6), (30, 5), (40, 4), (55, 3), (60, 2)]),
            tl.calculate_score(co, [(200, 6), (350, 5), (540, 4), (650, 3), (700, 2)]),
            tl.calculate_score(co2, [(3000, 6), (4500, 5), (5100, 4), (6000, 3), (6500, 2)]),
            tl.calculate_score(acidity, [(0.04, 4), (0.1, 3), (0.15, 2)]),
            tl.calculate_score(resistance, [(0.09, 1), (0.5, 2), (1, 3), (999999, 4)]),
            tl.calculate_score(bdv, [(35, 1), (47, 2), (51, 3), (999999, 4)]),
            tl.calculate_score(water, [(20, 4), (25, 3), (30, 2)]),
            tl.calculate_score(tand, [(0.1, 4), (0.5, 3), (1.1, 2)]),
            tl.calculate_score(furan, [(800, 5), (1500, 4), (3000, 3), (6000, 2)]),
            tl.calculate_score(age, [(10, 5), (20, 4), (35, 3), (50, 2)]),
            tl.calculate_score(ir, [(50, 1), (99, 2), (500, 3), (1000, 4), (999999, 4)]),
            tl.calculate_score(olk, [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2)]),
            tl.calculate_score(wrd, [(3, 5), (5, 4), (8, 3), (10, 2)])
        ]
        
        weights = [6, 4, 3, 1, 1, 1, 1, 5, 4, 4, 4, 2, 5, 5, 2, 1, 3]
        final_score = sum([s*w for s, w in zip(scores, weights)])
        health_index = final_score * 100 / 257
        
        with col2:
            st.markdown("### Result Analysis")
            
            # Gauge Chart using Plotly
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = health_index,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Health Index Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1E88E5"},
                    'steps': [
                        {'range': [0, 30], 'color': "#FF5252"},
                        {'range': [30, 70], 'color': "#FFC107"},
                        {'range': [70, 100], 'color': "#66BB6A"}],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Abnormalities
            abnormalities = tl.get_abnormalities(dga_data, thresholds_db[trafo_type])
            if abnormalities:
                st.error(f"Found {len(abnormalities)} Issues:")
                for ab in abnormalities:
                    st.write(f"⚠️ {ab}")
            else:
                st.success("No major abnormalities detected.")

# --- Page: ML Analysis ---
def page_ml_analysis():
    st.markdown("<h1 class='main-header'>ML-Powered RUL Prediction</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Training Data (CSV)", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())
        
        col1, col2 = st.columns(2)
        with col1:
            # Interactive Scatter
            x_axis = st.selectbox("X-Axis", data.columns)
            y_axis = st.selectbox("Y-Axis", data.columns, index=len(data.columns)-1)
            fig = px.scatter(data, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Histogram
            hist_col = st.selectbox("Distribution of", data.columns)
            fig2 = px.histogram(data, x=hist_col, nbins=20, title=f"Distribution of {hist_col}")
            st.plotly_chart(fig2, use_container_width=True)
            
        if st.button("Train Models (LazyPredict)"):
            if "Health Indx" in data.columns:
                with st.spinner("Training models... This might take a minute."):
                    X = data.drop(columns=["Health Indx"], errors='ignore').select_dtypes(include=[np.number])
                    y = data["Health Indx"]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
                    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
                    
                    st.session_state['models'] = reg
                    st.session_state['best_model_name'] = models.index[0]
                    
                    st.success(f"Training Complete! Best Model: {models.index[0]}")
                    st.dataframe(models.style.highlight_max(axis=0))
            else:
                st.error("Dataset must contain 'Health Indx' column.")

# --- Page: Furan Analysis ---
def page_furan_analysis():
    st.markdown("<h1 class='main-header'>Furan Analysis & RUL</h1>", unsafe_allow_html=True)
    
    mode = st.radio("Input Mode", ["Manual Entry", "CSV Trend Analysis"])
    
    if mode == "Manual Entry":
        fal = st.number_input("2FAL (ppb)", 0, 10000, step=10)
        age = st.number_input("Current Age (Years)", 0, 100, value=10)
        
        if st.button("Analyze Furan"):
            pct, rul, status = TransformerLogic.interpolate_furan_life(fal, age)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Est. Remaining Life %", f"{pct:.1f}%")
            col2.metric("Est. RUL (Years)", f"{rul:.1f} yrs")
            col3.metric("Interpretation", status)
            
            # Visual Bar
            st.progress(int(pct))
            
    elif mode == "CSV Trend Analysis":
        st.info("Upload CSV with 'Date' and 'Furan' columns.")
        f = st.file_uploader("Upload Furan History", type="csv")
        if f:
            df = pd.read_csv(f)
            # Basic cleaning
            cols_lower = [c.lower() for c in df.columns]
            if 'date' in cols_lower:
                date_col = df.columns[cols_lower.index('date')]
                df[date_col] = pd.to_datetime(df[date_col])
                
                val_col = [c for c in df.columns if "value" in c.lower() or "furan" in c.lower()][0]
                
                # Plotly Time Series
                fig = px.line(df, x=date_col, y=val_col, markers=True, title="Furan Trend Over Time")
                
                # Add Colored Zones
                fig.add_hrect(y0=0, y1=654, fillcolor="green", opacity=0.1, annotation_text="Normal")
                fig.add_hrect(y0=654, y1=1464, fillcolor="yellow", opacity=0.1, annotation_text="Accelerated")
                fig.add_hrect(y0=1464, y1=2374, fillcolor="orange", opacity=0.1, annotation_text="Excessive")
                fig.add_hrect(y0=2374, y1=10000, fillcolor="red", opacity=0.1, annotation_text="Danger")
                
                st.plotly_chart(fig, use_container_width=True)

# --- Page: DGA Analysis (Batch) ---
def page_dga_batch():
    st.markdown("<h1 class='main-header'>Batch DGA Analysis</h1>", unsafe_allow_html=True)
    f = st.file_uploader("Upload DGA File", type="csv")
    
    if f:
        df = pd.read_csv(f)
        required_cols = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
        
        # Check columns
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return

        results = []
        for idx, row in df.iterrows():
            status = TransformerLogic.evaluate_dga_status(row)
            duval = TransformerLogic.duval_triangle_python(row)
            results.append({"Row": idx, "Key Gas Status": status, "Duval Status": duval})
            
        res_df = pd.DataFrame(results)
        final = pd.concat([df, res_df], axis=1)
        
        st.write(final)
        st.download_button("Download Results", final.to_csv().encode('utf-8'), "dga_analyzed.csv", "text/csv")

# --- Page: Duval Triangle (Component) ---
def page_duval_triangle():
    st.markdown("<h1 class='main-header'>Duval's Triangle</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### Inputs (ppm)")
        ch4 = st.number_input("CH4", 0.0)
        c2h2 = st.number_input("C2H2", 0.0)
        c2h4 = st.number_input("C2H4", 0.0)
        
        # Python Calculation
        dga_dummy = {"CH4": ch4, "C2H2": c2h2, "C2H4": c2h4}
        result = TransformerLogic.duval_triangle_python(dga_dummy)
        st.success(f"Python Logic Result: {result}")

    with col2:
        # Reusing your existing HTML logic but injected cleaner
        # Note: In a production app, use a Plotly ternary chart instead of raw HTML/JS/Canvas
        
        # Let's generate a Plotly Ternary chart which is NATIVE to Streamlit and better than Canvas
        
        total = ch4 + c2h2 + c2h4
        if total > 0:
            ch4_p = (ch4/total)*100
            c2h2_p = (c2h2/total)*100
            c2h4_p = (c2h4/total)*100
            
            fig = go.Figure(go.Scatterternary({
                'mode': 'markers',
                'a': [ch4_p], # CH4 is usually top
                'b': [c2h4_p], # C2H4 right
                'c': [c2h2_p], # C2H2 left (Map these according to Duval definitions)
                'text': [f"Fault: {result}"],
                'marker': {'color': 'red', 'size': 14}
            }))
            
            fig.update_layout({
                'ternary': {
                    'sum': 100,
                    'aaxis': {'title': 'CH4 %', 'min': 0.01, 'linewidth':2, 'ticks': 'outside'},
                    'baxis': {'title': 'C2H4 %', 'min': 0.01, 'linewidth':2, 'ticks': 'outside'},
                    'caxis': {'title': 'C2H2 %', 'min': 0.01, 'linewidth':2, 'ticks': 'outside'}
                },
                'title': 'Duval Triangle Visualization (Interactive)'
            })
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Enter values to see the Triangle.")

# --- Main Router ---
def main():
    load_custom_css()
    
    # Session State Init
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # Login Check
    if not st.session_state["logged_in"]:
        login_page()
        return

    # Sidebar Logic
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=100) # Generic icon
        st.title("Analyzer Pro")
        st.write(f"Welcome, User")
        
        page = st.radio("Navigate", [
            "Home & Stats",
            "Health Index Calc", 
            "ML Prediction", 
            "Furan Analysis",
            "Duval Triangle"
        ])
        
        st.markdown("---")
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.rerun()

    # Routing
    if page == "Home & Stats":
        page_dga_batch() # Using batch analysis as home for now
    elif page == "Health Index Calc":
        page_health_index()
    elif page == "ML Prediction":
        page_ml_analysis()
    elif page == "Furan Analysis":
        page_furan_analysis()
    elif page == "Duval Triangle":
        page_duval_triangle()

if __name__ == "__main__":
    main()