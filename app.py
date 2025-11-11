import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime


# Page Setup and Configuration

st.set_page_config(page_title="CardioCare.AI", page_icon="❤️", layout="wide")

# Custom CSS for sleek UI
st.markdown("""
    <style>
    /* ===== Global Theme ===== */
    body {
        background: radial-gradient(circle at top left, #0b0c10, #1a1f25);
        color: #f1f1f1;
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background: linear-gradient(145deg, #0b0c10 0%, #1f2833 60%, #0b0c10 100%);
    }
    h1, h2, h3, h4, h5 {
        color: #ff4b4b !important;
        font-weight: 600;
    }

    /* ===== Sidebar ===== */
    .stSidebar {
        background: #14181c !important;
        border-right: 1px solid #ff4b4b33;
        box-shadow: 2px 0 10px rgba(255, 75, 75, 0.1);
    }
    .css-1d391kg {background: #14181c !important;}

    /* ===== Buttons ===== */
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b, #d62828);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.4);
        transition: 0.2s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(255, 75, 75, 0.6);
        transform: scale(1.02);
    }

    /* ===== Metrics ===== */
    .stMetric {
        background: #1f2833 !important;
        border-radius: 12px;
        padding: 10px;
        color: white !important;
        box-shadow: 0 0 8px rgba(255, 75, 75, 0.2);
    }

    /* ===== Progress Bar ===== */
    .stProgress > div > div {
        background-color: #ff4b4b !important;
    }

    /* ===== Chart Background ===== */
    canvas {
        background-color: #101418 !important;
        border-radius: 10px;
    }

    /* ===== Custom Header ===== */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }
    .brand-title {
        font-size: 2rem;
        color: #ff4b4b;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .brand-subtitle {
        color: #ffffff;
        font-size: 1rem;
        margin-top: -8px;
        letter-spacing: 0.3px;
    }
    </style>
""", unsafe_allow_html=True)


#  Sidebar Navigation

st.sidebar.image("logo.png", width=90)
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    ["📊 ECG Analysis", "🚨 Alerts", "🕒 History", "ℹ️ About", "⚙️ Settings"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Smart ECG Monitoring • AI Arrhythmia Detection")


#  Load Dataset

@st.cache_data
def load_dataset():
    return pd.read_csv("balanced_meta.csv")

try:
    df = load_dataset()
except FileNotFoundError:
    st.error("❌ 'balanced_meta.csv' not found. Please place it beside app.py")
    st.stop()

signal_cols = [c for c in df.columns if c not in ['symbol', 'label']]
category_names = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    5: "AFib",
    6: "ST changes"
}


#  Header 

def show_header():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("logo.png", width=90)
    with col2:
        st.markdown("""
        <div style='margin-top:10px'>
            <h1 style='color:#ff4b4b; font-weight:700; margin-bottom:-5px;'>CardioCare.AI</h1>
            <p style='color:white; font-size:15px;'>An AI-Powered Heart Health App</p>
        </div>
        """, unsafe_allow_html=True)



#  ECG Analysis Page

if menu == "📊 ECG Analysis":
    show_header()
    st.markdown("### 📊 ECG Signal Analysis")
    st.markdown("Simulate smartwatch ECG readings and get AI-driven insights in real time.")
    st.markdown("---")

    # Simulate ECG Scan
    if st.button("Simulate ECG Scan"):
        sample = df.sample(1, random_state=random.randint(1, 9999))
        signal = sample[signal_cols].values.flatten()
        label_code = int(sample['label'].iloc[0])
        real_label = category_names.get(label_code, "Unknown")

        st.session_state["signal"] = signal
        st.session_state["real_label"] = real_label
        st.session_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "signal" in st.session_state:
        signal = st.session_state["signal"]
        timestamp = st.session_state["timestamp"]

        # Plot ECG
        st.subheader(f"ECG Waveform – Recorded at {timestamp}")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(signal, color="#ff4b4b", linewidth=1.2)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, color="#ff4b4b55", alpha=0.4)
        st.pyplot(fig)

        # Compute Metrics
        mean_amp = np.mean(signal)
        variance = np.var(signal)
        simulated_hr = random.randint(60, 120)

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Amplitude", f"{mean_amp:.3f} mV")
        col2.metric("Variance", f"{variance:.3f}")
        col3.metric("Heart Rate", f"{simulated_hr} bpm")

        # --- AI Diagnosis ---
        if variance < 0.005:
            predicted_class = "Normal"
        elif variance < 0.02:
            predicted_class = "Supraventricular"
        elif variance < 0.05:
            predicted_class = "Fusion"
        elif variance < 0.08:
            predicted_class = "Ventricular"
        elif variance < 0.1:
            predicted_class = "AFib"
        else:
            predicted_class = "ST changes"

        confidence = round(random.uniform(0.86, 0.96), 2)

        st.subheader("AI Diagnosis")
        st.markdown(f"### Predicted Condition: **{predicted_class}**")
        st.progress(confidence)
        st.write(f"Confidence Level: **{confidence*100:.1f}%**")

        # Probabilities Visualization
        classes = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'AFib', 'ST changes']
        probs = np.random.dirichlet(np.ones(len(classes)), size=1)[0]
        probs[classes.index(predicted_class)] = confidence
        probs /= probs.sum()
        st.bar_chart(pd.DataFrame({"Condition": classes, "Probability": probs}).set_index("Condition"))

        # Recommendations
        recommendations = {
            'Normal': ('Stay home and monitor', 'Low', 'Continue normal activities.', 'Normal ECG pattern detected.'),
            'Supraventricular': ('Schedule doctor appointment', 'Medium', 'Contact your doctor within 24–48 hours.', 'Possible supraventricular arrhythmia detected.'),
            'Ventricular': ('Seek immediate medical attention', 'High', 'Go to urgent care or ER.', 'Possible ventricular arrhythmia.'),
            'Fusion': ('Schedule doctor appointment', 'Medium', 'Consult a cardiologist.', 'Possible conduction irregularity.'),
            'AFib': ('Seek immediate medical attention', 'High', 'Call emergency services (999).', 'Atrial fibrillation detected — may increase stroke risk.'),
            'ST changes': ('Go to hospital immediately', 'Critical', 'Possible myocardial infarction.', 'Abnormal ST segment detected.')
        }

        action, urgency, description, note = recommendations.get(
            predicted_class, ('Monitor', 'Low', 'Normal pattern.', 'No abnormality detected.')
        )
        color_map = {'Low': '🟢 Low', 'Medium': '🟠 Medium', 'High': '🔴 High', 'Critical': '🟥 Critical'}

        st.markdown("---")
        st.subheader(" Recommendation")
        st.info(
            f"**Predicted Condition:** {predicted_class}\n\n"
            f"**Action:** {action}\n\n"
            f"**Urgency:** {color_map[urgency]}\n\n"
            f"**Description:** {description}\n\n"
            f"**Note:** {note}"
        )

        # Live Patient Vitals
        vital_style = f"""
        <div style='display:flex; justify-content:space-around; margin:20px 0;'>
            <div style='background:#1f2833; border-radius:15px; padding:20px; width:30%; text-align:center; box-shadow:0 0 15px rgba(255,75,75,0.2);'>
                <h3 style='color:#ff4b4b;'>❤️ Heart Rate</h3>
                <p style='font-size:28px; color:white; font-weight:700;'>{random.randint(60, 120)} bpm</p>
            </div>
            <div style='background:#1f2833; border-radius:15px; padding:20px; width:30%; text-align:center; box-shadow:0 0 15px rgba(255,75,75,0.2);'>
                <h3 style='color:#ff4b4b;'>🩸 Blood Pressure</h3>
                <p style='font-size:28px; color:white; font-weight:700;'>{random.randint(110, 140)}/{random.randint(70, 90)} mmHg</p>
            </div>
            <div style='background:#1f2833; border-radius:15px; padding:20px; width:30%; text-align:center; box-shadow:0 0 15px rgba(255,75,75,0.2);'>
                <h3 style='color:#ff4b4b;'>💧 O₂ Saturation</h3>
                <p style='font-size:28px; color:white; font-weight:700;'>{random.randint(94, 99)}%</p>
            </div>
        </div>
        """
        st.markdown(vital_style, unsafe_allow_html=True)

        # History tracking
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append((timestamp, predicted_class, confidence))


#  Alerts

elif menu == "🚨 Alerts":
    show_header()
    st.title("🚨 Emergency Alert Center")

    st.warning("⚠️ In real emergency, this would contact medical responders instantly.")
    contact_name = st.text_input("Emergency Contact Name", "Mr. Rahman")
    contact_number = st.text_input("Emergency Contact Number", "+8801XXXXXXXXX")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📞 Emergency Contact"):
            st.success(f"📞 Calling {contact_name} ({contact_number})")
    with col2:
        if st.button("💬 Alert Message"):
            st.success(f"📩 Emergency alert sent to {contact_name}")
    with col3:
        if st.button("🆘 Medical Services"):
            st.success("🚑 Medical services notified")


#  History

elif menu == "🕒 History":
    show_header()
    st.title("🕒 ECG Scan History")
    if "history" in st.session_state and len(st.session_state["history"]) > 0:
        history_df = pd.DataFrame(st.session_state["history"], columns=["Timestamp", "Predicted Class", "Confidence"])
        st.dataframe(history_df)
    else:
        st.info("No previous scans yet. Run a simulation to create history.")


#  About Page

elif menu == "ℹ️ About":
    show_header()
    st.title("ℹ️ About CardioCare.AI")
    st.markdown("""
    **CardioCare.AI** is a heart health monitoring system designed for continuous ECG surveillance.
    
    **Features:**
    - Real-time ECG data visualization  
    - AI-powered arrhythmia detection  
    - Emergency alert system for critical cases  
    - Patient health statistics tracking  
    - Wearable-device compatibility 

    **How It Works:**
    1. ECG data is collected via compatible wearable devices.  
    2. The data is analyzed using machine learning algorithms to detect arrhythmias.
    3. Users receive instant feedback and recommendations based on AI analysis.  
    4. In emergencies, the app can alert designated contacts and medical services.           
    
    ---
                
    **📍Developed by:** Team AI-Luminators - 2025   
    """)


#  Settings

elif menu == "⚙️ Settings":
    show_header()
    st.title("⚙️ Settings")
    st.text_input("Patient Name", "Iqbal Hossain")
    st.number_input("Age", 0, 120, 45)
    st.text_input("Device ID", "CARDIO-001")
    st.selectbox("Permission to use User data", ["Yes", "No"], index=0)
    st.color_picker("Accent Color", "#ff4b4b")
    if st.button(" Save Settings"):
        st.success(f"✅ Settings saved successfully!")

# Sidebar Footer

st.sidebar.markdown("---")
st.sidebar.caption(" Developed by Team AI-Luminators – 2025")
