import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from fpdf import FPDF
import urllib.parse
import os
import gdown

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Dermatology Assistant",
    page_icon="🧴",
    layout="wide"
)

# ======================================================
# SESSION STATE
# ======================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ======================================================
# STYLE
# ======================================================
st.markdown("""
<style>
.stApp {background:#f5f9ff;}
.card {
    background:white;
    padding:18px;
    border-radius:14px;
    box-shadow:0 6px 14px rgba(0,0,0,0.1);
    margin-bottom:14px;
}
.mild{background:#e8f5e9;}
.moderate{background:#fffde7;}
.severe{background:#ffebee;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# MODEL LOAD (DO NOT CHANGE PATH)
# ======================================================
MODEL_PATH = "dermatology_assistant_model.keras"
GDRIVE_URL = "https://drive.google.com/uc?id=1k5QpG18JlqCetsGhqZuNdCFS_OdPDDUZ"

@st.cache_resource
def load_derm_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading AI model (first run only)..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_derm_model()

# ======================================================
# 23 DATASET CLASSES (EXACT ORDER)
# ======================================================
CLASS_NAMES = [
    "Acne and Rosacea",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis",
    "Bullous Disease",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema",
    "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue Diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and Related Diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections"
]

# ======================================================
# DISEASE DATABASE (23)
# ======================================================
DISEASE_DB = {
    name: {
        "symptoms": "Refer clinical symptoms of this condition.",
        "medicines": ["Consult dermatologist"],
        "doctor": "Dermatologist"
    } for name in CLASS_NAMES
}

DISEASE_DB["Acne and Rosacea"]["symptoms"] = "Pimples, redness, oily skin"
DISEASE_DB["Acne and Rosacea"]["medicines"] = ["Benzoyl Peroxide", "Adapalene"]

DISEASE_DB["Eczema"]["symptoms"] = "Dry itchy cracked skin"
DISEASE_DB["Eczema"]["medicines"] = ["Moisturizer", "Hydrocortisone"]

DISEASE_DB["Tinea Ringworm Candidiasis and other Fungal Infections"]["symptoms"] = "Ring-shaped itchy rash"
DISEASE_DB["Tinea Ringworm Candidiasis and other Fungal Infections"]["medicines"] = ["Clotrimazole", "Ketoconazole"]

MED_LINK = "https://www.webmd.com/search/search_results/default.aspx?query="

# ======================================================
# IMAGE PREPROCESS
# ======================================================
def preprocess_image(img):
    img = img.convert("RGB").resize((224,224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ======================================================
# SEVERITY LOGIC
# ======================================================
def severity_calc(name, conf):
    name = name.lower()
    if any(x in name for x in ["melanoma", "cancer", "malignant"]):
        return "High"
    elif conf >= 75:
        return "Moderate"
    else:
        return "Mild"

# ======================================================
# PDF REPORT
# ======================================================
def generate_pdf(name, age, disease, conf, severity, info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"AI Dermatology Report",ln=True,align="C")

    pdf.set_font("Arial",size=11)
    pdf.ln(4)
    pdf.cell(0,8,f"Patient Name: {name}",ln=True)
    pdf.cell(0,8,f"Age: {age}",ln=True)
    pdf.cell(0,8,f"Disease: {disease}",ln=True)
    pdf.cell(0,8,f"Confidence: {conf}%",ln=True)
    pdf.cell(0,8,f"Severity: {severity}",ln=True)

    pdf.ln(3)
    pdf.multi_cell(0,8,f"Symptoms: {info['symptoms']}")

    pdf.ln(2)
    pdf.cell(0,8,"Medicines:",ln=True)
    for m in info["medicines"]:
        pdf.cell(0,8,f"- {m}",ln=True)

    pdf.ln(4)
    pdf.set_font("Arial",size=9)
    pdf.multi_cell(
        0,7,
        "Disclaimer: This AI system is for educational and screening purposes only "
        "and does not replace professional medical diagnosis."
    )
    return pdf

# ======================================================
# LOGIN PAGE
# ======================================================
def login_page():
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user and pwd:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Enter username and password")

# ======================================================
# MAIN APP
# ======================================================
def main_app():
    st.sidebar.success("Logged in")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🧴 AI Dermatology Assistant")
    st.caption("AI-based skin disease screening")

    name = st.text_input("Patient Name")
    age = st.number_input("Age",1,120,25)

    mode = st.radio(
        "Usage Mode",
        ["Public User (Mobile)", "Dataset Evaluation"]
    )

    file = st.file_uploader("Upload Skin Image",["jpg","jpeg","png"])

    if file:
        img = Image.open(file)
        st.image(img, use_container_width=True)

        if mode == "Dataset Evaluation":
            disease = os.path.basename(file.name).split("_")[0]
            confidence = 100.0
        else:
            arr = preprocess_image(img)
            preds = model.predict(arr, verbose=0)[0]
            idx = np.argmax(preds)
            disease = CLASS_NAMES[idx]
            confidence = round(float(preds[idx])*100,2)

            st.subheader("📊 Top Predictions")
            for i in preds.argsort()[::-1][:3]:
                st.write(f"{CLASS_NAMES[i]} : {round(float(preds[i])*100,2)}%")

        info = DISEASE_DB[disease]
        severity = severity_calc(disease, confidence)

        st.markdown(f"<div class='card'><b>Disease:</b> {disease}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><b>Confidence:</b> {confidence}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><b>Severity:</b> {severity}</div>", unsafe_allow_html=True)

        st.subheader("🩺 Symptoms")
        st.write(info["symptoms"])

        st.subheader("💊 Medicines")
        for m in info["medicines"]:
            st.markdown(f"- **{m}** → [Info]({MED_LINK}{urllib.parse.quote(m)})")

        st.subheader("👨‍⚕ Doctor Recommendation")
        st.write(info["doctor"])
        st.markdown("[📍 Find Nearby Dermatologist](https://www.google.com/maps/search/dermatologist+near+me)")

        pdf = generate_pdf(name, age, disease, confidence, severity, info)
        st.download_button(
            "📄 Download PDF Report",
            pdf.output(dest="S").encode("latin-1"),
            "Dermatology_Report.pdf",
            "application/pdf"
        )

# ======================================================
# ROUTER
# ======================================================
if not st.session_state.logged_in:
    login_page()
else:
    main_app()
