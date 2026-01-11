import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
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
# MODEL LOAD (DO NOT CHANGE)
# ======================================================
MODEL_PATH = "dermatology_assistant_model.keras"
GDRIVE_URL = "https://drive.google.com/uc?id=1k5QpG18JlqCetsGhqZuNdCFS_OdPDDUZ"

@st.cache_resource
def load_derm_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model (first run only)..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_derm_model()

# ======================================================
# 23 DATASET CLASSES
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
# DISEASE DATABASE (ALL 23)
# ======================================================
DISEASE_DB = {
    "Acne and Rosacea": ("Pimples, redness, oily skin", "Mild", ["Benzoyl Peroxide", "Adapalene"]),
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": ("Scaly patches, non-healing sores", "Severe", ["Sunscreen SPF 30+", "5-Fluorouracil"]),
    "Atopic Dermatitis": ("Dry itchy inflamed skin", "Moderate", ["Moisturizers", "Hydrocortisone"]),
    "Bullous Disease": ("Fluid-filled blisters", "Severe", ["Systemic Corticosteroids"]),
    "Cellulitis Impetigo and other Bacterial Infections": ("Red swollen skin, pus", "Severe", ["Mupirocin", "Antibiotics"]),
    "Eczema": ("Cracked itchy skin", "Moderate", ["Emollients", "Topical Steroids"]),
    "Exanthems and Drug Eruptions": ("Sudden rash after medication", "Moderate", ["Antihistamines"]),
    "Hair Loss Photos Alopecia and other Hair Diseases": ("Hair thinning, bald patches", "Mild", ["Minoxidil", "Biotin"]),
    "Herpes HPV and other STDs": ("Blisters, warts", "Moderate", ["Acyclovir"]),
    "Light Diseases and Disorders of Pigmentation": ("Light or dark patches", "Mild", ["Azelaic Acid", "Sunscreen"]),
    "Lupus and other Connective Tissue Diseases": ("Butterfly rash, photosensitivity", "Severe", ["Hydroxychloroquine"]),
    "Melanoma Skin Cancer Nevi and Moles": ("Irregular mole, bleeding", "Severe", ["Immediate Specialist Care"]),
    "Nail Fungus and other Nail Disease": ("Thick discolored nails", "Moderate", ["Antifungal Lacquer"]),
    "Poison Ivy Photos and other Contact Dermatitis": ("Itchy contact rash", "Mild", ["Topical Steroids"]),
    "Psoriasis pictures Lichen Planus and Related Diseases": ("Silvery scaly plaques", "Moderate", ["Coal Tar", "Vitamin D Cream"]),
    "Scabies Lyme Disease and other Infestations and Bites": ("Severe itching, burrows", "Moderate", ["Permethrin Cream"]),
    "Seborrheic Keratoses and other Benign Tumors": ("Waxy benign growths", "Mild", ["Observation"]),
    "Systemic Disease": ("Skin signs of internal illness", "Severe", ["Treat underlying disease"]),
    "Tinea Ringworm Candidiasis and other Fungal Infections": ("Ring-shaped itchy rash", "Moderate", ["Clotrimazole", "Ketoconazole"]),
    "Urticaria Hives": ("Raised itchy wheals", "Mild", ["Antihistamines"]),
    "Vascular Tumors": ("Red or purple lesions", "Moderate", ["Laser Therapy"]),
    "Vasculitis Photos": ("Purpura, ulcers", "Severe", ["Immunosuppressants"]),
    "Warts Molluscum and other Viral Infections": ("Small raised bumps", "Mild", ["Salicylic Acid"])
}

MED_LINK = "https://www.webmd.com/search/search_results/default.aspx?query="

# ======================================================
# IMAGE PREPROCESSING (IMPROVED)
# ======================================================
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((224,224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.clip(arr, 0, 1)
    return np.expand_dims(arr, axis=0)

# ======================================================
# PDF REPORT
# ======================================================
def create_pdf(name, age, disease, confidence, severity, symptoms, medicines):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"AI Dermatology Report",ln=True,align="C")
    pdf.ln(4)

    pdf.set_font("Arial",size=12)
    pdf.cell(0,8,f"Patient Name: {name}",ln=True)
    pdf.cell(0,8,f"Age: {age}",ln=True)
    pdf.cell(0,8,f"Disease: {disease}",ln=True)
    pdf.cell(0,8,f"Model Confidence: {confidence}%",ln=True)
    pdf.cell(0,8,f"Severity: {severity}",ln=True)

    pdf.ln(4)
    pdf.multi_cell(0,8,f"Symptoms: {symptoms}")

    pdf.ln(3)
    pdf.cell(0,8,"Medicines:",ln=True)
    for m in medicines:
        pdf.cell(0,8,f"- {m}",ln=True)

    pdf.ln(6)
    pdf.set_font("Arial",size=10)
    pdf.multi_cell(
        0,8,
        "Disclaimer: This AI-based system is for educational and screening purposes only. "
        "It does not replace professional medical diagnosis."
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
            st.error("Please enter username and password")

# ======================================================
# MAIN APP
# ======================================================
def main_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Patient Details", "Upload Image", "Results"])

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    if page == "Patient Details":
        st.session_state["name"] = st.text_input("Patient Name")
        st.session_state["age"] = st.number_input("Age", 1, 120, 25)

    if page == "Upload Image":
        file = st.file_uploader("Upload Skin Image", ["jpg","jpeg","png"])
        if file:
            st.session_state["image"] = Image.open(file)
            st.image(st.session_state["image"], use_container_width=True)

    if page == "Results" and "image" in st.session_state:
        img = st.session_state["image"]
        preds = model.predict(preprocess(img), verbose=0)[0]

        top3_idx = np.argsort(preds)[-3:][::-1]
        main_idx = top3_idx[0]

        disease = CLASS_NAMES[main_idx]
        confidence = round(float(preds[main_idx]) * 100, 2)
        symptoms, severity, medicines = DISEASE_DB[disease]

        color = "severe" if severity=="Severe" else "moderate" if severity=="Moderate" else "mild"

        st.markdown(f"<div class='card {color}'><b>Disease:</b> {disease}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><b>Confidence:</b> {confidence}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><b>Severity:</b> {severity}</div>", unsafe_allow_html=True)

        if confidence < 60:
            st.warning(
                "⚠️ Low confidence prediction. Some skin diseases have overlapping visual features. "
                "Consult a dermatologist for confirmation."
            )

        st.subheader("🔍 Top 3 Predictions")
        for i in top3_idx:
            st.write(f"{CLASS_NAMES[i]} — {round(float(preds[i])*100,2)}%")

        st.subheader("🩺 Symptoms")
        st.write(symptoms)

        st.subheader("💊 Medicine Recommendation")
        for m in medicines:
            st.markdown(f"- **{m}** → [Info]({MED_LINK}{urllib.parse.quote(m)})")

        st.subheader("👨‍⚕ Doctor Recommendation")
        st.write("Consult a certified dermatologist")
        st.markdown("[📍 Find Nearby Dermatologist](https://www.google.com/maps/search/dermatologist+near+me)")

        pdf = create_pdf(
            st.session_state.get("name",""),
            st.session_state.get("age",""),
            disease,
            confidence,
            severity,
            symptoms,
            medicines
        )

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
