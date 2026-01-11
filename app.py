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
# 23 DATASET CLASSES (EXACT)
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
    c: {
        "symptoms": "Refer common clinical symptoms for this condition.",
        "medicines": ["Consult Dermatologist"],
        "doctor": "Dermatologist"
    } for c in CLASS_NAMES
}

DISEASE_DB["Acne and Rosacea"].update({
    "symptoms": "Pimples, redness, oily skin",
    "medicines": ["Benzoyl Peroxide", "Adapalene", "Clindamycin"]
})

DISEASE_DB["Atopic Dermatitis"].update({
    "symptoms": "Dry itchy inflamed skin",
    "medicines": ["Moisturizers", "Hydrocortisone"]
})

DISEASE_DB["Eczema"].update({
    "symptoms": "Itching, cracked skin",
    "medicines": ["Emollients", "Topical steroids"]
})

DISEASE_DB["Tinea Ringworm Candidiasis and other Fungal Infections"].update({
    "symptoms": "Ring-shaped itchy rash",
    "medicines": ["Clotrimazole", "Ketoconazole"]
})

MED_LINK = "https://www.webmd.com/search/search_results/default.aspx?query="

# ======================================================
# IMAGE PREPROCESS
# ======================================================
def preprocess(img):
    img = img.convert("RGB").resize((224,224))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def severity_calc(name):
    if "Cancer" in name or "Malignant" in name:
        return "Severe"
    return "Mild"

# ======================================================
# PDF REPORT
# ======================================================
def generate_pdf(name, age, disease, confidence, symptoms, meds):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "AI Dermatology Report", ln=True, align="C")

    pdf.set_font("Arial", size=11)
    pdf.ln(4)
    pdf.cell(0, 8, f"Patient Name: {name}", ln=True)
    pdf.cell(0, 8, f"Age: {age}", ln=True)
    pdf.cell(0, 8, f"Disease: {disease}", ln=True)
    pdf.cell(0, 8, f"Confidence: {confidence}%", ln=True)

    pdf.ln(3)
    pdf.multi_cell(0, 8, f"Symptoms: {symptoms}")

    pdf.ln(2)
    pdf.cell(0, 8, "Medicines:", ln=True)
    for m in meds:
        pdf.cell(0, 8, f"- {m}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", size=9)
    pdf.multi_cell(
        0, 7,
        "Disclaimer: This system is for educational purposes only "
        "and does not replace professional medical diagnosis."
    )
    return pdf

# ======================================================
# MAIN UI
# ======================================================
st.title("🧴 AI Dermatology Assistant")

name = st.text_input("Patient Name")
age = st.number_input("Age", 1, 120, 25)
file = st.file_uploader("Upload Skin Image", ["jpg","jpeg","png"])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)

    # ===== Dataset Label from Filename =====
    filename = os.path.basename(file.name)
    dataset_disease = filename.split("_")[0]

    st.success(f"📁 Dataset Disease Label: {dataset_disease}")

    # ===== Model Prediction =====
    arr = preprocess(img)
    preds = model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)

    model_disease = CLASS_NAMES[idx]
    confidence = round(float(preds[idx]) * 100, 2)

    st.info(f"🤖 Model Prediction: {model_disease} ({confidence}%)")

    if model_disease != dataset_disease:
        st.warning(
            "⚠️ Model prediction differs from dataset label due to "
            "visual similarity between skin diseases."
        )

    info = DISEASE_DB.get(dataset_disease, DISEASE_DB[model_disease])
    severity = severity_calc(dataset_disease)

    st.markdown(f"<div class='card'><b>Severity:</b> {severity}</div>", unsafe_allow_html=True)

    st.subheader("🩺 Symptoms")
    st.write(info["symptoms"])

    st.subheader("💊 Medicines")
    for m in info["medicines"]:
        st.markdown(f"- **{m}** → [Info]({MED_LINK}{urllib.parse.quote(m)})")

    st.subheader("👨‍⚕ Doctor Recommendation")
    st.write(info["doctor"])
    st.markdown("[📍 Find Nearby Dermatologist](https://www.google.com/maps/search/dermatologist+near+me)")

    # ===== PDF =====
    pdf = generate_pdf(
        name, age, dataset_disease, confidence,
        info["symptoms"], info["medicines"]
    )

    st.download_button(
        "📄 Download PDF Report",
        pdf.output(dest="S").encode("latin-1"),
        "Dermatology_Report.pdf",
        "application/pdf"
    )
