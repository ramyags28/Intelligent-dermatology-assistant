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
    padding:20px;
    border-radius:15px;
    box-shadow:0 6px 15px rgba(0,0,0,0.08);
    margin-bottom:15px;
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
# DATASET CLASSES (23 – EXACT)
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
# FULL DISEASE DATABASE (SYMPTOMS + MEDICINE + DOCTOR)
# ======================================================
DISEASE_DB = {
    "Acne and Rosacea": {
        "symptoms": "Pimples, redness, oily skin, facial flushing",
        "medicines": ["Benzoyl Peroxide", "Adapalene", "Clindamycin"],
        "doctor": "Dermatologist"
    },
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": {
        "symptoms": "Rough scaly patches, non-healing sores",
        "medicines": ["5-Fluorouracil", "Imiquimod"],
        "doctor": "Dermato-oncologist"
    },
    "Atopic Dermatitis": {
        "symptoms": "Dry itchy inflamed skin",
        "medicines": ["Moisturizers", "Hydrocortisone"],
        "doctor": "Dermatologist"
    },
    "Bullous Disease": {
        "symptoms": "Fluid-filled blisters, skin peeling",
        "medicines": ["Systemic corticosteroids"],
        "doctor": "Immunodermatologist"
    },
    "Cellulitis Impetigo and other Bacterial Infections": {
        "symptoms": "Red swollen painful skin, pus",
        "medicines": ["Mupirocin", "Oral antibiotics"],
        "doctor": "Dermatologist"
    },
    "Eczema": {
        "symptoms": "Dry itchy cracked skin",
        "medicines": ["Emollients", "Topical steroids"],
        "doctor": "Dermatologist"
    },
    "Exanthems and Drug Eruptions": {
        "symptoms": "Sudden widespread rash",
        "medicines": ["Antihistamines"],
        "doctor": "Dermatologist"
    },
    "Hair Loss Photos Alopecia and other Hair Diseases": {
        "symptoms": "Hair thinning, bald patches",
        "medicines": ["Minoxidil", "Biotin"],
        "doctor": "Trichologist"
    },
    "Herpes HPV and other STDs": {
        "symptoms": "Painful blisters, warts",
        "medicines": ["Acyclovir"],
        "doctor": "Dermatologist"
    },
    "Light Diseases and Disorders of Pigmentation": {
        "symptoms": "Dark or light patches",
        "medicines": ["Azelaic acid", "Sunscreen"],
        "doctor": "Dermatologist"
    },
    "Lupus and other Connective Tissue Diseases": {
        "symptoms": "Butterfly rash, photosensitivity",
        "medicines": ["Hydroxychloroquine"],
        "doctor": "Rheumatologist"
    },
    "Melanoma Skin Cancer Nevi and Moles": {
        "symptoms": "Irregular mole, color change",
        "medicines": ["Specialist evaluation"],
        "doctor": "Dermato-oncologist"
    },
    "Nail Fungus and other Nail Disease": {
        "symptoms": "Discolored thick nails",
        "medicines": ["Antifungal nail lacquer"],
        "doctor": "Dermatologist"
    },
    "Poison Ivy Photos and other Contact Dermatitis": {
        "symptoms": "Itchy rash after contact",
        "medicines": ["Topical steroids"],
        "doctor": "Dermatologist"
    },
    "Psoriasis pictures Lichen Planus and Related Diseases": {
        "symptoms": "Silvery scaly plaques",
        "medicines": ["Vitamin D analogues", "Coal tar"],
        "doctor": "Dermatologist"
    },
    "Scabies Lyme Disease and other Infestations and Bites": {
        "symptoms": "Severe itching, burrows",
        "medicines": ["Permethrin cream"],
        "doctor": "Dermatologist"
    },
    "Seborrheic Keratoses and other Benign Tumors": {
        "symptoms": "Benign growths",
        "medicines": ["Observation / Cryotherapy"],
        "doctor": "Dermatologist"
    },
    "Systemic Disease": {
        "symptoms": "Skin signs of internal disease",
        "medicines": ["Treat underlying disease"],
        "doctor": "Physician"
    },
    "Tinea Ringworm Candidiasis and other Fungal Infections": {
        "symptoms": "Ring-shaped itchy rash",
        "medicines": ["Clotrimazole", "Ketoconazole"],
        "doctor": "Dermatologist"
    },
    "Urticaria Hives": {
        "symptoms": "Raised itchy wheals",
        "medicines": ["Antihistamines"],
        "doctor": "Allergist"
    },
    "Vascular Tumors": {
        "symptoms": "Red or purple lesions",
        "medicines": ["Laser therapy"],
        "doctor": "Dermatologist"
    },
    "Vasculitis Photos": {
        "symptoms": "Purpura, ulcers",
        "medicines": ["Immunosuppressants"],
        "doctor": "Rheumatologist"
    },
    "Warts Molluscum and other Viral Infections": {
        "symptoms": "Warts, skin bumps",
        "medicines": ["Salicylic acid"],
        "doctor": "Dermatologist"
    }
}

MED_LINK = "https://www.webmd.com/search/search_results/default.aspx?query="

# ======================================================
# IMAGE PREPROCESS
# ======================================================
def preprocess(img):
    img = img.convert("RGB").resize((224,224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, 0)

def severity_calc(disease):
    if "Cancer" in disease or "Malignant" in disease:
        return "Severe"
    return "Mild"

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

    arr = preprocess(img)
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)

    disease = CLASS_NAMES[idx]
    confidence = round(float(preds[idx]) * 100, 2)
    severity = severity_calc(disease)
    info = DISEASE_DB[disease]

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
    st.markdown("[📍 Find Nearby Doctor](https://www.google.com/maps/search/dermatologist+near+me)")

    st.subheader("📊 All Class Confidence Scores")
    for i, p in enumerate(preds):
        st.write(f"{CLASS_NAMES[i]} : {round(float(p)*100,2)}%")
