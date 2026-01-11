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
    padding:20px;
    border-radius:15px;
    box-shadow:0 6px 15px rgba(0,0,0,0.08);
    margin-bottom:15px;
}
.mild{background:#e8f5e9;}
.severe{background:#ffebee;}
.unknown{background:#eceff1;}
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
# FULL DISEASE DATABASE (23 CLASSES)
# ======================================================
DISEASE_DB = {
    "Acne and Rosacea": {
        "symptoms": "Pimples, redness, oily skin, facial flushing",
        "medicines": ["Benzoyl Peroxide", "Adapalene", "Clindamycin"],
        "doctor": "Dermatologist"
    },
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": {
        "symptoms": "Rough scaly patches, non-healing sores",
        "medicines": ["Sunscreen SPF 30+", "5-Fluorouracil"],
        "doctor": "Dermato-Oncologist"
    },
    "Atopic Dermatitis": {
        "symptoms": "Dry itchy inflamed skin",
        "medicines": ["Moisturizers", "Hydrocortisone"],
        "doctor": "Dermatologist"
    },
    "Bullous Disease": {
        "symptoms": "Fluid-filled blisters, skin peeling",
        "medicines": ["Systemic Corticosteroids"],
        "doctor": "Immunodermatologist"
    },
    "Cellulitis Impetigo and other Bacterial Infections": {
        "symptoms": "Red swollen skin, pus, fever",
        "medicines": ["Mupirocin", "Oral Antibiotics"],
        "doctor": "Dermatologist"
    },
    "Eczema": {
        "symptoms": "Itching, cracked skin, redness",
        "medicines": ["Emollients", "Topical Steroids"],
        "doctor": "Dermatologist"
    },
    "Exanthems and Drug Eruptions": {
        "symptoms": "Sudden rash after drug intake",
        "medicines": ["Antihistamines"],
        "doctor": "Dermatologist"
    },
    "Hair Loss Photos Alopecia and other Hair Diseases": {
        "symptoms": "Hair thinning, bald patches",
        "medicines": ["Minoxidil", "Biotin"],
        "doctor": "Trichologist"
    },
    "Herpes HPV and other STDs": {
        "symptoms": "Blisters, warts, painful sores",
        "medicines": ["Acyclovir"],
        "doctor": "Dermatologist"
    },
    "Light Diseases and Disorders of Pigmentation": {
        "symptoms": "Dark or light skin patches",
        "medicines": ["Azelaic Acid", "Sunscreen"],
        "doctor": "Dermatologist"
    },
    "Lupus and other Connective Tissue Diseases": {
        "symptoms": "Butterfly rash, photosensitivity",
        "medicines": ["Hydroxychloroquine"],
        "doctor": "Rheumatologist"
    },
    "Melanoma Skin Cancer Nevi and Moles": {
        "symptoms": "Irregular mole, bleeding",
        "medicines": ["Immediate Specialist Evaluation"],
        "doctor": "Dermato-Oncologist"
    },
    "Nail Fungus and other Nail Disease": {
        "symptoms": "Discolored thick nails",
        "medicines": ["Antifungal Nail Lacquer"],
        "doctor": "Dermatologist"
    },
    "Poison Ivy Photos and other Contact Dermatitis": {
        "symptoms": "Itchy rash after contact",
        "medicines": ["Topical Steroids"],
        "doctor": "Dermatologist"
    },
    "Psoriasis pictures Lichen Planus and Related Diseases": {
        "symptoms": "Silvery scaly plaques",
        "medicines": ["Coal Tar", "Vitamin D Cream"],
        "doctor": "Dermatologist"
    },
    "Scabies Lyme Disease and other Infestations and Bites": {
        "symptoms": "Severe itching, burrows",
        "medicines": ["Permethrin Cream"],
        "doctor": "Dermatologist"
    },
    "Seborrheic Keratoses and other Benign Tumors": {
        "symptoms": "Waxy skin growths",
        "medicines": ["Observation", "Cryotherapy"],
        "doctor": "Dermatologist"
    },
    "Systemic Disease": {
        "symptoms": "Skin signs of internal illness",
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
        "medicines": ["Laser Therapy", "Beta Blockers"],
        "doctor": "Dermatologist"
    },
    "Vasculitis Photos": {
        "symptoms": "Purpura, ulcers",
        "medicines": ["Immunosuppressants"],
        "doctor": "Rheumatologist"
    },
    "Warts Molluscum and other Viral Infections": {
        "symptoms": "Small raised bumps",
        "medicines": ["Salicylic Acid"],
        "doctor": "Dermatologist"
    }
}

MED_LINK = "https://www.webmd.com/search/search_results/default.aspx?query="

# ======================================================
# IMAGE PREPROCESSING
# ======================================================
def preprocess(img):
    img = img.convert("RGB").resize((224,224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, 0)

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
    st.sidebar.title("🧴 Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Upload Image", "Webcam", "Results"])

    st.sidebar.success("Logged in")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    if page == "Home":
        st.title("AI Dermatology Assistant")
        st.info(
            "⚠️ Disclaimer: This AI system is for educational and screening purposes only. "
            "It does not replace professional medical diagnosis."
        )

    if page in ["Upload Image", "Webcam"]:
        st.session_state["name"] = st.text_input("Patient Name")
        st.session_state["age"] = st.number_input("Age", 1, 120, 25)

        if page == "Upload Image":
            file = st.file_uploader("Upload Skin Image", ["jpg","jpeg","png"])
            if file:
                st.session_state["image"] = Image.open(file)

        if page == "Webcam":
            cam = st.camera_input("Capture Image")
            if cam:
                st.session_state["image"] = Image.open(cam)

        if "image" in st.session_state:
            st.image(st.session_state["image"], use_container_width=True)

    if page == "Results" and "image" in st.session_state:
        img = st.session_state["image"]
        preds = model.predict(preprocess(img))[0]
        idx = int(np.argmax(preds))
        confidence = round(float(preds[idx]) * 100, 2)

        if confidence < 40:
            disease = "Unknown Disease"
            info = {
                "symptoms": "Image does not match trained dataset",
                "medicines": ["Consult dermatologist"],
                "doctor": "Dermatologist"
            }
        else:
            disease = CLASS_NAMES[idx]
            info = DISEASE_DB[disease]

        st.markdown(f"<div class='card'><b>Disease:</b> {disease}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><b>Accuracy:</b> {confidence}%</div>", unsafe_allow_html=True)

        st.subheader("Symptoms")
        st.write(info["symptoms"])

        st.subheader("Medicines")
        for m in info["medicines"]:
            st.markdown(f"- **{m}** → [Info]({MED_LINK}{urllib.parse.quote(m)})")

        st.subheader("Doctor Recommendation")
        st.write(info["doctor"])

        st.subheader("Top-3 Predictions")
        for i in np.argsort(preds)[::-1][:3]:
            st.write(f"{CLASS_NAMES[i]} : {round(float(preds[i])*100,2)}%")

# ======================================================
# ROUTER
# ======================================================
if not st.session_state.logged_in:
    login_page()
else:
    main_app()
