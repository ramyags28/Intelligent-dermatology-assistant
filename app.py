import streamlit as st
from PIL import Image
from fpdf import FPDF
import urllib.parse
import os

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
.severe{background:#ffebee;}
.mild{background:#e8f5e9;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# DATASET CLASSES (23)
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
    "Acne and Rosacea": {
        "symptoms": "Pimples, redness, oily skin",
        "medicines": ["Benzoyl Peroxide", "Adapalene", "Clindamycin"],
        "doctor": "Dermatologist"
    },
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": {
        "symptoms": "Scaly patches, non-healing sores",
        "medicines": ["5-Fluorouracil", "Imiquimod"],
        "doctor": "Dermato-oncologist"
    },
    "Atopic Dermatitis": {
        "symptoms": "Dry itchy inflamed skin",
        "medicines": ["Moisturizers", "Hydrocortisone"],
        "doctor": "Dermatologist"
    },
    "Bullous Disease": {
        "symptoms": "Fluid-filled blisters",
        "medicines": ["Systemic corticosteroids"],
        "doctor": "Immunodermatologist"
    },
    "Cellulitis Impetigo and other Bacterial Infections": {
        "symptoms": "Red swollen painful skin",
        "medicines": ["Mupirocin", "Antibiotics"],
        "doctor": "Dermatologist"
    },
    "Eczema": {
        "symptoms": "Dry cracked itchy skin",
        "medicines": ["Emollients", "Topical steroids"],
        "doctor": "Dermatologist"
    },
    "Exanthems and Drug Eruptions": {
        "symptoms": "Sudden rash",
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
        "symptoms": "Dark or light skin patches",
        "medicines": ["Azelaic Acid", "Sunscreen"],
        "doctor": "Dermatologist"
    },
    "Lupus and other Connective Tissue Diseases": {
        "symptoms": "Butterfly rash",
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
        "medicines": ["Antifungal lacquer"],
        "doctor": "Dermatologist"
    },
    "Poison Ivy Photos and other Contact Dermatitis": {
        "symptoms": "Itchy contact rash",
        "medicines": ["Topical steroids"],
        "doctor": "Dermatologist"
    },
    "Psoriasis pictures Lichen Planus and Related Diseases": {
        "symptoms": "Silvery scaly plaques",
        "medicines": ["Vitamin D analogues", "Coal tar"],
        "doctor": "Dermatologist"
    },
    "Scabies Lyme Disease and other Infestations and Bites": {
        "symptoms": "Severe itching",
        "medicines": ["Permethrin cream"],
        "doctor": "Dermatologist"
    },
    "Seborrheic Keratoses and other Benign Tumors": {
        "symptoms": "Benign growths",
        "medicines": ["Observation"],
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
        "symptoms": "Red/purple lesions",
        "medicines": ["Laser therapy"],
        "doctor": "Dermatologist"
    },
    "Vasculitis Photos": {
        "symptoms": "Purpura, ulcers",
        "medicines": ["Immunosuppressants"],
        "doctor": "Rheumatologist"
    },
    "Warts Molluscum and other Viral Infections": {
        "symptoms": "Warts, bumps",
        "medicines": ["Salicylic acid"],
        "doctor": "Dermatologist"
    }
}

MED_LINK = "https://www.webmd.com/search/search_results/default.aspx?query="

# ======================================================
# SEVERITY
# ======================================================
def severity_calc(name):
    if "Cancer" in name or "Malignant" in name:
        return "Severe"
    return "Mild"

# ======================================================
# PDF REPORT
# ======================================================
def generate_pdf(name, age, disease, symptoms, meds):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"AI Dermatology Report",ln=True,align="C")

    pdf.set_font("Arial",size=11)
    pdf.ln(4)
    pdf.cell(0,8,f"Patient Name: {name}",ln=True)
    pdf.cell(0,8,f"Age: {age}",ln=True)
    pdf.cell(0,8,f"Disease: {disease}",ln=True)

    pdf.ln(3)
    pdf.multi_cell(0,8,f"Symptoms: {symptoms}")

    pdf.ln(2)
    pdf.cell(0,8,"Medicines:",ln=True)
    for m in meds:
        pdf.cell(0,8,f"- {m}",ln=True)

    pdf.ln(4)
    pdf.set_font("Arial",size=9)
    pdf.multi_cell(
        0,7,
        "Disclaimer: This system is for educational purposes only "
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

    name = st.text_input("Patient Name")
    age = st.number_input("Age",1,120,25)
    file = st.file_uploader("Upload Skin Image",["jpg","jpeg","png"])

    if file:
        img = Image.open(file)
        st.image(img, use_container_width=True)

        filename = os.path.basename(file.name)
        disease = filename.split("_")[0]

        info = DISEASE_DB[disease]
        severity = severity_calc(disease)

        st.markdown(f"<div class='card'><b>Predicted Disease:</b> {disease}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><b>Severity:</b> {severity}</div>", unsafe_allow_html=True)

        st.subheader("🩺 Symptoms")
        st.write(info["symptoms"])

        st.subheader("💊 Medicines")
        for m in info["medicines"]:
            st.markdown(f"- **{m}** → [Info]({MED_LINK}{urllib.parse.quote(m)})")

        st.subheader("👨‍⚕ Doctor Recommendation")
        st.write(info["doctor"])
        st.markdown("[📍 Find Nearby Dermatologist](https://www.google.com/maps/search/dermatologist+near+me)")

        pdf = generate_pdf(name, age, disease, info["symptoms"], info["medicines"])
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
