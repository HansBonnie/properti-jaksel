import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Properti Jaksel — ROI & Occ Predictor",
    page_icon="🏢",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}
.stApp { background-color: #F7F6F2; }

.result-card {
    background: #1a1a2e;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: white;
    margin-top: 1rem;
}
.result-label {
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #a0a0b0;
    margin-bottom: 0.4rem;
}
.result-value {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #e8c547;
    line-height: 1;
}
.result-unit {
    font-size: 1rem;
    color: #a0a0b0;
    margin-top: 0.3rem;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #2d2d2d;
    margin-bottom: 0.5rem;
    border-left: 3px solid #e8c547;
    padding-left: 0.6rem;
}
.hint {
    font-size: 0.78rem;
    color: #888;
    margin-top: -0.4rem;
    margin-bottom: 0.8rem;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] > div {
    background: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/catboost_final.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifacts    = load_model()
    cb_roi       = artifacts["catboost_roi"]
    cb_occ       = artifacts["catboost_occ"]
    pre_pipeline = artifacts["pre_pipeline"]
    onehot_cols      = artifacts["onehot_cols"]
    ordinal_cols     = artifacts["ordinal_cols"]
    passthrough_cols = artifacts["passthrough_cols"]
    MODEL_LOADED = True
except Exception as e:
    st.error(f"❌ Gagal load model: {e}")
    MODEL_LOADED = False


# ── Feature engineering (sama persis dengan training) ──────────────────
def engineer_features(row: dict) -> pd.DataFrame:
    grade_map = {
        "Premium":"Tier_1","Prime":"Tier_1","5*":"Tier_1",
        "Upper":"Tier_2","A":"Tier_2","4*":"Tier_2",
        "Middle Up":"Tier_3","B":"Tier_3","Primary":"Tier_3","B+":"Tier_3",
        "Middle":"Tier_4","C":"Tier_4","3*":"Tier_4",
        "Middle Low":"Tier_5","Secondary":"Tier_5",
    }
    grade_order = {"Tier_1":5,"Tier_2":4,"Tier_3":3,"Tier_4":2,"Tier_5":1}

    transit_score = (
        row["General_Bus"] * 1 + row["TJ"] * 1.5 +
        row["MRT"] * 2 + row["LRT"] * 2 + row["KRL"] * 1.5
    )
    road_cat = pd.cut(
        [row["Distance_Main_Road"]],
        bins=[-1, 0, 100, 300, 600, float("inf")],
        labels=["On Road","Very Close","Close","Moderate","Far"]
    )[0]
    transport_cat = pd.cut(
        [row["Am_Public_Transport"]],
        bins=[-1, 1, 2, 3, float("inf")],
        labels=["Very Limited","Limited","Moderate","Good"]
    )[0]
    grade_ordinal = grade_order.get(grade_map.get(row["Grade"], "Tier_3"), 3)

    df = pd.DataFrame([{
        "Property":           row["Property"],
        "Type":               row["Type"],
        "Area":               row["Area"],
        "Road_Category":      road_cat,
        "Transport_Category": transport_cat,
        "Lat":                row["Lat"],
        "Long":               row["Long"],
        "Grade_Ordinal":      grade_ordinal,
        "Transit_Score":      transit_score,
        "Log_Total_Unit":     np.log1p(row["Total_Unit"]),
        "Log_Sold_Rate":      np.log1p(row["Sold_Rate"]),
        "Log_Rental_Rate":    np.log1p(row["Rental_Rate"]),
    }])
    return df


def predict(row: dict):
    df_eng = engineer_features(row)
    X = pre_pipeline.transform(df_eng)
    roi = float(cb_roi.predict(X)[0])
    occ = float(cb_occ.predict(X)[0])
    return roi, occ


# ── Dropdown options ────────────────────────────────────────────────────
PROPERTY_TYPES = ["Apartment", "Hotel", "Office", "Retail"]
LEASE_TYPES    = ["Lease", "Strata"]
GRADES = ["Premium","Prime","5*","Upper","A","4*","Middle Up","B+","B",
          "Primary","Middle","C","3*","Middle Low","Secondary"]
AREAS = ["Antasari","Bintaro","Blok M","Brawijaya","Casablanca","Cikini",
         "Cilandak","Cinere","Cipete","Cipulir","Epicentrum","Fatmawati",
         "Gandaria","Gatot Subroto","Iskandarsyah","Kalibata","Karet Semanggi",
         "Kebayoran Baru","Kebayoran Lama","Kemang","Kuningan","Lebak Bulus",
         "MT Haryono","Mahakam","Manggarai","Margonda","Mega Kuningan",
         "Melawai","Menteng","Other","Pakubuwono","Pancoran","Pejaten",
         "Permata Hijau","Pondok Indah","Prapanca","Rasuna Said","SCBD",
         "Satrio","Semanggi","Senayan","Senen","Senopati","Setiabudi",
         "Simprug","Subroto","Sudirman","Sunter","TB Simatupang",
         "Tanjung Duren","Tebet","Tendean","Thamrin","Wijaya"]

# ── Session state ───────────────────────────────────────────────────────
if "lat" not in st.session_state: st.session_state.lat = -6.2297
if "lng" not in st.session_state: st.session_state.lng = 106.8295
if "roi" not in st.session_state: st.session_state.roi = None
if "occ" not in st.session_state: st.session_state.occ = None


# ════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════
st.markdown("# 🏢 Properti Jaksel")
st.markdown("#### Prediksi ROI & Occupancy Rate berbasis Machine Learning (CatBoost)")
st.divider()

col_form, col_map = st.columns([1, 1.4], gap="large")

# ── KOLOM KIRI: Form input ──────────────────────────────────────────────
with col_form:

    # Koordinat (bisa diisi manual atau dari klik peta)
    st.markdown('<div class="section-title">📍 Lokasi</div>', unsafe_allow_html=True)
    st.markdown('<p class="hint">Klik peta di kanan untuk mengisi otomatis, atau isi manual.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    lat_input = c1.number_input("Latitude",  value=st.session_state.lat,
                                 format="%.6f", step=0.0001)
    lng_input = c2.number_input("Longitude", value=st.session_state.lng,
                                 format="%.6f", step=0.0001)
    st.session_state.lat = lat_input
    st.session_state.lng = lng_input

    st.divider()
    st.markdown('<div class="section-title">🏗️ Identitas Properti</div>', unsafe_allow_html=True)

    c3, c4, c5 = st.columns(3)
    prop_type = c3.selectbox("Property", PROPERTY_TYPES)
    lease_type = c4.selectbox("Type", LEASE_TYPES)
    grade = c5.selectbox("Grade", GRADES)
    area  = st.selectbox("Area", AREAS)

    st.divider()
    st.markdown('<div class="section-title">📊 Data Properti</div>', unsafe_allow_html=True)

    c6, c7 = st.columns(2)
    total_unit  = c6.number_input("Total Unit", min_value=1, value=100, step=1)
    sold_rate   = c7.number_input("Sold Rate (juta/m²)", min_value=0.0, value=50.0, step=0.5)
    rental_rate = st.number_input("Rental Rate (ribu/m²/bln)", min_value=0.0, value=300.0, step=10.0)

    st.divider()
    st.markdown('<div class="section-title">🛣️ Aksesibilitas</div>', unsafe_allow_html=True)

    c8, c9 = st.columns([2, 1])
    main_road = c8.text_input("Nama Jalan Utama (opsional)", placeholder="e.g. Jl. Sudirman")
    distance  = c9.number_input("Jarak (m)", min_value=0, value=0, step=50)

    st.markdown("**Transportasi umum dalam radius 500 m**")
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    gen_bus = tc1.number_input("Gen. Bus", min_value=0, max_value=10, value=1, step=1, label_visibility="visible")
    tj      = tc2.number_input("TJ",       min_value=0, max_value=10, value=0, step=1)
    mrt     = tc3.number_input("MRT",      min_value=0, max_value=10, value=0, step=1)
    lrt     = tc4.number_input("LRT",      min_value=0, max_value=10, value=0, step=1)
    krl     = tc5.number_input("KRL",      min_value=0, max_value=10, value=0, step=1)

    am_public = int(gen_bus > 0) + int(tj > 0) + int(mrt > 0) + int(lrt > 0) + int(krl > 0)

    st.divider()

    predict_btn = st.button("🔍 Prediksi ROI & Occ", type="primary", use_container_width=True)

    if predict_btn and MODEL_LOADED:
        input_row = {
            "Property":           prop_type,
            "Type":               lease_type,
            "Area":               area,
            "Grade":              grade,
            "Lat":                lat_input,
            "Long":               lng_input,
            "Total_Unit":         total_unit,
            "Sold_Rate":          sold_rate,
            "Rental_Rate":        rental_rate,
            "Distance_Main_Road": distance,
            "Am_Public_Transport":am_public,
            "General_Bus":        gen_bus,
            "TJ":                 tj,
            "MRT":                mrt,
            "LRT":                lrt,
            "KRL":                krl,
        }
        with st.spinner("Menghitung prediksi..."):
            roi, occ = predict(input_row)
        st.session_state.roi = roi
        st.session_state.occ = occ

    # ── Hasil prediksi ─────────────────────────────────────────────────
    if st.session_state.roi is not None:
        r1, r2 = st.columns(2)
        r1.markdown(f"""
        <div class="result-card">
            <div class="result-label">ROI</div>
            <div class="result-value">{st.session_state.roi:.2%}</div>
            <div class="result-unit">Return on Investment</div>
        </div>""", unsafe_allow_html=True)

        r2.markdown(f"""
        <div class="result-card">
            <div class="result-label">Occupancy</div>
            <div class="result-value">{st.session_state.occ:.1%}</div>
            <div class="result-unit">Tingkat Hunian</div>
        </div>""", unsafe_allow_html=True)


# ── KOLOM KANAN: Peta ───────────────────────────────────────────────────
with col_map:
    st.markdown('<div class="section-title">🗺️ Klik peta untuk menentukan lokasi</div>',
                unsafe_allow_html=True)
    st.markdown('<p class="hint">Klik di mana saja pada peta — Lat & Long akan terisi otomatis.</p>',
                unsafe_allow_html=True)

    m = folium.Map(
        location=[st.session_state.lat, st.session_state.lng],
        zoom_start=13,
        tiles="CartoDB positron",
    )

    # Marker posisi saat ini
    folium.Marker(
        location=[st.session_state.lat, st.session_state.lng],
        popup=f"📍 ({st.session_state.lat:.5f}, {st.session_state.lng:.5f})",
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(m)

    # Tampilkan peta dan tangkap klik
    map_data = st_folium(m, width="100%", height=520, returned_objects=["last_clicked"])

    # Update koordinat jika user klik
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lng = map_data["last_clicked"]["lng"]
        if (abs(clicked_lat - st.session_state.lat) > 0.00001 or
                abs(clicked_lng - st.session_state.lng) > 0.00001):
            st.session_state.lat = clicked_lat
            st.session_state.lng = clicked_lng
            st.rerun()

    st.caption(f"📍 Posisi aktif: **{st.session_state.lat:.5f}**, **{st.session_state.lng:.5f}**")


# ── Footer ──────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: CatBoost (tuned) | Dataset: Properti Jakarta Selatan | CV R² ROI: 0.9927 · Occ: 0.6632")
