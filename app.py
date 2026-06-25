import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium
import math
 
st.set_page_config(
    page_title="Properti Jaksel — ROI & Occ Predictor",
    page_icon="🏢",
    layout="wide",
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');
 
html, body, [class*="css"], p, div, span, label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #1a1a1a;
}
.stApp { background-color: #f0ede8; }
 
.app-header {
    background: #1c1c1e;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}
.app-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.8rem;
    color: #f5e642 !important;
    margin: 0;
    line-height: 1.1;
}
.app-subtitle {
    font-size: 0.82rem;
    color: #9a9a9a !important;
    margin: 0.2rem 0 0;
}
.section-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1c1c1e !important;
    margin: 1.2rem 0 0.6rem;
    padding-left: 0.7rem;
    border-left: 3px solid #f5e642;
}
.hint {
    font-size: 0.75rem;
    color: #888888 !important;
    margin: -0.3rem 0 0.8rem;
    font-style: italic;
}
 
/* Input dark style — broad selectors for Streamlit Cloud compatibility */
input, textarea {
    background-color: #1c1c1e !important;
    color: #ffffff !important;
    border: 1.5px solid #3a3a3a !important;
    border-radius: 8px !important;
}
input::placeholder, textarea::placeholder {
    color: #888888 !important;
}
 
/* Selectbox — container */
div[data-testid="stSelectbox"] > div > div,
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div > div {
    background-color: #1c1c1e !important;
    color: #ffffff !important;
    border: 1.5px solid #3a3a3a !important;
    border-radius: 8px !important;
}
/* Selectbox — teks nilai yang dipilih */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div[class*="singleValue"],
div[data-testid="stSelectbox"] span,
div[data-testid="stSelectbox"] div[class*="placeholder"] {
    color: #ffffff !important;
}
div[data-testid="stSelectbox"] svg,
div[data-baseweb="select"] svg { fill: #ffffff !important; }
 
/* Selectbox dropdown list */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[role="listbox"] {
    background-color: #1c1c1e !important;
}
li[role="option"],
div[data-baseweb="popover"] li {
    background-color: #1c1c1e !important;
    color: #ffffff !important;
}
li[role="option"]:hover,
div[data-baseweb="popover"] li:hover {
    background-color: #2e2e2e !important;
}
 
/* Labels */
label, .stNumberInput label, .stTextInput label, .stSelectbox label {
    color: #ffffff !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}
 
/* +/- buttons */
div[data-testid="stNumberInput"] button,
button[data-testid="baseButton-minimal"] {
    background-color: #3a3a3a !important;
    color: #ffffff !important;
    border: none !important;
}
 
/* Button */
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #f5e642 !important;
    color: #1c1c1e !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1rem !important;
}
 
/* Result cards */
.result-row { display: flex; gap: 1rem; margin-top: 1rem; }
.result-card {
    flex: 1;
    background: #1c1c1e;
    border-radius: 14px;
    padding: 1.4rem 1rem;
    text-align: center;
}
.result-label {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #888888 !important;
    margin-bottom: 0.5rem;
}
.result-value {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.8rem;
    color: #f5e642 !important;
    line-height: 1;
}
.result-unit { font-size: 0.75rem; color: #666666 !important; margin-top: 0.3rem; }
.result-good { border-top: 3px solid #4ade80; }
.result-mid  { border-top: 3px solid #facc15; }
.result-low  { border-top: 3px solid #f87171; }
 
.coord-badge {
    background: #1c1c1e;
    color: #f5e642 !important;
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    font-size: 0.8rem;
    font-family: monospace;
    display: inline-block;
    margin-top: 0.5rem;
}
.map-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1c1c1e !important;
    padding-left: 0.7rem;
    border-left: 3px solid #f5e642;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)
 
 
@st.cache_resource
def load_model():
    with open("models/catboost_final.pkl", "rb") as f:
        return pickle.load(f)
 
@st.cache_data
def load_dataset():
    df = pd.read_excel("Dataset_Properti_Jaksel.xlsx")
    df['Property'] = df['Property'].replace('Apartemen', 'Apartment')
    df_bldg = df.groupby(
        ['Building_Name', 'Property', 'Area', 'Lat', 'Long']
    ).agg(
        Total_Unit=('Total_Unit', 'first'),
        Sold_Rate=('Sold_Rate', 'mean'),
        Rental_Rate=('Rental_Rate', 'mean'),
        Occ=('Occ', 'mean'),
        ROI=('ROI', 'mean'),
    ).reset_index()
    return df_bldg
 
try:
    artifacts        = load_model()
    cb_roi           = artifacts["catboost_roi"]
    cb_occ           = artifacts["catboost_occ"]
    pre_pipeline     = artifacts["pre_pipeline"]
    onehot_cols      = artifacts["onehot_cols"]
    ordinal_cols     = artifacts["ordinal_cols"]
    passthrough_cols = artifacts["passthrough_cols"]
    df_buildings     = load_dataset()
    MODEL_LOADED     = True
except Exception as e:
    st.error(f"Gagal load model/data: {e}")
    MODEL_LOADED = False
    df_buildings = pd.DataFrame()
 
 
def engineer_features(row):
    grade_map = {
        "Premium":"Tier_1","Prime":"Tier_1","5*":"Tier_1",
        "Upper":"Tier_2","A":"Tier_2","4*":"Tier_2",
        "Middle Up":"Tier_3","B":"Tier_3","Primary":"Tier_3","B+":"Tier_3",
        "Middle":"Tier_4","C":"Tier_4","3*":"Tier_4",
        "Middle Low":"Tier_5","Secondary":"Tier_5",
    }
    grade_order = {"Tier_1":5,"Tier_2":4,"Tier_3":3,"Tier_4":2,"Tier_5":1}
    transit_score = (row["General_Bus"]*1 + row["TJ"]*1.5 +
                     row["MRT"]*2 + row["LRT"]*2 + row["KRL"]*1.5)
    road_cat = pd.cut([row["Distance_Main_Road"]],
                      bins=[-1,0,100,300,600,float("inf")],
                      labels=["On Road","Very Close","Close","Moderate","Far"])[0]
    transport_cat = pd.cut([row["Am_Public_Transport"]],
                           bins=[-1,1,2,3,float("inf")],
                           labels=["Very Limited","Limited","Moderate","Good"])[0]
    grade_ordinal = grade_order.get(grade_map.get(row["Grade"], "Tier_3"), 3)
    return pd.DataFrame([{
        "Property": row["Property"], "Type": row["Type"], "Area": row["Area"],
        "Road_Category": road_cat, "Transport_Category": transport_cat,
        "Lat": row["Lat"], "Long": row["Long"],
        "Grade_Ordinal": grade_ordinal, "Transit_Score": transit_score,
        "Log_Total_Unit": np.log1p(row["Total_Unit"]),
        "Log_Sold_Rate": np.log1p(row["Sold_Rate"]),
        "Log_Rental_Rate": np.log1p(row["Rental_Rate"]),
    }])

 
def predict(row):
    X = pre_pipeline.transform(engineer_features(row))
    return float(cb_roi.predict(X)[0]), float(cb_occ.predict(X)[0])
 
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))
 
 
PROPERTY_TYPES = ["Apartment", "Hotel", "Office", "Retail"]
LEASE_TYPES    = ["Lease", "Strata"]
GRADES = ["Premium","Prime","5*","Upper","A","4*","Middle Up","B+","B",
          "Primary","Middle","C","3*","Middle Low","Secondary"]
AREAS  = ["Antasari","Bintaro","Blok M","Brawijaya","Casablanca","Cikini",
          "Cilandak","Cinere","Cipete","Cipulir","Epicentrum","Fatmawati",
          "Gandaria","Gatot Subroto","Iskandarsyah","Kalibata","Karet Semanggi",
          "Kebayoran Baru","Kebayoran Lama","Kemang","Kuningan","Lebak Bulus",
          "MT Haryono","Mahakam","Manggarai","Margonda","Mega Kuningan",
          "Melawai","Menteng","Other","Pakubuwono","Pancoran","Pejaten",
          "Permata Hijau","Pondok Indah","Prapanca","Rasuna Said","SCBD",
          "Satrio","Semanggi","Senayan","Senen","Senopati","Setiabudi",
          "Simprug","Subroto","Sudirman","Sunter","TB Simatupang",
          "Tanjung Duren","Tebet","Tendean","Thamrin","Wijaya"]
MARKER_COLORS = {
    "Apartment":"#3b82f6","Hotel":"#f59e0b",
    "Office":"#10b981","Retail":"#ef4444",
}
 
for k, v in [("lat",-6.2297),("lng",106.8295),("roi",None),("occ",None)]:
    if k not in st.session_state:
        st.session_state[k] = v
 
 
# HEADER
st.markdown("""
<div class="app-header">
  <p class="app-title">🏢 Properti Jaksel</p>
  <p class="app-subtitle">Prediksi ROI &amp; Occupancy Rate · CatBoost ML Model · CV R² ROI 0.9927</p>
</div>
""", unsafe_allow_html=True)
 
col_form, col_map = st.columns([1, 1.5], gap="large")
 
# FORM
with col_form:  
    st.markdown('<div class="section-title">📍 Lokasi</div>', unsafe_allow_html=True)
    st.markdown('<p class="hint">Klik peta di kanan → Lat/Long terisi otomatis.</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    lat_input = c1.number_input("Latitude",  value=st.session_state.lat, format="%.6f", step=0.0001)
    lng_input = c2.number_input("Longitude", value=st.session_state.lng, format="%.6f", step=0.0001)
    st.session_state.lat = lat_input
    st.session_state.lng = lng_input
 
    st.markdown('<div class="section-title">🏗️ Identitas Properti</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    prop_type  = c3.selectbox("Property", PROPERTY_TYPES)
    lease_type = c4.selectbox("Type", LEASE_TYPES)
    grade      = c5.selectbox("Grade", GRADES)
    area       = st.selectbox("Area", AREAS)
 
    st.markdown('<div class="section-title">📊 Data Properti</div>', unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    total_unit  = c6.number_input("Total Unit", min_value=1, value=100, step=1)
    sold_rate   = c7.number_input("Sold Rate (juta/m²)", min_value=0.0, value=50.0, step=0.5)
    rental_rate = st.number_input("Rental Rate (ribu/m²/bln)", min_value=0.0, value=300.0, step=10.0)
 
    st.markdown('<div class="section-title">🛣️ Aksesibilitas</div>', unsafe_allow_html=True)
    c8, c9 = st.columns([2.5, 1])
    c8.text_input("Nama Jalan Utama (opsional)", placeholder="e.g. Jl. Sudirman")
    distance = c9.number_input("Jarak (m)", min_value=0, value=0, step=50)
    st.markdown("**Transportasi umum radius 500 m:**")
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    gen_bus = tc1.number_input("Bus", min_value=0, max_value=10, value=1, step=1)
    tj      = tc2.number_input("TJ",  min_value=0, max_value=10, value=0, step=1)
    mrt     = tc3.number_input("MRT", min_value=0, max_value=10, value=0, step=1)
    lrt     = tc4.number_input("LRT", min_value=0, max_value=10, value=0, step=1)
    krl     = tc5.number_input("KRL", min_value=0, max_value=10, value=0, step=1)
    am_public = sum([gen_bus>0, tj>0, mrt>0, lrt>0, krl>0])
 
    st.markdown("")
    predict_btn = st.button("🔍 Prediksi ROI & Occupancy", type="primary", use_container_width=True)
 
    if predict_btn and MODEL_LOADED:
        input_row = {
            "Property":prop_type,"Type":lease_type,"Area":area,"Grade":grade,
            "Lat":lat_input,"Long":lng_input,
            "Total_Unit":total_unit,
            "Sold_Rate":sold_rate*1_000_000,
            "Rental_Rate":rental_rate*1_000,
            "Distance_Main_Road":distance,"Am_Public_Transport":am_public,
            "General_Bus":gen_bus,"TJ":tj,"MRT":mrt,"LRT":lrt,"KRL":krl,
        }
        with st.spinner("Menghitung prediksi..."):
            roi, occ = predict(input_row)
        st.session_state.roi = roi
        st.session_state.occ = occ
 
    if st.session_state.roi is not None:
        rv, ov = st.session_state.roi, st.session_state.occ
        rc = "result-good" if rv >= 0.08 else ("result-mid" if rv >= 0.05 else "result-low")
        oc = "result-good" if ov >= 0.80 else ("result-mid" if ov >= 0.60 else "result-low")
        st.markdown(f"""
        <div class="result-row">
          <div class="result-card {rc}">
            <div class="result-label">ROI</div>
            <div class="result-value">{rv:.1%}</div>
            <div class="result-unit">Return on Investment</div>
          </div>
          <div class="result-card {oc}">
            <div class="result-label">Occupancy</div>
            <div class="result-value">{ov:.1%}</div>
            <div class="result-unit">Tingkat Hunian</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
# PETA
with col_map:
    st.markdown('<div class="map-title">🗺️ Klik peta untuk menentukan lokasi</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="hint">Marker berwarna = {prop_type} dalam radius 3 km · Klik marker untuk detail properti</p>',
        unsafe_allow_html=True)
 
    m = folium.Map(location=[st.session_state.lat, st.session_state.lng],
                   zoom_start=13, tiles="CartoDB positron")
 
    # Radius circle
    folium.Circle(
        location=[st.session_state.lat, st.session_state.lng],
        radius=3000, color="#f5e642", weight=2,
        fill=True, fill_color="#f5e642", fill_opacity=0.06,
        tooltip="Radius 3 km",
    ).add_to(m)
 
    # Marker lokasi pilihan
    folium.Marker(
        location=[st.session_state.lat, st.session_state.lng],
        popup=f"📍 Lokasi pilihan ({st.session_state.lat:.5f}, {st.session_state.lng:.5f})",
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
        tooltip="📍 Lokasi pilihan",
    ).add_to(m)
 
    # Properti dalam radius
    if MODEL_LOADED and not df_buildings.empty:
        df_f = df_buildings[df_buildings["Property"] == prop_type].copy()
        df_f["dist_km"] = df_f.apply(
            lambda r: haversine_km(st.session_state.lat, st.session_state.lng, r["Lat"], r["Long"]), axis=1)
        df_r = df_f[df_f["dist_km"] <= 3.0]
        mc   = MARKER_COLORS.get(prop_type, "#6366f1")
 
        for _, row in df_r.iterrows():
            sold_s   = f"Rp {row['Sold_Rate']/1e6:.1f} jt/m²"   if row['Sold_Rate'] > 0   else "N/A"
            rent_s   = f"Rp {row['Rental_Rate']/1e3:.0f} rb/m²"  if row['Rental_Rate'] > 0 else "N/A"
            occ_s    = f"{row['Occ']:.1%}"  if row['Occ'] > 0  else "N/A"
            roi_s    = f"{row['ROI']:.1%}"  if row['ROI'] > 0  else "N/A"
            popup_html = f"""
            <div style="font-family:sans-serif;min-width:210px;font-size:13px;line-height:1.7;">
              <b style="font-size:14px;">{row['Building_Name']}</b><br>
              <span style="color:#666;">{row['Property']} · {row['Area']}</span>
              <hr style="margin:6px 0;">
              🏠 Total Unit &nbsp;: <b>{int(row['Total_Unit'])}</b><br>
              💰 Sold Rate &nbsp;&nbsp;: <b>{sold_s}</b><br>
              🏷️ Rental Rate : <b>{rent_s}</b><br>
              📊 Occ (hist)&nbsp; : <b>{occ_s}</b><br>
              📈 ROI (hist)&nbsp; : <b>{roi_s}</b>
            </div>"""
            folium.CircleMarker(
                location=[row["Lat"], row["Long"]],
                radius=7, color="white", weight=1.5,
                fill=True, fill_color=mc, fill_opacity=0.85,
                tooltip=f"🏢 {row['Building_Name']}",
                popup=folium.Popup(popup_html, max_width=260),
            ).add_to(m)
 
        # Mini legend
        legend = f"""
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:white;padding:10px 14px;border-radius:10px;
                    border:1px solid #ddd;font-family:sans-serif;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
          <b>{prop_type} dalam radius 3 km</b><br>
          <span style="color:{mc};">●</span> {len(df_r)} properti ditemukan
        </div>"""
        m.get_root().html.add_child(folium.Element(legend))
 
    map_data = st_folium(m, width="100%", height=560, returned_objects=["last_clicked"])
 
    if map_data and map_data.get("last_clicked"):
        clat = map_data["last_clicked"]["lat"]
        clng = map_data["last_clicked"]["lng"]
        if abs(clat - st.session_state.lat) > 0.00001 or abs(clng - st.session_state.lng) > 0.00001:
            st.session_state.lat = clat
            st.session_state.lng = clng
            st.rerun()
 
    st.markdown(
        f'<div class="coord-badge">📍 {st.session_state.lat:.5f}, {st.session_state.lng:.5f}</div>',
        unsafe_allow_html=True)
 
    st.markdown("**Legend warna marker:**")
    lc = st.columns(4)
    for i, (pt, pc) in enumerate(MARKER_COLORS.items()):
        lc[i].markdown(f'<span style="color:{pc};font-size:1.1rem;">●</span> {pt}', unsafe_allow_html=True)
 
st.divider()
st.caption("Model: CatBoost (tuned) · Dataset: Properti Jakarta Selatan · CV R² ROI: 0.9927 · Occ: 0.6632")
