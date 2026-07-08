"""
Analisis Debit Andalan — SNI 6738:2015
Streamlit App — Input: koordinat outlet + upload file KMZ/KML DAS
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, io, math, zipfile, re
import xml.etree.ElementTree as ET

from hydrology import (
    thiessen_weights, weighted_areal_rainfall,
    etp_penman_monteith, mock_water_balance,
    flow_duration_curve, get_q_at_exceedance,
    log_pearson_iii_params, log_pearson_iii_quantile,
    run_analysis,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Debit Andalan — SNI 6738:2015",
    page_icon="💧",
    layout="wide",
)

MONTHS = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agt","Sep","Okt","Nov","Des"]

EXAMPLE_PAYLOAD = {
    "das_area_km2": 450.0,
    "titik_tinjau": {"lat": -0.15, "lon": 110.47, "nama": "Outlet Sungai Kapuas"},
    "stations": [
        {"id":"S1","name":"Sta. Gunung Mas",      "lat":-0.10,"lon":110.45},
        {"id":"S2","name":"Sta. Sungai Raya",     "lat":-0.25,"lon":110.55},
        {"id":"S3","name":"Sta. Pontianak Utara", "lat":-0.05,"lon":110.38},
    ],
    "station_rain": {
        "S1":[280,255,320,290,210,140,120,130,190,260,310,295],
        "S2":[265,240,300,275,195,130,110,125,180,245,290,280],
        "S3":[290,260,330,300,220,150,130,140,200,270,320,305],
    },
    "climate": [
        {"month":1, "T":27.2,"RH":84,"Rs":13.5,"uz":1.2},
        {"month":2, "T":27.5,"RH":82,"Rs":14.2,"uz":1.3},
        {"month":3, "T":27.8,"RH":83,"Rs":15.0,"uz":1.1},
        {"month":4, "T":27.9,"RH":82,"Rs":15.5,"uz":1.2},
        {"month":5, "T":28.1,"RH":80,"Rs":15.8,"uz":1.4},
        {"month":6, "T":27.8,"RH":79,"Rs":14.9,"uz":1.5},
        {"month":7, "T":27.5,"RH":78,"Rs":14.5,"uz":1.6},
        {"month":8, "T":27.7,"RH":79,"Rs":15.0,"uz":1.5},
        {"month":9, "T":27.9,"RH":81,"Rs":14.8,"uz":1.3},
        {"month":10,"T":27.8,"RH":83,"Rs":14.0,"uz":1.2},
        {"month":11,"T":27.4,"RH":84,"Rs":13.5,"uz":1.1},
        {"month":12,"T":27.1,"RH":85,"Rs":13.0,"uz":1.1},
    ],
    "das_params": {"sm_cap":200,"i":0.45,"k":0.55,"gws_init":50,"sm_init":100},
}

# ─────────────────────────────────────────────
# HELPER: Parse KMZ / KML → polygon coords + luas
# ─────────────────────────────────────────────
NS = {
    "kml":  "http://www.opengis.net/kml/2.2",
    "kml22":"http://earth.google.com/kml/2.2",
    "kml21":"http://earth.google.com/kml/2.1",
}

def _parse_kml_tree(root) -> list[list[list[float]]]:
    """Ekstrak semua koordinat Polygon dari tree KML."""
    polygons = []
    # cari semua tag koordinat di bawah Polygon
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]  # strip namespace
        if tag == "coordinates":
            parent_tags = [e.tag.split("}")[-1] for e in root.iter()
                           if elem in list(e)]
            raw = elem.text.strip() if elem.text else ""
            if not raw:
                continue
            coords = []
            for token in raw.split():
                parts = token.split(",")
                if len(parts) >= 2:
                    try:
                        coords.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        pass
            if len(coords) >= 3:
                polygons.append(coords)
    return polygons


def parse_kmz_kml(uploaded_file) -> tuple[list, float | None]:
    """
    Baca file KMZ atau KML yang diupload.
    Return: (list_polygon_coords, luas_km2 | None)
    """
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if name.endswith(".kmz"):
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
            kml_name = next((n for n in z.namelist() if n.endswith(".kml")), None)
            if not kml_name:
                return [], None
            kml_bytes = z.read(kml_name)
    else:
        kml_bytes = raw_bytes

    try:
        root = ET.fromstring(kml_bytes)
    except ET.ParseError:
        return [], None

    polygons = _parse_kml_tree(root)

    # Hitung luas (km²) — shoelace formula, asumsi koordinat desimal derajat
    def shoelace_km2(coords):
        R = 6371.0
        n = len(coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            lon1, lat1 = math.radians(coords[i][0]), math.radians(coords[i][1])
            lon2, lat2 = math.radians(coords[j][0]), math.radians(coords[j][1])
            area += (lon2 - lon1) * (math.sin(lat1) + math.sin(lat2))
        return abs(area) * R * R / 2.0

    total_luas = None
    if polygons:
        total_luas = round(sum(shoelace_km2(p) for p in polygons), 2)

    return polygons, total_luas


# ─────────────────────────────────────────────
# HELPER: Thiessen Voronoi polygon
# ─────────────────────────────────────────────
def buat_thiessen_polygon(stations: list[dict], bbox_pad: float = 0.3):
    from scipy.spatial import Voronoi
    pts = np.array([[s["lon"], s["lat"]] for s in stations])
    n = len(pts)

    if n == 1:
        lon, lat = pts[0]
        return [[[lon-bbox_pad, lat-bbox_pad],[lon+bbox_pad, lat-bbox_pad],
                 [lon+bbox_pad, lat+bbox_pad],[lon-bbox_pad, lat+bbox_pad],
                 [lon-bbox_pad, lat-bbox_pad]]]

    min_lon = pts[:,0].min() - bbox_pad
    max_lon = pts[:,0].max() + bbox_pad
    min_lat = pts[:,1].min() - bbox_pad
    max_lat = pts[:,1].max() + bbox_pad

    mirror = np.array([
        [min_lon-1, (min_lat+max_lat)/2],
        [max_lon+1, (min_lat+max_lat)/2],
        [(min_lon+max_lon)/2, min_lat-1],
        [(min_lon+max_lon)/2, max_lat+1],
    ])
    vor = Voronoi(np.vstack([pts, mirror]))

    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0

    def intersect(a, b, c, d):
        A1,B1 = b[1]-a[1], a[0]-b[0]
        C1 = A1*a[0]+B1*a[1]
        A2,B2 = d[1]-c[1], c[0]-d[0]
        C2 = A2*c[0]+B2*c[1]
        det = A1*B2 - A2*B1
        if abs(det) < 1e-10: return a
        return ((C1*B2-C2*B1)/det, (A1*C2-A2*C1)/det)

    def clip(poly):
        edges = [
            ([min_lon,min_lat],[min_lon,max_lat]),
            ([min_lon,max_lat],[max_lon,max_lat]),
            ([max_lon,max_lat],[max_lon,min_lat]),
            ([max_lon,min_lat],[min_lon,min_lat]),
        ]
        out = list(poly)
        for (a,b) in edges:
            if not out: break
            inp, out = out, []
            for i in range(len(inp)):
                cur, prev = inp[i], inp[i-1]
                if inside(cur,a,b):
                    if not inside(prev,a,b):
                        out.append(intersect(prev,cur,a,b))
                    out.append(cur)
                elif inside(prev,a,b):
                    out.append(intersect(prev,cur,a,b))
        return out

    polygons = []
    for i in range(n):
        region = vor.regions[vor.point_region[i]]
        if not region: polygons.append([]); continue
        verts = [vor.vertices[v] for v in region if v != -1]
        clipped = clip(verts)
        if clipped:
            clipped.append(clipped[0])
            polygons.append([[p[0],p[1]] for p in clipped])
        else:
            polygons.append([])
    return polygons


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for _k, _v in [
    ("stations", [{"id":"S1","name":"Stasiun 1","lat":0.0,"lon":0.0}]),
    ("rain_data", {"S1":[0.0]*12}),
    ("result", None), ("payload", None),
    ("tt_lat", 0.0), ("tt_lon", 0.0),
    ("tt_nama", "Titik Tinjau / Outlet DAS"),
    ("das_polygons", None), ("das_luas_kmz", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


def load_example():
    p = EXAMPLE_PAYLOAD
    st.session_state.stations  = [dict(s) for s in p["stations"]]
    st.session_state.rain_data = {k: list(v) for k,v in p["station_rain"].items()}
    st.session_state.das_area  = p["das_area_km2"]
    st.session_state.sm_cap    = p["das_params"]["sm_cap"]
    st.session_state.i_coef    = p["das_params"]["i"]
    st.session_state.k_coef    = p["das_params"]["k"]
    st.session_state.gws_init  = p["das_params"]["gws_init"]
    st.session_state.sm_init   = p["das_params"]["sm_init"]
    st.session_state.climate   = p["climate"]
    tt = p.get("titik_tinjau", {})
    st.session_state.tt_lat    = tt.get("lat", 0.0)
    st.session_state.tt_lon    = tt.get("lon", 0.0)
    st.session_state.tt_nama   = tt.get("nama", "Titik Tinjau / Outlet DAS")
    st.session_state.das_polygons  = None
    st.session_state.das_luas_kmz  = None


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg,#0d9e87 0%,#4f9cf9 100%);
    padding:1.2rem 1.6rem; border-radius:12px; margin-bottom:1.5rem;">
    <h1 style="color:white;margin:0;font-size:1.5rem;">💧 Analisis Debit Andalan</h1>
    <p style="color:rgba(255,255,255,0.85);margin:4px 0 0;font-size:0.9rem;">
        Metode F.J. Mock — Neraca Air Bulanan &nbsp;|&nbsp; SNI 6738:2015
    </p>
</div>
""", unsafe_allow_html=True)

col_btn, col_info = st.columns([1,3])
with col_btn:
    if st.button("📥 Muat Contoh Data", use_container_width=True):
        load_example()
        st.success("Contoh data berhasil dimuat!")
with col_info:
    st.info("Isi form → tekan **▶ Jalankan Analisis**. Atau muat contoh data Pontianak (Kalimantan Barat).")

st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameter DAS")

    # ── Titik Tinjau ──────────────────────────
    st.markdown("#### 📍 Titik Tinjau (Outlet DAS)")
    st.caption("Koordinat outlet/bendung — referensi bobot Thiessen (IDW).")

    tt_nama = st.text_input("Nama Titik Tinjau",
                             value=st.session_state.get("tt_nama","Titik Tinjau / Outlet DAS"),
                             key="tt_nama")
    col_tt1, col_tt2 = st.columns(2)
    with col_tt1:
        tt_lat = st.number_input("Latitude",  value=float(st.session_state.get("tt_lat",0.0)),
                                  format="%.4f", key="tt_lat")
    with col_tt2:
        tt_lon = st.number_input("Longitude", value=float(st.session_state.get("tt_lon",0.0)),
                                  format="%.4f", key="tt_lon")

    if tt_lat == 0.0 and tt_lon == 0.0:
        st.warning("⚠️ Koordinat outlet belum diisi.")
    else:
        st.success(f"📍 ({tt_lat:.4f}, {tt_lon:.4f})")

    st.divider()

    # ── Upload KMZ / KML ──────────────────────
    st.markdown("#### 🗺️ Batas DAS (KMZ / KML)")
    st.caption("Upload file KMZ/KML hasil digitasi/export Google Earth atau QGIS.")

    kmz_file = st.file_uploader("Upload KMZ / KML", type=["kmz","kml"],
                                  label_visibility="collapsed")

    if kmz_file is not None:
        with st.spinner("Membaca file…"):
            polys, luas = parse_kmz_kml(kmz_file)
        if polys:
            st.session_state.das_polygons = polys
            st.session_state.das_luas_kmz = luas
            if luas:
                # Isi otomatis luas DAS
                st.session_state["das_area"] = luas
                st.success(f"✅ {len(polys)} polygon terbaca — Luas ≈ **{luas:.2f} km²**")
            else:
                st.success(f"✅ {len(polys)} polygon terbaca.")
        else:
            st.error("Gagal membaca polygon dari file. Pastikan file KMZ/KML berisi Polygon.")

    if st.session_state.das_luas_kmz:
        st.info(f"📐 Luas DAS dari KMZ: **{st.session_state.das_luas_kmz:.2f} km²**")

    st.divider()

    # ── Parameter DAS & Mock ──────────────────
    das_area = st.number_input("Luas DAS (km²)", min_value=1.0, max_value=100000.0,
                                value=float(st.session_state.get("das_area",450.0)),
                                step=10.0, key="das_area",
                                help="Terisi otomatis jika upload KMZ/KML.")

    st.markdown("#### 🌊 Parameter Model Mock")
    sm_cap   = st.number_input("Kapasitas Lengas Tanah (mm)", 50, 500,
                                int(st.session_state.get("sm_cap",200)), 10, key="sm_cap")
    i_coef   = st.slider("Koef. Infiltrasi (i)", 0.0, 1.0,
                          float(st.session_state.get("i_coef",0.45)), 0.05, key="i_coef")
    k_coef   = st.slider("Koef. Resesi Air Tanah (k)", 0.0, 1.0,
                          float(st.session_state.get("k_coef",0.55)), 0.05, key="k_coef")
    gws_init = st.number_input("Tampungan Air Tanah Awal (mm)", 0, 500,
                                int(st.session_state.get("gws_init",50)), 10, key="gws_init")
    sm_init  = st.number_input("Lengas Tanah Awal (mm)", 0, 500,
                                int(st.session_state.get("sm_init",100)), 10, key="sm_init")
    st.caption("**i** ≈ 0.3–0.5 | **k** ≈ 0.5–0.8 | **SM Cap** ≈ 100–300 mm")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Stasiun & Hujan", "🌡️ Data Iklim",
    "▶ Jalankan Analisis", "📊 Hasil",
])

# ══════════════════════════════════════════════
# TAB 1: Stasiun & Hujan
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Data Stasiun Hujan & Curah Hujan Bulanan")
    col_add, col_del = st.columns(2)
    with col_add:
        if st.button("➕ Tambah Stasiun", use_container_width=True):
            n = len(st.session_state.stations)+1
            sid = f"S{n}"
            st.session_state.stations.append({"id":sid,"name":f"Stasiun {n}","lat":0.0,"lon":0.0})
            st.session_state.rain_data[sid] = [0.0]*12
    with col_del:
        if len(st.session_state.stations) > 1:
            if st.button("➖ Hapus Stasiun Terakhir", use_container_width=True):
                removed = st.session_state.stations.pop()
                st.session_state.rain_data.pop(removed["id"], None)

    st.divider()
    for idx, sta in enumerate(st.session_state.stations):
        sid = sta["id"]
        with st.expander(f"🔵 {sta['name']} ({sid})", expanded=True):
            c1,c2,c3,c4 = st.columns([2,2,1,1])
            with c1: sta["name"] = st.text_input("Nama Stasiun", sta["name"], key=f"sname_{idx}")
            with c2:
                new_id = st.text_input("ID Stasiun", sid, key=f"sid_{idx}")
                if new_id != sid:
                    st.session_state.rain_data[new_id] = st.session_state.rain_data.pop(sid,[0.0]*12)
                    sta["id"] = new_id; sid = new_id
            with c3: sta["lat"] = st.number_input("Latitude",  value=float(sta["lat"]), format="%.4f", key=f"lat_{idx}")
            with c4: sta["lon"] = st.number_input("Longitude", value=float(sta["lon"]), format="%.4f", key=f"lon_{idx}")

            st.markdown("**Curah Hujan Bulanan (mm):**")
            rain_vals = st.session_state.rain_data.get(sid, [0.0]*12)
            new_rain = []
            for m, col in enumerate(st.columns(12)):
                v = col.number_input(MONTHS[m], min_value=0.0, max_value=2000.0,
                                     value=float(rain_vals[m]) if m < len(rain_vals) else 0.0,
                                     step=5.0, key=f"rain_{sid}_{m}")
                new_rain.append(v)
            st.session_state.rain_data[sid] = new_rain

# ══════════════════════════════════════════════
# TAB 2: Data Iklim
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Data Iklim Bulanan")
    default_climate = st.session_state.get("climate", EXAMPLE_PAYLOAD["climate"])
    climate_df = pd.DataFrame([{
        "Bulan": MONTHS[c["month"]-1],
        "T (°C)": c["T"], "RH (%)": c["RH"],
        "Rs (MJ/m²/hr)": c["Rs"], "uz (m/s)": c["uz"],
    } for c in default_climate])

    edited_climate = st.data_editor(
        climate_df, use_container_width=True, num_rows="fixed",
        column_config={
            "Bulan":          st.column_config.TextColumn("Bulan", disabled=True),
            "T (°C)":         st.column_config.NumberColumn("T (°C)",        min_value=0.0, max_value=50.0, format="%.1f"),
            "RH (%)":         st.column_config.NumberColumn("RH (%)",         min_value=0,   max_value=100),
            "Rs (MJ/m²/hr)":  st.column_config.NumberColumn("Rs (MJ/m²/hr)", min_value=0.0, format="%.1f"),
            "uz (m/s)":       st.column_config.NumberColumn("uz (m/s)",       min_value=0.0, format="%.2f"),
        }, key="climate_editor",
    )
    st.info("**T** = Suhu | **RH** = Kelembaban | **Rs** = Radiasi matahari | **uz** = Kecepatan angin (z=2m)")

# ══════════════════════════════════════════════
# TAB 3: Jalankan
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Ringkasan Input & Jalankan Analisis")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Parameter DAS:**")
        st.table(pd.DataFrame({
            "Parameter": ["Titik Tinjau","Koordinat","Luas DAS","SM Kap","i","k","GWS Awal","SM Awal"],
            "Nilai": [
                st.session_state.tt_nama,
                f"({st.session_state.tt_lat:.4f}, {st.session_state.tt_lon:.4f})",
                f"{st.session_state.das_area} km²",
                f"{st.session_state.sm_cap} mm",
                st.session_state.i_coef, st.session_state.k_coef,
                f"{st.session_state.gws_init} mm", f"{st.session_state.sm_init} mm",
            ],
        }))
    with col_s2:
        st.markdown("**Stasiun Hujan:**")
        st.dataframe(pd.DataFrame([
            {"ID":s["id"],"Nama":s["name"],"Lat":s["lat"],"Lon":s["lon"]}
            for s in st.session_state.stations
        ]), use_container_width=True, hide_index=True)
        if st.session_state.das_polygons:
            st.success(f"🗺️ File KMZ/KML sudah diupload — {len(st.session_state.das_polygons)} polygon")
        else:
            st.warning("🗺️ File KMZ/KML belum diupload (opsional — upload di sidebar).")

    st.divider()
    if st.button("▶ Jalankan Analisis", type="primary", use_container_width=True):
        errors = []
        if not st.session_state.stations:
            errors.append("Minimal 1 stasiun hujan diperlukan.")
        for s in st.session_state.stations:
            rain = st.session_state.rain_data.get(s["id"],[])
            if len(rain) < 12 or all(v==0 for v in rain):
                errors.append(f"Data hujan stasiun {s['name']} ({s['id']}) belum diisi.")
        if errors:
            for e in errors: st.error(e)
        else:
            with st.spinner("⏳ Menjalankan model Mock…"):
                try:
                    climate_list = [{
                        "month": int(m)+1,
                        "T":  float(row["T (°C)"]),
                        "RH": float(row["RH (%)"]),
                        "Rs": float(row["Rs (MJ/m²/hr)"]),
                        "uz": float(row["uz (m/s)"]),
                    } for m, row in edited_climate.iterrows()]

                    payload = {
                        "das_area_km2": st.session_state.das_area,
                        "titik_tinjau": {
                            "lat":  st.session_state.tt_lat,
                            "lon":  st.session_state.tt_lon,
                            "nama": st.session_state.tt_nama,
                        },
                        "stations": [dict(s) for s in st.session_state.stations],
                        "station_rain": {
                            s["id"]: st.session_state.rain_data[s["id"]]
                            for s in st.session_state.stations
                        },
                        "climate": climate_list,
                        "das_params": {
                            "sm_cap":   st.session_state.sm_cap,
                            "i":        st.session_state.i_coef,
                            "k":        st.session_state.k_coef,
                            "gws_init": st.session_state.gws_init,
                            "sm_init":  st.session_state.sm_init,
                        },
                    }
                    result = run_analysis(payload)
                    st.session_state.result  = result
                    st.session_state.payload = payload
                    st.success("✅ Analisis selesai! Lihat tab **📊 Hasil**.")
                except Exception as ex:
                    import traceback
                    st.error(f"❌ Error: {ex}")
                    st.code(traceback.format_exc())

# ══════════════════════════════════════════════
# TAB 4: Hasil
# ══════════════════════════════════════════════
with tab4:
    if st.session_state.result is None:
        st.info("Belum ada hasil. Jalankan analisis di tab **▶ Jalankan Analisis** terlebih dahulu.")
        st.stop()

    r     = st.session_state.result
    q     = r["q_andalan"]
    mock  = r["mock_table"]
    lp3_k = r["lp3_kala_ulang"]
    lp3_p = r["lp3_params"]

    # ── Debit Andalan ─────────────────────────
    st.subheader("📊 Debit Andalan (SNI 6738:2015)")
    c80, c90, c95 = st.columns(3)
    c80.metric("Q80 — Irigasi",  f"{q['Q80']} m³/s", delta="Probabilitas 80%", delta_color="off")
    c90.metric("Q90 — PLTA",     f"{q['Q90']} m³/s", delta="Probabilitas 90%", delta_color="off")
    c95.metric("Q95 — Air Baku", f"{q['Q95']} m³/s", delta="Probabilitas 95%", delta_color="off")

    st.divider()

    # ══════════════════════════════════════════
    # PETA: DAS polygon + Thiessen + stasiun + outlet
    # ══════════════════════════════════════════
    st.subheader("🗺️ Peta DAS & Polygon Thiessen")

    stations_result = r["stations"]
    try:
        thiessen_polys = buat_thiessen_polygon(stations_result)
    except Exception:
        thiessen_polys = []

    FILL_COLORS   = ["rgba(28,207,176,0.2)","rgba(79,156,249,0.2)","rgba(245,166,35,0.2)",
                     "rgba(155,127,250,0.2)","rgba(240,112,112,0.2)","rgba(94,201,122,0.2)"]
    BORDER_COLORS = ["#1ccfb0","#4f9cf9","#f5a623","#9b7ffa","#f07070","#5ec97a"]

    fig_map = go.Figure()

    # 1. Polygon DAS dari KMZ
    das_polys = st.session_state.das_polygons
    if das_polys:
        for poly in das_polys:
            lons = [p[0] for p in poly]
            lats = [p[1] for p in poly]
            fig_map.add_trace(go.Scattermapbox(
                lon=lons, lat=lats, mode="lines",
                line=dict(color="#f5a623", width=2.5),
                fill="toself", fillcolor="rgba(245,166,35,0.07)",
                name="Batas DAS", showlegend=True, hoverinfo="name",
            ))

    # 2. Polygon Thiessen
    for i, (sta, poly) in enumerate(zip(stations_result, thiessen_polys)):
        if not poly: continue
        ci = i % len(FILL_COLORS)
        fig_map.add_trace(go.Scattermapbox(
            lon=[p[0] for p in poly], lat=[p[1] for p in poly],
            mode="lines", fill="toself",
            fillcolor=FILL_COLORS[ci], line=dict(color=BORDER_COLORS[ci], width=1.8),
            name=f"Thiessen {sta['name']} ({sta['weight']*100:.1f}%)",
            hovertemplate=(
                f"<b>{sta['name']}</b><br>Bobot: {sta['weight']*100:.1f}%<br>"
                f"Lat: {sta['lat']}<br>Lon: {sta['lon']}<extra></extra>"
            ),
        ))

    # 3. Titik stasiun
    fig_map.add_trace(go.Scattermapbox(
        lon=[s["lon"] for s in stations_result],
        lat=[s["lat"] for s in stations_result],
        mode="markers+text",
        marker=dict(size=12, color="#4f9cf9"),
        text=[s["name"] for s in stations_result],
        textposition="top right",
        name="Stasiun Hujan",
        hovertext=[f"{s['name']}<br>Bobot: {s['weight']*100:.1f}%" for s in stations_result],
        hoverinfo="text",
    ))

    # 4. Titik tinjau / outlet
    tt_info = st.session_state.payload.get("titik_tinjau", {}) if st.session_state.payload else {}
    tt_lon_v = tt_info.get("lon", 0.0)
    tt_lat_v = tt_info.get("lat", 0.0)
    tt_nm_v  = tt_info.get("nama", "Titik Tinjau")
    if tt_lon_v != 0.0 or tt_lat_v != 0.0:
        fig_map.add_trace(go.Scattermapbox(
            lon=[tt_lon_v], lat=[tt_lat_v],
            mode="markers+text",
            marker=dict(size=16, color="#f07070"),
            text=[tt_nm_v], textposition="top right",
            name="Titik Tinjau (Outlet)",
            hovertemplate=f"<b>{tt_nm_v}</b><br>Lat: {tt_lat_v}<br>Lon: {tt_lon_v}<extra></extra>",
        ))

    # Center peta
    all_lons = [s["lon"] for s in stations_result] + ([tt_lon_v] if tt_lon_v!=0 else [])
    all_lats = [s["lat"] for s in stations_result] + ([tt_lat_v] if tt_lat_v!=0 else [])
    fig_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=sum(all_lats)/len(all_lats), lon=sum(all_lons)/len(all_lons)),
            zoom=9,
        ),
        height=520, margin=dict(l=0,r=0,t=0,b=0),
        legend=dict(bgcolor="rgba(26,29,39,0.85)", font=dict(color="#c8cad5",size=11),
                    bordercolor="rgba(255,255,255,0.15)", borderwidth=1),
        paper_bgcolor="#1a1d27",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Tabel & download Thiessen
    with st.expander("📋 Detail Bobot Thiessen"):
        tlat = tt_info.get("lat",0.0); tlon = tt_info.get("lon",0.0)
        if tlat!=0.0 or tlon!=0.0:
            st.info(f"📍 Referensi IDW: **{tt_info.get('nama','-')}** — Lat `{tlat}`, Lon `{tlon}`")
        else:
            st.warning("⚠️ Bobot dihitung dari centroid otomatis (koordinat outlet belum diisi).")
        st.dataframe(pd.DataFrame([
            {"ID":s["id"],"Nama":s["name"],"Lat":s["lat"],"Lon":s["lon"],
             "Bobot (%)": round(s["weight"]*100,2)}
            for s in stations_result
        ]), use_container_width=True, hide_index=True)

    # Download GeoJSON Thiessen
    if any(thiessen_polys):
        thiessen_gj = {
            "type":"FeatureCollection",
            "features": [{
                "type":"Feature",
                "properties":{"id":s["id"],"name":s["name"],"weight":s["weight"],
                              "weight_pct":round(s["weight"]*100,2)},
                "geometry":{"type":"Polygon","coordinates":[poly]},
            } for s,poly in zip(stations_result,thiessen_polys) if poly],
        }
        st.download_button(
            "⬇️ Download GeoJSON Polygon Thiessen",
            data=json.dumps(thiessen_gj, ensure_ascii=False, indent=2),
            file_name="polygon_thiessen.geojson", mime="application/json",
        )

    st.divider()

    # ── Debit Bulanan ──────────────────────────
    st.subheader("📈 Debit Bulanan (m³/s)")
    fig_q = go.Figure(go.Bar(
        x=MONTHS, y=mock["debit_m3s"],
        marker_color="rgba(28,207,176,0.7)", marker_line_color="#1ccfb0", marker_line_width=1.2,
    ))
    fig_q.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="m³/s",
        plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27", font_color="#c8cad5",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"), yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
    st.plotly_chart(fig_q, use_container_width=True)

    # ── Hujan & ETP ───────────────────────────
    st.subheader("🌧️ Hujan Kawasan & ETP Bulanan")
    fig_re = go.Figure()
    fig_re.add_trace(go.Bar(x=MONTHS, y=r["areal_rain"], name="Hujan Kawasan (mm)",
        marker_color="rgba(79,156,249,0.6)", marker_line_color="#4f9cf9", marker_line_width=1))
    fig_re.add_trace(go.Scatter(x=MONTHS, y=r["etp_monthly"], name="ETP (mm)", mode="lines+markers",
        line=dict(color="#f5a623",width=2), marker=dict(size=6)))
    fig_re.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="mm",
        plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27", font_color="#c8cad5",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"), yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_re, use_container_width=True)

    # ── FDC ───────────────────────────────────
    st.subheader("📉 Kurva Durasi Aliran (FDC)")
    fdc = r["fdc"]
    fig_fdc = go.Figure(go.Scatter(x=fdc["exceedance"], y=fdc["debit"], mode="lines",
        line=dict(color="#9b7ffa",width=2), fill="tozeroy", fillcolor="rgba(155,127,250,0.12)"))
    for pct,label,color in [(80,"Q80","#1ccfb0"),(90,"Q90","#4f9cf9"),(95,"Q95","#f5a623")]:
        fig_fdc.add_vline(x=pct, line_dash="dash", line_color=color, line_width=1.2,
                          annotation_text=f"{label}={q[f'Q{pct}']} m³/s", annotation_font_color=color)
    fig_fdc.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10),
        xaxis_title="% Waktu terlampaui", yaxis_title="m³/s",
        plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27", font_color="#c8cad5",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"), yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
    st.plotly_chart(fig_fdc, use_container_width=True)

    # ── LP3 ───────────────────────────────────
    st.subheader("🔄 Debit Kala Ulang — Log Pearson III")
    st.caption(f"mean={lp3_p['mean']} | σ={lp3_p['std']} | Cs={lp3_p['skew']} | n={lp3_p['n']}")
    for col,(key,val) in zip(st.columns(len(lp3_k)), lp3_k.items()):
        col.metric(key, f"{val} m³/s")
    fig_lp3 = go.Figure(go.Bar(x=list(lp3_k.keys()), y=list(lp3_k.values()),
        marker_color="rgba(155,127,250,0.7)", marker_line_color="#9b7ffa", marker_line_width=1))
    fig_lp3.update_layout(height=240, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="m³/s",
        plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27", font_color="#c8cad5",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"), yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
    st.plotly_chart(fig_lp3, use_container_width=True)

    st.divider()

    # ── Tabel Mock ────────────────────────────
    st.subheader("📋 Tabel Neraca Air Bulanan (Mock)")
    cols_show   = ["month_name","rain","etp","aet","water_surplus",
                   "direct_runoff","infiltration","baseflow","total_flow_mm","debit_m3s"]
    labels_show = ["Bulan","Hujan (mm)","ETP (mm)","AET (mm)","Surplus (mm)",
                   "Run-off (mm)","Infiltrasi (mm)","Base-flow (mm)","Total (mm)","Debit (m³/s)"]
    mock_df = pd.DataFrame({lbl: mock[col] for col,lbl in zip(cols_show,labels_show)})
    st.dataframe(mock_df, use_container_width=True, hide_index=True)

    # ── Export ────────────────────────────────
    st.divider()
    st.subheader("💾 Ekspor Hasil")
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        buf = io.StringIO()
        mock_df.to_csv(buf, index=False)
        st.download_button("⬇️ Download Tabel (CSV)", data=buf.getvalue(),
                           file_name="neraca_air_mock.csv", mime="text/csv",
                           use_container_width=True)
    with col_ex2:
        st.download_button("⬇️ Download Hasil Lengkap (JSON)",
                           data=json.dumps(r, indent=2, ensure_ascii=False),
                           file_name="hasil_debit_andalan.json", mime="application/json",
                           use_container_width=True)
