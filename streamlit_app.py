"""
Analisis Debit Andalan — SNI 6738:2015
Streamlit App dengan integrasi Google Earth Engine (Deliniasi DAS + Polygon Thiessen)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
import math

from hydrology import (
    thiessen_weights,
    weighted_areal_rainfall,
    etp_penman_monteith,
    mock_water_balance,
    flow_duration_curve,
    get_q_at_exceedance,
    log_pearson_iii_params,
    log_pearson_iii_quantile,
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

MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
          "Jul", "Agt", "Sep", "Okt", "Nov", "Des"]

EXAMPLE_PAYLOAD = {
    "das_area_km2": 450.0,
    "titik_tinjau": {"lat": -0.15, "lon": 110.47, "nama": "Outlet Sungai Kapuas"},
    "stations": [
        {"id": "S1", "name": "Sta. Gunung Mas",      "lat": -0.10, "lon": 110.45},
        {"id": "S2", "name": "Sta. Sungai Raya",     "lat": -0.25, "lon": 110.55},
        {"id": "S3", "name": "Sta. Pontianak Utara", "lat": -0.05, "lon": 110.38},
    ],
    "station_rain": {
        "S1": [280, 255, 320, 290, 210, 140, 120, 130, 190, 260, 310, 295],
        "S2": [265, 240, 300, 275, 195, 130, 110, 125, 180, 245, 290, 280],
        "S3": [290, 260, 330, 300, 220, 150, 130, 140, 200, 270, 320, 305],
    },
    "climate": [
        {"month": 1,  "T": 27.2, "RH": 84, "Rs": 13.5, "uz": 1.2},
        {"month": 2,  "T": 27.5, "RH": 82, "Rs": 14.2, "uz": 1.3},
        {"month": 3,  "T": 27.8, "RH": 83, "Rs": 15.0, "uz": 1.1},
        {"month": 4,  "T": 27.9, "RH": 82, "Rs": 15.5, "uz": 1.2},
        {"month": 5,  "T": 28.1, "RH": 80, "Rs": 15.8, "uz": 1.4},
        {"month": 6,  "T": 27.8, "RH": 79, "Rs": 14.9, "uz": 1.5},
        {"month": 7,  "T": 27.5, "RH": 78, "Rs": 14.5, "uz": 1.6},
        {"month": 8,  "T": 27.7, "RH": 79, "Rs": 15.0, "uz": 1.5},
        {"month": 9,  "T": 27.9, "RH": 81, "Rs": 14.8, "uz": 1.3},
        {"month": 10, "T": 27.8, "RH": 83, "Rs": 14.0, "uz": 1.2},
        {"month": 11, "T": 27.4, "RH": 84, "Rs": 13.5, "uz": 1.1},
        {"month": 12, "T": 27.1, "RH": 85, "Rs": 13.0, "uz": 1.1},
    ],
    "das_params": {
        "sm_cap": 200,
        "i": 0.45,
        "k": 0.55,
        "gws_init": 50,
        "sm_init": 100,
    },
}


# ─────────────────────────────────────────────
# GEE HELPER FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_gee():
    """Inisialisasi GEE — dipanggil sekali, hasilnya di-cache."""
    try:
        import ee
        ee.Initialize()
        return True, None
    except Exception as e1:
        try:
            import ee
            ee.Authenticate()
            ee.Initialize()
            return True, None
        except Exception as e2:
            return False, str(e2)


def ekstrak_das_gee(lon: float, lat: float):
    """
    Ambil polygon DAS dari HydroSHEDS Level-12 berdasarkan titik outlet.
    Return: (geojson_dict | None, luas_km2 | None, error_str | None)
    """
    try:
        import ee
        titik = ee.Geometry.Point([lon, lat])
        basins = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_12")
        das = basins.filterBounds(titik)
        luas_km2 = das.geometry().area().divide(1e6).getInfo()
        geojson = das.getInfo()   # dict GeoJSON FeatureCollection
        return geojson, round(luas_km2, 2), None
    except Exception as e:
        return None, None, str(e)


# ─────────────────────────────────────────────
# THIESSEN POLYGON — Voronoi via Scipy
# ─────────────────────────────────────────────
def buat_thiessen_polygon(stations: list[dict], bbox_pad: float = 0.3):
    """
    Buat polygon Thiessen (Voronoi) dari koordinat stasiun.
    Mengembalikan list polygon sebagai list koordinat [lon, lat].
    Menggunakan scipy.spatial.Voronoi dengan mirror-point trick untuk region terbuka.
    """
    from scipy.spatial import Voronoi
    import numpy as np

    pts = np.array([[s["lon"], s["lat"]] for s in stations])
    n = len(pts)

    if n == 1:
        # Hanya 1 stasiun — kembalikan bounding box besar sebagai "polygon"
        lon, lat = pts[0]
        pad = bbox_pad
        return [[[lon-pad, lat-pad], [lon+pad, lat-pad],
                 [lon+pad, lat+pad], [lon-pad, lat+pad], [lon-pad, lat-pad]]]

    # Bounding box + padding
    min_lon = pts[:, 0].min() - bbox_pad
    max_lon = pts[:, 0].max() + bbox_pad
    min_lat = pts[:, 1].min() - bbox_pad
    max_lat = pts[:, 1].max() + bbox_pad

    # Tambah 4 titik cermin di luar bbox agar semua region tertutup
    mirror_pts = np.array([
        [min_lon - 1, (min_lat + max_lat) / 2],
        [max_lon + 1, (min_lat + max_lat) / 2],
        [(min_lon + max_lon) / 2, min_lat - 1],
        [(min_lon + max_lon) / 2, max_lat + 1],
    ])
    all_pts = np.vstack([pts, mirror_pts])

    vor = Voronoi(all_pts)

    def clip_polygon_to_bbox(poly_coords):
        """Clip polygon ke bounding box pakai Sutherland-Hodgman sederhana."""
        def inside(p, a, b):
            return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0

        def intersection(a, b, c, d):
            A1, B1 = b[1]-a[1], a[0]-b[0]
            C1 = A1*a[0] + B1*a[1]
            A2, B2 = d[1]-c[1], c[0]-d[0]
            C2 = A2*c[0] + B2*c[1]
            det = A1*B2 - A2*B1
            if abs(det) < 1e-10:
                return a
            return ((C1*B2 - C2*B1)/det, (A1*C2 - A2*C1)/det)

        clip_edges = [
            ([min_lon, min_lat], [min_lon, max_lat]),
            ([min_lon, max_lat], [max_lon, max_lat]),
            ([max_lon, max_lat], [max_lon, min_lat]),
            ([max_lon, min_lat], [min_lon, min_lat]),
        ]
        output = list(poly_coords)
        for (a, b) in clip_edges:
            if not output:
                break
            inp = output
            output = []
            for i in range(len(inp)):
                cur = inp[i]
                prev = inp[i-1]
                if inside(cur, a, b):
                    if not inside(prev, a, b):
                        output.append(intersection(prev, cur, a, b))
                    output.append(cur)
                elif inside(prev, a, b):
                    output.append(intersection(prev, cur, a, b))
        return output

    polygons = []
    for i in range(n):  # hanya stasiun asli, bukan mirror
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            # Region terbuka — ambil vertices yang ada + clip
            verts = [vor.vertices[v] for v in region if v != -1]
            if not verts:
                polygons.append([])
                continue
        else:
            verts = [vor.vertices[v] for v in region]

        clipped = clip_polygon_to_bbox(verts)
        if clipped:
            clipped.append(clipped[0])  # tutup polygon
            polygons.append([[p[0], p[1]] for p in clipped])
        else:
            polygons.append([])

    return polygons


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for _k, _v in [
    ("stations", [{"id": "S1", "name": "Stasiun 1", "lat": 0.0, "lon": 0.0}]),
    ("rain_data", {"S1": [0.0] * 12}),
    ("result", None),
    ("payload", None),
    ("tt_lat", 0.0),
    ("tt_lon", 0.0),
    ("tt_nama", "Titik Tinjau / Outlet DAS"),
    ("das_geojson", None),
    ("das_luas_gee", None),
    ("gee_error", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


def load_example():
    p = EXAMPLE_PAYLOAD
    st.session_state.stations  = [dict(s) for s in p["stations"]]
    st.session_state.rain_data = {k: list(v) for k, v in p["station_rain"].items()}
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
    st.session_state.das_geojson  = None
    st.session_state.das_luas_gee = None


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0d9e87 0%, #4f9cf9 100%);
    padding: 1.2rem 1.6rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
">
    <h1 style="color:white; margin:0; font-size:1.5rem;">
        💧 Analisis Debit Andalan
    </h1>
    <p style="color:rgba(255,255,255,0.85); margin:4px 0 0; font-size:0.9rem;">
        Metode F.J. Mock — Neraca Air Bulanan &nbsp;|&nbsp; SNI 6738:2015
        &nbsp;|&nbsp; Deliniasi DAS via Google Earth Engine
    </p>
</div>
""", unsafe_allow_html=True)

col_btn, col_info = st.columns([1, 3])
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
    st.caption("Koordinat outlet/bendung — dasar bobot Thiessen & deliniasi DAS.")

    tt_nama = st.text_input(
        "Nama Titik Tinjau",
        value=st.session_state.get("tt_nama", "Titik Tinjau / Outlet DAS"),
        key="tt_nama",
    )
    col_tt1, col_tt2 = st.columns(2)
    with col_tt1:
        tt_lat = st.number_input("Latitude",  value=float(st.session_state.get("tt_lat", 0.0)),
                                  format="%.4f", key="tt_lat")
    with col_tt2:
        tt_lon = st.number_input("Longitude", value=float(st.session_state.get("tt_lon", 0.0)),
                                  format="%.4f", key="tt_lon")

    if tt_lat == 0.0 and tt_lon == 0.0:
        st.warning("⚠️ Titik tinjau belum diisi.")
    else:
        st.success(f"📍 ({tt_lat:.4f}, {tt_lon:.4f})")

    # Tombol Deliniasi DAS
    st.markdown("**🛰️ Deliniasi DAS (Google Earth Engine)**")
    st.caption("Otomatis membuat polygon DAS dari titik outlet via HydroSHEDS.")

    if st.button("🌍 Ambil Polygon DAS dari GEE", use_container_width=True,
                  disabled=(tt_lat == 0.0 and tt_lon == 0.0)):
        with st.spinner("Menghubungi Google Earth Engine…"):
            ok, err = init_gee()
            if not ok:
                st.session_state.gee_error = f"GEE gagal: {err}"
            else:
                geojson, luas, err2 = ekstrak_das_gee(tt_lon, tt_lat)
                if err2:
                    st.session_state.gee_error = err2
                else:
                    st.session_state.das_geojson  = geojson
                    st.session_state.das_luas_gee = luas
                    st.session_state.das_area     = luas   # update otomatis ke luas DAS
                    st.session_state.gee_error    = None
                    st.success(f"✅ DAS berhasil dideliniasi! Luas: {luas:.2f} km²")

    if st.session_state.gee_error:
        st.error(st.session_state.gee_error)

    if st.session_state.das_luas_gee:
        st.info(f"📐 Luas DAS dari GEE: **{st.session_state.das_luas_gee:.2f} km²**")

        # Tombol download GeoJSON DAS
        geojson_str = json.dumps(st.session_state.das_geojson, ensure_ascii=False, indent=2)
        st.download_button(
            "⬇️ Download GeoJSON DAS",
            data=geojson_str,
            file_name="batas_das.geojson",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()

    # ── Parameter DAS & Mock ───────────────────
    das_area = st.number_input(
        "Luas DAS (km²)", min_value=1.0, max_value=100000.0,
        value=float(st.session_state.get("das_area", 450.0)),
        step=10.0, key="das_area",
        help="Terisi otomatis jika menggunakan deliniasi GEE."
    )

    st.markdown("#### 🌊 Parameter Model Mock")
    sm_cap   = st.number_input("Kapasitas Lengas Tanah (mm)", 50, 500,
                                int(st.session_state.get("sm_cap", 200)), 10, key="sm_cap")
    i_coef   = st.slider("Koef. Infiltrasi (i)", 0.0, 1.0,
                          float(st.session_state.get("i_coef", 0.45)), 0.05, key="i_coef")
    k_coef   = st.slider("Koef. Resesi Air Tanah (k)", 0.0, 1.0,
                          float(st.session_state.get("k_coef", 0.55)), 0.05, key="k_coef")
    gws_init = st.number_input("Tampungan Air Tanah Awal (mm)", 0, 500,
                                int(st.session_state.get("gws_init", 50)), 10, key="gws_init")
    sm_init  = st.number_input("Lengas Tanah Awal (mm)", 0, 500,
                                int(st.session_state.get("sm_init", 100)), 10, key="sm_init")

    st.divider()
    st.markdown("**Panduan Singkat:**")
    st.caption("""
- **i** ≈ 0.3–0.5 (lempung–pasir)
- **k** ≈ 0.5–0.8 (resesi lambat = nilai besar)
- **SM Cap** ≈ 100–300 mm tergantung jenis tanah
    """)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Stasiun & Hujan",
    "🌡️ Data Iklim",
    "▶ Jalankan Analisis",
    "📊 Hasil",
])


# ══════════════════════════════════════════════
# TAB 1: Stasiun & Hujan
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Data Stasiun Hujan & Curah Hujan Bulanan")

    col_add, col_del = st.columns(2)
    with col_add:
        if st.button("➕ Tambah Stasiun", use_container_width=True):
            n = len(st.session_state.stations) + 1
            sid = f"S{n}"
            st.session_state.stations.append({"id": sid, "name": f"Stasiun {n}", "lat": 0.0, "lon": 0.0})
            st.session_state.rain_data[sid] = [0.0] * 12
    with col_del:
        if len(st.session_state.stations) > 1:
            if st.button("➖ Hapus Stasiun Terakhir", use_container_width=True):
                removed = st.session_state.stations.pop()
                st.session_state.rain_data.pop(removed["id"], None)

    st.divider()

    for idx, sta in enumerate(st.session_state.stations):
        sid = sta["id"]
        with st.expander(f"🔵 {sta['name']} ({sid})", expanded=True):
            c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
            with c1:
                sta["name"] = st.text_input("Nama Stasiun", sta["name"], key=f"sname_{idx}")
            with c2:
                new_id = st.text_input("ID Stasiun", sid, key=f"sid_{idx}")
                if new_id != sid:
                    st.session_state.rain_data[new_id] = st.session_state.rain_data.pop(sid, [0.0]*12)
                    sta["id"] = new_id
                    sid = new_id
            with c3:
                sta["lat"] = st.number_input("Latitude",  value=float(sta["lat"]), format="%.4f", key=f"lat_{idx}")
            with c4:
                sta["lon"] = st.number_input("Longitude", value=float(sta["lon"]), format="%.4f", key=f"lon_{idx}")

            st.markdown("**Curah Hujan Bulanan (mm):**")
            rain_vals = st.session_state.rain_data.get(sid, [0.0]*12)
            cols_rain = st.columns(12)
            new_rain = []
            for m, col in enumerate(cols_rain):
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
    st.caption("Masukkan data iklim rata-rata untuk setiap bulan di DAS.")

    default_climate = st.session_state.get("climate", EXAMPLE_PAYLOAD["climate"])
    climate_df = pd.DataFrame([{
        "Bulan": MONTHS[c["month"]-1],
        "T (°C)": c["T"], "RH (%)": c["RH"],
        "Rs (MJ/m²/hr)": c["Rs"], "uz (m/s)": c["uz"],
    } for c in default_climate])

    edited_climate = st.data_editor(
        climate_df, use_container_width=True, num_rows="fixed",
        column_config={
            "Bulan": st.column_config.TextColumn("Bulan", disabled=True),
            "T (°C)":         st.column_config.NumberColumn("T (°C)",         min_value=0.0,  max_value=50.0, format="%.1f"),
            "RH (%)":         st.column_config.NumberColumn("RH (%)",          min_value=0,    max_value=100),
            "Rs (MJ/m²/hr)":  st.column_config.NumberColumn("Rs (MJ/m²/hr)",  min_value=0.0,  format="%.1f"),
            "uz (m/s)":       st.column_config.NumberColumn("uz (m/s)",        min_value=0.0,  format="%.2f"),
        },
        key="climate_editor",
    )
    st.info("**T** = Suhu rata-rata | **RH** = Kelembaban relatif | **Rs** = Radiasi matahari | **uz** = Kecepatan angin (z=2m)")


# ══════════════════════════════════════════════
# TAB 3: Jalankan Analisis
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Ringkasan Input & Jalankan Analisis")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Parameter DAS:**")
        st.table(pd.DataFrame({
            "Parameter": ["Titik Tinjau", "Koordinat", "Luas DAS", "SM Kap", "i", "k", "GWS Awal", "SM Awal"],
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
            {"ID": s["id"], "Nama": s["name"], "Lat": s["lat"], "Lon": s["lon"]}
            for s in st.session_state.stations
        ]), use_container_width=True, hide_index=True)

        if st.session_state.das_luas_gee:
            st.success(f"🛰️ Polygon DAS dari GEE siap — {st.session_state.das_luas_gee:.2f} km²")
        else:
            st.warning("🛰️ Polygon DAS belum dideliniasi dari GEE (opsional).")

    st.divider()

    if st.button("▶ Jalankan Analisis", type="primary", use_container_width=True):
        errors = []
        if not st.session_state.stations:
            errors.append("Minimal 1 stasiun hujan diperlukan.")
        for s in st.session_state.stations:
            rain = st.session_state.rain_data.get(s["id"], [])
            if len(rain) < 12 or all(v == 0 for v in rain):
                errors.append(f"Data hujan stasiun {s['name']} ({s['id']}) belum diisi.")

        if errors:
            for e in errors:
                st.error(e)
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
                            "sm_cap": st.session_state.sm_cap,
                            "i":      st.session_state.i_coef,
                            "k":      st.session_state.k_coef,
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

    # ── Debit Andalan Summary ─────────────────────
    st.subheader("📊 Debit Andalan (SNI 6738:2015)")
    c80, c90, c95 = st.columns(3)
    c80.metric("Q80 — Irigasi",   f"{q['Q80']} m³/s", delta="Probabilitas 80%", delta_color="off")
    c90.metric("Q90 — PLTA",      f"{q['Q90']} m³/s", delta="Probabilitas 90%", delta_color="off")
    c95.metric("Q95 — Air Baku",  f"{q['Q95']} m³/s", delta="Probabilitas 95%", delta_color="off")

    st.divider()

    # ══════════════════════════════════════════════
    # PETA: Polygon DAS + Polygon Thiessen + Stasiun
    # ══════════════════════════════════════════════
    st.subheader("🗺️ Peta DAS & Polygon Thiessen")

    stations_result = r["stations"]

    # Hitung polygon Thiessen
    try:
        thiessen_polys = buat_thiessen_polygon(stations_result)
    except Exception:
        thiessen_polys = []

    COLORS_THIESSEN = [
        "rgba(28,207,176,0.25)", "rgba(79,156,249,0.25)",
        "rgba(245,166,35,0.25)", "rgba(155,127,250,0.25)",
        "rgba(240,112,112,0.25)", "rgba(94,201,122,0.25)",
    ]
    BORDER_COLORS = ["#1ccfb0", "#4f9cf9", "#f5a623", "#9b7ffa", "#f07070", "#5ec97a"]

    fig_map = go.Figure()

    # 1. Polygon DAS dari GEE (jika ada)
    das_gj = st.session_state.das_geojson
    if das_gj:
        features = das_gj.get("features", [])
        for feat in features:
            geom = feat.get("geometry", {})
            gtype = geom.get("type", "")
            coords_raw = geom.get("coordinates", [])
            rings = []
            if gtype == "Polygon":
                rings = coords_raw
            elif gtype == "MultiPolygon":
                rings = [r[0] for r in coords_raw]

            for ring in rings:
                lons = [p[0] for p in ring]
                lats = [p[1] for p in ring]
                fig_map.add_trace(go.Scattermapbox(
                    lon=lons, lat=lats, mode="lines",
                    line=dict(color="#f5a623", width=2.5),
                    fill="toself", fillcolor="rgba(245,166,35,0.08)",
                    name="Batas DAS (GEE)", showlegend=True,
                    hoverinfo="name",
                ))

    # 2. Polygon Thiessen
    for i, (sta, poly) in enumerate(zip(stations_result, thiessen_polys)):
        if not poly:
            continue
        lons = [p[0] for p in poly]
        lats = [p[1] for p in poly]
        color_fill   = COLORS_THIESSEN[i % len(COLORS_THIESSEN)]
        color_border = BORDER_COLORS[i % len(BORDER_COLORS)]
        fig_map.add_trace(go.Scattermapbox(
            lon=lons, lat=lats, mode="lines",
            fill="toself", fillcolor=color_fill,
            line=dict(color=color_border, width=1.8),
            name=f"Thiessen {sta['name']} ({sta['weight']*100:.1f}%)",
            hovertemplate=(
                f"<b>{sta['name']}</b><br>"
                f"Bobot: {sta['weight']*100:.1f}%<br>"
                f"Lat: {sta['lat']}<br>Lon: {sta['lon']}<extra></extra>"
            ),
        ))

    # 3. Titik stasiun hujan
    sta_lons  = [s["lon"]  for s in stations_result]
    sta_lats  = [s["lat"]  for s in stations_result]
    sta_names = [f"{s['name']}<br>Bobot: {s['weight']*100:.1f}%" for s in stations_result]
    fig_map.add_trace(go.Scattermapbox(
        lon=sta_lons, lat=sta_lats,
        mode="markers+text",
        marker=dict(size=12, color="#4f9cf9", symbol="circle"),
        text=[s["name"] for s in stations_result],
        textposition="top right",
        name="Stasiun Hujan",
        hovertext=sta_names,
        hoverinfo="text",
    ))

    # 4. Titik tinjau (outlet)
    tt_info = st.session_state.payload.get("titik_tinjau", {}) if st.session_state.payload else {}
    tt_lon_v = tt_info.get("lon", 0.0)
    tt_lat_v = tt_info.get("lat", 0.0)
    tt_nm_v  = tt_info.get("nama", "Titik Tinjau")
    if tt_lon_v != 0.0 or tt_lat_v != 0.0:
        fig_map.add_trace(go.Scattermapbox(
            lon=[tt_lon_v], lat=[tt_lat_v],
            mode="markers+text",
            marker=dict(size=16, color="#f07070", symbol="circle"),
            text=[tt_nm_v],
            textposition="top right",
            name="Titik Tinjau (Outlet)",
            hovertemplate=f"<b>{tt_nm_v}</b><br>Lat: {tt_lat_v}<br>Lon: {tt_lon_v}<extra></extra>",
        ))

    # Center peta
    all_lons = sta_lons + ([tt_lon_v] if tt_lon_v != 0 else [])
    all_lats = sta_lats + ([tt_lat_v] if tt_lat_v != 0 else [])
    center_lon = sum(all_lons)/len(all_lons) if all_lons else 110.0
    center_lat = sum(all_lats)/len(all_lats) if all_lats else -0.1

    fig_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=9,
        ),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor="rgba(26,29,39,0.85)",
            font=dict(color="#c8cad5", size=11),
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
        ),
        paper_bgcolor="#1a1d27",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Tabel bobot Thiessen
    with st.expander("📋 Detail Bobot Thiessen", expanded=False):
        tt_val = st.session_state.payload.get("titik_tinjau", {}) if st.session_state.payload else {}
        tlat, tlon = tt_val.get("lat",0.0), tt_val.get("lon",0.0)
        if tlat != 0.0 or tlon != 0.0:
            st.info(f"📍 **Referensi IDW:** {tt_val.get('nama','-')} — Lat `{tlat}`, Lon `{tlon}`")
        else:
            st.warning("⚠️ Bobot dihitung dari centroid otomatis (titik tinjau belum diisi).")

        st.dataframe(pd.DataFrame([
            {"ID": s["id"], "Nama": s["name"],
             "Lat": s["lat"], "Lon": s["lon"],
             "Bobot (%)": round(s["weight"]*100, 2)}
            for s in stations_result
        ]), use_container_width=True, hide_index=True)

    # Download GeoJSON Thiessen
    if thiessen_polys:
        thiessen_gj = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "id": s["id"],
                        "name": s["name"],
                        "weight": s["weight"],
                        "weight_pct": round(s["weight"]*100, 2),
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [poly] if poly else [[]],
                    },
                }
                for s, poly in zip(stations_result, thiessen_polys)
                if poly
            ],
        }
        st.download_button(
            "⬇️ Download GeoJSON Polygon Thiessen",
            data=json.dumps(thiessen_gj, ensure_ascii=False, indent=2),
            file_name="polygon_thiessen.geojson",
            mime="application/json",
        )

    st.divider()

    # ── Debit Bulanan ──────────────────────────
    st.subheader("📈 Debit Bulanan (m³/s)")
    fig_q = go.Figure(go.Bar(
        x=MONTHS, y=mock["debit_m3s"],
        marker_color="rgba(28,207,176,0.7)",
        marker_line_color="#1ccfb0", marker_line_width=1.2,
        name="Debit (m³/s)",
    ))
    fig_q.update_layout(
        height=320, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="m³/s",
        plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27", font_color="#c8cad5",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig_q, use_container_width=True)

    # ── Hujan & ETP ───────────────────────────
    st.subheader("🌧️ Hujan Kawasan & ETP Bulanan")
    fig_re = go.Figure()
    fig_re.add_trace(go.Bar(
        x=MONTHS, y=r["areal_rain"], name="Hujan Kawasan (mm)",
        marker_color="rgba(79,156,249,0.6)", marker_line_color="#4f9cf9", marker_line_width=1,
    ))
    fig_re.add_trace(go.Scatter(
        x=MONTHS, y=r["etp_monthly"], name="ETP (mm)", mode="lines+markers",
        line=dict(color="#f5a623", width=2), marker=dict(size=6),
    ))
    fig_re.update_layout(
        height=320, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="mm",
        plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27", font_color="#c8cad5",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_re, use_container_width=True)

    # ── FDC ───────────────────────────────────
    st.subheader("📉 Kurva Durasi Aliran (FDC)")
    fdc = r["fdc"]
    fig_fdc = go.Figure(go.Scatter(
        x=fdc["exceedance"], y=fdc["debit"], mode="lines", name="FDC",
        line=dict(color="#9b7ffa", width=2),
        fill="tozeroy", fillcolor="rgba(155,127,250,0.12)",
    ))
    for pct, label, color in [(80,"Q80","#1ccfb0"), (90,"Q90","#4f9cf9"), (95,"Q95","#f5a623")]:
        fig_fdc.add_vline(x=pct, line_dash="dash", line_color=color, line_width=1.2,
                          annotation_text=f"{label}={q[f'Q{pct}']} m³/s",
                          annotation_font_color=color)
    fig_fdc.update_layout(
        height=340, margin=dict(l=10,r=10,t=10,b=10),
        xaxis_title="% Waktu terlampaui", yaxis_title="m³/s",
        plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27", font_color="#c8cad5",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig_fdc, use_container_width=True)

    # ── LP3 ───────────────────────────────────
    st.subheader("🔄 Debit Kala Ulang — Log Pearson III")
    st.caption(f"mean={lp3_p['mean']} | σ={lp3_p['std']} | Cs={lp3_p['skew']} | n={lp3_p['n']}")
    for col, (key, val) in zip(st.columns(len(lp3_k)), lp3_k.items()):
        col.metric(key, f"{val} m³/s")

    fig_lp3 = go.Figure(go.Bar(
        x=list(lp3_k.keys()), y=list(lp3_k.values()),
        marker_color="rgba(155,127,250,0.7)",
        marker_line_color="#9b7ffa", marker_line_width=1,
    ))
    fig_lp3.update_layout(
        height=260, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="m³/s",
        plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27", font_color="#c8cad5",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig_lp3, use_container_width=True)

    st.divider()

    # ── Tabel Mock ────────────────────────────
    st.subheader("📋 Tabel Neraca Air Bulanan (Mock)")
    cols_show  = ["month_name","rain","etp","aet","water_surplus",
                  "direct_runoff","infiltration","baseflow","total_flow_mm","debit_m3s"]
    labels_show = ["Bulan","Hujan (mm)","ETP (mm)","AET (mm)","Surplus (mm)",
                   "Run-off (mm)","Infiltrasi (mm)","Base-flow (mm)","Total (mm)","Debit (m³/s)"]
    mock_df = pd.DataFrame({lbl: mock[col] for col, lbl in zip(cols_show, labels_show)})
    st.dataframe(mock_df, use_container_width=True, hide_index=True)

    # ── Export ────────────────────────────────
    st.divider()
    st.subheader("💾 Ekspor Hasil")
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        csv_buf = io.StringIO()
        mock_df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download Tabel (CSV)", data=csv_buf.getvalue(),
                           file_name="neraca_air_mock.csv", mime="text/csv",
                           use_container_width=True)
    with col_ex2:
        st.download_button("⬇️ Download Hasil Lengkap (JSON)",
                           data=json.dumps(r, indent=2, ensure_ascii=False),
                           file_name="hasil_debit_andalan.json", mime="application/json",
                           use_container_width=True)
