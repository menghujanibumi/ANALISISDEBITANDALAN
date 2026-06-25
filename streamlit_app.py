"""
Analisis Debit Andalan — SNI 6738:2015
Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io

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
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "stations" not in st.session_state:
    st.session_state.stations = [{"id": "S1", "name": "Stasiun 1", "lat": 0.0, "lon": 0.0}]
if "rain_data" not in st.session_state:
    st.session_state.rain_data = {"S1": [0.0] * 12}
if "result" not in st.session_state:
    st.session_state.result = None
if "payload" not in st.session_state:
    st.session_state.payload = None


def load_example():
    p = EXAMPLE_PAYLOAD
    st.session_state.stations = [dict(s) for s in p["stations"]]
    st.session_state.rain_data = {k: list(v) for k, v in p["station_rain"].items()}
    st.session_state.das_area = p["das_area_km2"]
    st.session_state.sm_cap   = p["das_params"]["sm_cap"]
    st.session_state.i_coef   = p["das_params"]["i"]
    st.session_state.k_coef   = p["das_params"]["k"]
    st.session_state.gws_init = p["das_params"]["gws_init"]
    st.session_state.sm_init  = p["das_params"]["sm_init"]
    st.session_state.climate  = p["climate"]


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
    </p>
</div>
""", unsafe_allow_html=True)

# Tombol load contoh
col_btn, col_info = st.columns([1, 3])
with col_btn:
    if st.button("📥 Muat Contoh Data", use_container_width=True):
        load_example()
        st.success("Contoh data berhasil dimuat!")

with col_info:
    st.info("Isi form di bawah lalu tekan **▶ Jalankan Analisis**. Atau gunakan contoh data Pontianak (Kalimantan Barat).")

st.divider()

# ─────────────────────────────────────────────
# SIDEBAR — Parameter DAS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameter DAS")

    das_area = st.number_input(
        "Luas DAS (km²)", min_value=1.0, max_value=100000.0,
        value=float(st.session_state.get("das_area", 450.0)),
        step=10.0, key="das_area"
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
- **i** ≈ 0.3–0.5 (tanah lempung–pasir)
- **k** ≈ 0.5–0.8 (resesi lambat = nilai besar)
- **SM Cap** ≈ 100–300 mm tergantung jenis tanah
    """)

# ─────────────────────────────────────────────
# TAB UTAMA
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📍 Stasiun & Hujan", "🌡️ Data Iklim", "▶ Jalankan Analisis", "📊 Hasil"])

# ══════════════════════════════════════════════
# TAB 1: Stasiun & Hujan
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Data Stasiun Hujan & Curah Hujan Bulanan")

    col_add, col_del = st.columns([1, 1])
    with col_add:
        if st.button("➕ Tambah Stasiun", use_container_width=True):
            n = len(st.session_state.stations) + 1
            sid = f"S{n}"
            st.session_state.stations.append(
                {"id": sid, "name": f"Stasiun {n}", "lat": 0.0, "lon": 0.0}
            )
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
                sta["name"] = st.text_input("Nama Stasiun", sta["name"],
                                             key=f"sname_{idx}")
            with c2:
                sta["id"] = st.text_input("ID Stasiun", sid, key=f"sid_{idx}")
                if sta["id"] != sid:
                    st.session_state.rain_data[sta["id"]] = st.session_state.rain_data.pop(sid, [0.0]*12)
                    sid = sta["id"]
                    st.session_state.stations[idx]["id"] = sid
            with c3:
                sta["lat"] = st.number_input("Latitude", value=float(sta["lat"]),
                                              format="%.4f", key=f"lat_{idx}")
            with c4:
                sta["lon"] = st.number_input("Longitude", value=float(sta["lon"]),
                                              format="%.4f", key=f"lon_{idx}")

            st.markdown("**Curah Hujan Bulanan (mm):**")
            rain_vals = st.session_state.rain_data.get(sid, [0.0] * 12)

            cols_rain = st.columns(12)
            new_rain = []
            for m, col in enumerate(cols_rain):
                with col:
                    v = col.number_input(
                        MONTHS[m], min_value=0.0, max_value=2000.0,
                        value=float(rain_vals[m]) if m < len(rain_vals) else 0.0,
                        step=5.0, key=f"rain_{sid}_{m}", label_visibility="visible"
                    )
                    new_rain.append(v)
            st.session_state.rain_data[sid] = new_rain


# ══════════════════════════════════════════════
# TAB 2: Data Iklim
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Data Iklim Bulanan")
    st.caption("Masukkan data iklim rata-rata untuk setiap bulan di DAS.")

    default_climate = st.session_state.get("climate", EXAMPLE_PAYLOAD["climate"])

    climate_df = pd.DataFrame([
        {
            "Bulan": MONTHS[c["month"] - 1],
            "T (°C)": c["T"],
            "RH (%)": c["RH"],
            "Rs (MJ/m²/hr)": c["Rs"],
            "uz (m/s)": c["uz"],
        }
        for c in default_climate
    ])

    edited_climate = st.data_editor(
        climate_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Bulan": st.column_config.TextColumn("Bulan", disabled=True),
            "T (°C)": st.column_config.NumberColumn("T (°C)", min_value=0.0, max_value=50.0, format="%.1f"),
            "RH (%)": st.column_config.NumberColumn("RH (%)", min_value=0, max_value=100),
            "Rs (MJ/m²/hr)": st.column_config.NumberColumn("Rs (MJ/m²/hr)", min_value=0.0, format="%.1f"),
            "uz (m/s)": st.column_config.NumberColumn("uz (m/s)", min_value=0.0, format="%.2f"),
        },
        key="climate_editor"
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
            "Parameter": ["Luas DAS", "SM Kapasitas", "Koef. Infiltrasi (i)",
                          "Koef. Resesi (k)", "GWS Awal", "SM Awal"],
            "Nilai": [f"{st.session_state.das_area} km²",
                      f"{st.session_state.sm_cap} mm",
                      st.session_state.i_coef,
                      st.session_state.k_coef,
                      f"{st.session_state.gws_init} mm",
                      f"{st.session_state.sm_init} mm"],
        }))

    with col_s2:
        st.markdown("**Stasiun Hujan:**")
        sta_df = pd.DataFrame([
            {"ID": s["id"], "Nama": s["name"], "Lat": s["lat"], "Lon": s["lon"]}
            for s in st.session_state.stations
        ])
        st.dataframe(sta_df, use_container_width=True, hide_index=True)

    st.divider()

    run_btn = st.button("▶ Jalankan Analisis", type="primary", use_container_width=True)

    if run_btn:
        # Validasi
        errors = []
        if not st.session_state.stations:
            errors.append("Minimal 1 stasiun hujan diperlukan.")
        for s in st.session_state.stations:
            sid = s["id"]
            rain = st.session_state.rain_data.get(sid, [])
            if len(rain) < 12 or all(v == 0 for v in rain):
                errors.append(f"Data hujan stasiun {s['name']} ({sid}) belum diisi.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner("⏳ Menjalankan model Mock…"):
                try:
                    # Susun climate dari editor
                    climate_list = []
                    for m_idx, row in edited_climate.iterrows():
                        climate_list.append({
                            "month": m_idx + 1,
                            "T":   float(row["T (°C)"]),
                            "RH":  float(row["RH (%)"]),
                            "Rs":  float(row["Rs (MJ/m²/hr)"]),
                            "uz":  float(row["uz (m/s)"]),
                        })

                    payload = {
                        "das_area_km2": st.session_state.das_area,
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
    else:
        r = st.session_state.result
        q = r["q_andalan"]
        mock = r["mock_table"]
        lp3_k = r["lp3_kala_ulang"]
        lp3_p = r["lp3_params"]

        # ── Debit Andalan Summary ──────────────────
        st.subheader("📊 Debit Andalan (SNI 6738:2015)")
        c80, c90, c95 = st.columns(3)
        c80.metric("Q80 — Irigasi", f"{q['Q80']} m³/s",
                   delta="Probabilitas 80%", delta_color="off")
        c90.metric("Q90 — PLTA",   f"{q['Q90']} m³/s",
                   delta="Probabilitas 90%", delta_color="off")
        c95.metric("Q95 — Air Baku", f"{q['Q95']} m³/s",
                   delta="Probabilitas 95%", delta_color="off")

        # Bobot Thiessen
        with st.expander("🗺️ Bobot Thiessen Stasiun"):
            tw_df = pd.DataFrame([
                {"ID": s["id"], "Nama": s["name"],
                 "Lat": s["lat"], "Lon": s["lon"],
                 "Bobot (%)": round(s["weight"] * 100, 2)}
                for s in r["stations"]
            ])
            st.dataframe(tw_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Debit Bulanan ──────────────────────────
        st.subheader("📈 Debit Bulanan (m³/s)")
        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(
            x=MONTHS, y=mock["debit_m3s"],
            marker_color="rgba(28,207,176,0.7)",
            marker_line_color="#1ccfb0",
            marker_line_width=1.2,
            name="Debit (m³/s)"
        ))
        fig_q.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="m³/s",
            plot_bgcolor="#1a1d27",
            paper_bgcolor="#1a1d27",
            font_color="#c8cad5",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(fig_q, use_container_width=True)

        # ── Hujan & ETP ───────────────────────────
        st.subheader("🌧️ Hujan Kawasan & ETP Bulanan")
        fig_re = make_subplots(specs=[[{"secondary_y": False}]])
        fig_re.add_trace(go.Bar(
            x=MONTHS, y=r["areal_rain"],
            name="Hujan Kawasan (mm)",
            marker_color="rgba(79,156,249,0.6)",
            marker_line_color="#4f9cf9", marker_line_width=1,
        ))
        fig_re.add_trace(go.Scatter(
            x=MONTHS, y=r["etp_monthly"],
            name="ETP (mm)", mode="lines+markers",
            line=dict(color="#f5a623", width=2),
            marker=dict(size=6),
        ))
        fig_re.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="mm",
            plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27",
            font_color="#c8cad5",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_re, use_container_width=True)

        # ── FDC ───────────────────────────────────
        st.subheader("📉 Kurva Durasi Aliran (FDC)")
        fdc = r["fdc"]
        fig_fdc = go.Figure()
        fig_fdc.add_trace(go.Scatter(
            x=fdc["exceedance"], y=fdc["debit"],
            mode="lines", name="FDC",
            line=dict(color="#9b7ffa", width=2),
            fill="tozeroy", fillcolor="rgba(155,127,250,0.12)",
        ))
        for pct, label, color in [(80, "Q80", "#1ccfb0"), (90, "Q90", "#4f9cf9"), (95, "Q95", "#f5a623")]:
            fig_fdc.add_vline(
                x=pct, line_dash="dash", line_color=color, line_width=1.2,
                annotation_text=f"{label}={q[f'Q{pct}']} m³/s",
                annotation_font_color=color,
            )
        fig_fdc.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="% Waktu terlampaui",
            yaxis_title="m³/s",
            plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27",
            font_color="#c8cad5",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(fig_fdc, use_container_width=True)

        # ── LP3 ───────────────────────────────────
        st.subheader("🔄 Debit Kala Ulang — Log Pearson III")
        st.caption(f"Parameter: mean={lp3_p['mean']}, σ={lp3_p['std']}, Cs={lp3_p['skew']}, n={lp3_p['n']}")
        lp3_cols = st.columns(len(lp3_k))
        for col, (key, val) in zip(lp3_cols, lp3_k.items()):
            col.metric(key, f"{val} m³/s")

        fig_lp3 = go.Figure(go.Bar(
            x=list(lp3_k.keys()), y=list(lp3_k.values()),
            marker_color="rgba(155,127,250,0.7)",
            marker_line_color="#9b7ffa", marker_line_width=1,
        ))
        fig_lp3.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="m³/s",
            plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27",
            font_color="#c8cad5",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(fig_lp3, use_container_width=True)

        st.divider()

        # ── Tabel Mock ────────────────────────────
        st.subheader("📋 Tabel Neraca Air Bulanan (Mock)")
        cols_show = ["month_name", "rain", "etp", "aet", "water_surplus",
                     "direct_runoff", "infiltration", "baseflow", "total_flow_mm", "debit_m3s"]
        labels_show = ["Bulan", "Hujan (mm)", "ETP (mm)", "AET (mm)", "Surplus (mm)",
                       "Run-off (mm)", "Infiltrasi (mm)", "Base-flow (mm)", "Total (mm)", "Debit (m³/s)"]
        mock_table_df = pd.DataFrame({lbl: mock[col] for col, lbl in zip(cols_show, labels_show)})
        st.dataframe(mock_table_df, use_container_width=True, hide_index=True)

        # ── Export ────────────────────────────────
        st.divider()
        st.subheader("💾 Ekspor Hasil")

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            csv_buf = io.StringIO()
            mock_table_df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download Tabel (CSV)",
                data=csv_buf.getvalue(),
                file_name="neraca_air_mock.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_ex2:
            json_str = json.dumps(r, indent=2, ensure_ascii=False)
            st.download_button(
                "⬇️ Download Hasil Lengkap (JSON)",
                data=json_str,
                file_name="hasil_debit_andalan.json",
                mime="application/json",
                use_container_width=True,
            )
