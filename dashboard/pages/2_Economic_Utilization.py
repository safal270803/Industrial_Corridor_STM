"""
Page 2 — RQ2: Economic Utilization & Cluster Typology
VIIRS Nighttime Lights · IUR · Ghost Grid Analysis
"""

import os
import ee
import geemap
import solara
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────
#  Asset & Relative Path Mapping (Synchronized with Standalone Dashboard)
# ──────────────────────────────────────────────────────────────────────

PAGES_DIR = os.path.dirname(__file__)                 # dashboard/pages/
DASHBOARD_ROOT = os.path.dirname(PAGES_DIR)           # dashboard/
GEOJSON_PATH = os.path.join(DASHBOARD_ROOT, "assets", "Dholera_Taluk.geojson")

# ──────────────────────────────────────────────────────────────────────
#  GEE Cluster Typology Engine (Direct Port of Notebook Algorithms)
# ──────────────────────────────────────────────────────────────────────

def _get_roi():
    """Ingests local GeoJSON boundary from assets folder and returns GEE Geometry."""
    if os.path.exists(GEOJSON_PATH):
        return geemap.geojson_to_ee(GEOJSON_PATH).geometry()
    else:
        print(f"⚠️ Boundary missing at {GEOJSON_PATH}. Applying regional fallback polygon.")
        return ee.Geometry.Polygon([[
            [72.00, 22.15], [72.35, 22.15], [72.35, 22.50], [72.00, 22.50], [72.00, 22.15]
        ]])


def _s2_2025(roi):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate("2025-10-01", "2025-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
        .median()
        .clip(roi)
    )


def _viirs(roi):
    return (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate("2025-10-01", "2026-03-31")
        .select("avg_rad")
        .median()
        .clip(roi)
    )


def _cluster_map(s2, viirs_roi, roi):
    # Spectral Index Formulations
    ndbi  = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")
    mndwi = s2.normalizedDifference(["B3",  "B11"]).rename("MNDWI")
    savi  = s2.expression(
        "((NIR - RED) * 1.5) / (NIR + RED + 0.5)",
        {"NIR": s2.select("B8"), "RED": s2.select("B4")}
    ).rename("SAVI")

    # Confirmed Baseline Built-up Mask (From your verified RQ1 parameters)
    built_up = ndbi.gt(0.05).And(mndwi.lt(0)).And(savi.lt(0.18))
    
    # Active Radiance Mask (Low background threshold adjusted for regional context)
    is_active_night = viirs_roi.gt(0.6)
    
    # Under-Construction Signature (Captures specific structural soil disturbance)
    is_soil_disturbed = (
        savi.gt(0.18).And(savi.lt(0.3))
        .And(ndbi.gt(0.05))
        .And(mndwi.lt(-0.4))
    )

    # 4-Tier Categorical Classification Topology
    return (
        ee.Image(0)
        .where(is_soil_disturbed,                              3) # 3 = Orange: Under-Construction
        .where(built_up.And(is_active_night.Not()),            2) # 2 = Purple: Dormant / Speculative
        .where(built_up.And(is_active_night),                  1) # 1 = Red: Active Industrial
        .clip(roi) # CRITICAL FIX: Explicitly bounds calculations to avoid global timeouts
        .rename("ClusterType")
    )


# ──────────────────────────────────────────────────────────────────────
#  Empirical Data Objects
# ──────────────────────────────────────────────────────────────────────

CLUSTER_STATS = [
    {"label": "🔴 Active Industrial",     "km2": 18.836, "pct": 35.2, "color": "#e63946"},
    {"label": "🟣 Dormant / Speculative", "km2": 16.673, "pct": 31.1, "color": "#6a0572"},
    {"label": "🟠 Under-Construction",    "km2": 18.053, "pct": 33.7, "color": "#f4a261"},
]
TOTAL_KM2 = 53.562
IUR = 35.2


@solara.component
def StatCard(label, value, sub, color):
    with solara.Column(
        style=(
            f"background:#161b22; border:1px solid #30363d; border-left:3px solid {color};"
            "border-radius:8px; padding:16px 20px; flex:1; min-width:180px; gap:4px;"
        )
    ):
        solara.Text(label, style="font-size:12px; color:#8b949e; font-family:'IBM Plex Mono',monospace;")
        solara.Text(value, style=f"font-size:22px; font-weight:700; color:{color}; font-family:'IBM Plex Mono',monospace;")
        solara.Text(sub,   style="font-size:11px; color:#8b949e; font-family:'IBM Plex Mono',monospace;")


# ──────────────────────────────────────────────────────────────────────
#  Plotly Graphics Rendering Functions
# ──────────────────────────────────────────────────────────────────────

def _iur_chart():
    labels = [s["label"] for s in CLUSTER_STATS]
    values = [s["pct"]   for s in CLUSTER_STATS]
    colors = [s["color"] for s in CLUSTER_STATS]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#0d1117", width=2)),
        textinfo="percent",
        textfont=dict(family="IBM Plex Mono", size=12, color="#e6edf3"),
        hovertemplate="%{label}<br>%{value}%<extra></extra>",
    ))

    fig.add_annotation(
        text=f"<b>{IUR}%</b><br><span style='font-size:11px'>IUR</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(family="IBM Plex Mono", size=18, color="#e6edf3"),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        title=dict(
            text="Industrial Utilisation Ratio (IUR)",
            font=dict(family="IBM Plex Mono", size=13, color="#e6edf3"),
        ),
        legend=dict(
            bgcolor="#161b22", bordercolor="#30363d",
            font=dict(family="IBM Plex Mono", color="#c9d1d9", size=11),
            orientation="v", yanchor="center", y=0.5, xanchor="left", x=1.02
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=320,
    )
    return fig


def _area_chart():
    labels = ["Active", "Dormant", "Under-Const."]
    km2    = [s["km2"]   for s in CLUSTER_STATS]
    colors = [s["color"] for s in CLUSTER_STATS]

    fig = go.Figure(go.Bar(
        x=labels, y=km2,
        marker=dict(color=colors, line=dict(color="#0d1117", width=1)),
        text=[f"{v} km²" for v in km2],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color="#c9d1d9"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        title=dict(text="Spatial Footprint Scale (km²)", font=dict(family="IBM Plex Mono", size=13, color="#e6edf3")),
        yaxis=dict(title="Square Kilometers", gridcolor="#21262d", color="#8b949e"),
        xaxis=dict(color="#8b949e"),
        margin=dict(l=50, r=20, t=60, b=50),
        height=320,
        showlegend=False,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
#  Main Page Layout Component
# ──────────────────────────────────────────────────────────────────────

@solara.component
def Page():
    with solara.Column(style="gap:24px; background:#0d1117; min-height:100vh; width:100%;"):

        # ── Header ──
        with solara.Column(style="gap:4px;"):
            solara.Text(
                "RQ2 — Economic Utilization & Cluster Typology",
                style="font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:700; color:#e6edf3;",
            )
            solara.Text(
                "VIIRS Nighttime Radiance (Oct 2025–Mar 2026)  ·  Sentinel-2 Index Fusion Matrix",
                style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#8b949e; letter-spacing:1px;",
            )
            solara.Text(
                "What proportion of Dholera's built-up footprint is actively generating economic activity, "
                "and how is structural utilization distributed across the corridor grid?",
                style="font-size:14px; color:#c9d1d9; max-width:860px; line-height:1.6;",
            )

        # ── IUR Framework Metrics Matrix Row ──
        with solara.Row(style="gap:16px; flex-wrap:wrap; align-items:stretch; width:100%;"):
            with solara.Column(
                style=(
                    "background:linear-gradient(135deg,#161b22 0%,#1c2128 100%);"
                    "border:1px solid #e63946; border-radius:10px; padding:24px 32px;"
                    "align-items:center; justify-content:center; min-width:200px; gap:4px;"
                )
            ):
                solara.Text("Industrial Utilisation Ratio",
                            style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8b949e; letter-spacing:1.5px;")
                solara.Text(f"{IUR}%",
                            style="font-family:'IBM Plex Mono',monospace; font-size:46px; font-weight:700; color:#e63946;")
                solara.Text(f"of {TOTAL_KM2} km² footprint",
                            style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8b949e;")

            with solara.Row(style="gap:12px; flex:1; min-width:300px; flex-wrap:wrap;"):
                for s in CLUSTER_STATS:
                    StatCard(s["label"], f"{s['km2']} km²", f"{s['pct']}% of development grid", s["color"])

        # ── Interactive Cluster Typology Map Panel ──
        with solara.Column(style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden; width:100%;"):
            with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; align-items:center; gap:12px;"):
                solara.Text("🌃", style="font-size:18px;")
                solara.Text(
                    "Macroeconomic Cluster Spatial Distribution Matrix",
                    style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;",
                )

            try:
                roi = _get_roi()
                s2  = _s2_2025(roi)
                viirs_roi = _viirs(roi)
                cluster = _cluster_map(s2, viirs_roi, roi)

                # Initialize local interactive map
                m = geemap.Map(center=[22.37, 72.05], zoom=11)
                m.add_basemap("HYBRID")

                # Map Layer Ingestion
                m.addLayer(viirs_roi, {"min": 0, "max": 15, "palette": ["000000", "3d0066", "ff6600", "ffff99"]}, "VIIRS Nightlights (Inferno)", False)
                m.addLayer(cluster, {"min": 0, "max": 3, "palette": ["1a1a1a", "#e63946", "#6a0572", "#f4a261"]}, "Cluster Typology Classification")
                
                m.add_legend(
                    title="Typology Class Index",
                    legend_dict={
                        "Active Industrial Core": "e63946",
                        "Dormant / Speculative Framework": "6a0572",
                        "Active Ground Construction": "f4a261"
                    }
                )
                m.layout.height = "500px"
                solara.display(m)

            except Exception as e:
                with solara.Column(style="padding:40px; align-items:center; width:100%;"):
                    solara.Text(
                        f"⚠️ GEE Processing Pipeline Disconnection: {str(e)}",
                        style="color:#f0883e; font-family:'IBM Plex Mono',monospace; font-size:12px;",
                    )

        # ── Analytics & Empirical Visualization Matrix Row ──
        with solara.Row(style="gap:16px; flex-wrap:wrap; width:100%;"):
            with solara.Column(style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden; flex:1; min-width:320px;"):
                with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; background:#0d1117;"):
                    solara.Text("Proportional IUR Footprint Composition", style="font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; color:#e6edf3;")
                with solara.Column(style="padding:12px; width:100%;"):
                    solara.FigurePlotly(_iur_chart())

            with solara.Column(style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden; flex:1; min-width:320px;"):
                with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; background:#0d1117;"):
                    solara.Text("Mass Scale Magnitude Distribution", style="font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; color:#e6edf3;")
                with solara.Column(style="padding:12px; width:100%;"):
                    solara.FigurePlotly(_area_chart())

        # ── Core Urban Typology Insights ──
        with solara.Column(style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px 24px; gap:14px; width:100%;"):
            solara.Text("Core Urban Geography Insights", style="font-family:'IBM Plex Mono',monospace; font-size:14px; font-weight:700; color:#e6edf3;")
            
            insights = [
                ("Physical infrastructure assets have dramatically outstripped localized industrial capitalization. An IUR signature of 35.2% means that nearly two-thirds of the structurally paved grid generates zero active nighttime radiant output.", "#e63946"),
                ("The 31.1% Speculative/Dormant metric highlights an intentional institutional planning vector. Subdivided transport networks are intentionally deployed ahead of active factory allocation to cement market-ready spatial value.", "#6a0572"),
                ("The 33.7% Under-Construction classification confirms a structural development pipeline that runs parallel in scale to the active footprint, showcasing a continuous spatial pivot.", "#f4a261"),
            ]
            for text, color in insights:
                with solara.Row(style="align-items:flex-start; gap:12px; width:100%;"):
                    solara.Text("▸", style=f"color:{color}; font-size:16px; line-height:1.5; flex-shrink:0;")
                    solara.Text(text, style="font-size:13px; color:#c9d1d9; line-height:1.6;")

"""
# ──────────────────────────────────────────────────────────────────────
#  Standalone Code Execution Handler (For Local Script Testing)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Triggering standalone spatial audit for Page 2...")
    print(f"Verifying asset file link: {GEOJSON_PATH} -> Exists: {os.path.exists(GEOJSON_PATH)}")
    try:
        ee.Initialize()
        roi_test = _get_roi()
        print(f"✅ Boundary compilation successful. ROI Geometry: {roi_test.type().getInfo()}")
        s2_test = _s2_2025(roi_test)
        print(f"✅ Satellite Pipeline compilation successful. Spectral bands: {s2_test.bandNames().getInfo()}")
        print("🎉 Script structural logic is mathematically secure.")
    except Exception as error:
        print(f"❌ Structural compiling check failed: {str(error)}")
        """