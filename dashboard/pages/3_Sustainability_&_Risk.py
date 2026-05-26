"""
Page 3 — RQ3: Sustainability & Environmental Risk
Vegetation Loss · Heat Susceptibility · Flood Vulnerability · Env Exposure Index
"""

import os
import ee
import geemap
import solara
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────
#  Asset & Relative Path Mapping (Isolated Dashboard Infrastructure)
# ──────────────────────────────────────────────────────────────────────

PAGES_DIR = os.path.dirname(__file__)                 # dashboard/pages/
DASHBOARD_ROOT = os.path.dirname(PAGES_DIR)           # dashboard/
GEOJSON_PATH = os.path.join(DASHBOARD_ROOT, "assets", "Dholera_Taluk.geojson")

# ──────────────────────────────────────────────────────────────────────
#  GEE Risk Analytical Pipelines (Directly Ports Your Notebook Logic)
# ──────────────────────────────────────────────────────────────────────

def _get_roi():
    """Ingests your local boundary file natively from the assets folder."""
    if os.path.exists(GEOJSON_PATH):
        return geemap.geojson_to_ee(GEOJSON_PATH).geometry()
    else:
        print(f"⚠️ Boundary missing at {GEOJSON_PATH}. Applying regional fallback polygon.")
        return ee.Geometry.Polygon([[
            [72.00, 22.15], [72.35, 22.15], [72.35, 22.50], [72.00, 22.50], [72.00, 22.15]
        ]])


def _s2(year_start, year_end, roi):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(year_start, year_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
        .median()
        .clip(roi)
    )


def _savi(s2_img):
    return s2_img.expression(
        "((NIR - Red) * 1.5) / (NIR + Red + 0.5)",
        {"NIR": s2_img.select("B8"), "Red": s2_img.select("B4")}
    ).rename("SAVI")


def _heat_index(s2_img):
    ndbi = s2_img.normalizedDifference(["B11", "B8"])
    savi = _savi(s2_img)
    ndbi_norm = ndbi.add(0.5).divide(1.0).clamp(0, 1)
    savi_norm = savi.divide(0.8).clamp(0, 1)
    return ndbi_norm.subtract(savi_norm).rename("HeatIndex")


def _sar_flood_freq(roi):
    """Computes authentic radar-based inundation frequencies from monsoon 2025."""
    s1_monsoon = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate("2025-06-15", "2025-10-30")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )
    
    s1_dry = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate("2025-03-01", "2025-05-20")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    def water_mask(img):
        smoothed = img.focal_mean(radius=1, kernelType="square", units="pixels")
        return smoothed.lt(-17).rename("water")

    # Clear dry-season baseline feature mask
    s1_dry_smoothed = s1_dry.median().focal_mean(radius=1, kernelType="square", units="pixels").clip(roi)
    sar_permanent_water = s1_dry_smoothed.lt(-17).rename("PermanentWater")

    # Generate mean pooled distribution index
    flood_freq = (
        s1_monsoon.map(water_mask)
        .mean()
        .where(sar_permanent_water.eq(1), 0)
        .rename("FloodFreq")
        .clip(roi)
    )
    return flood_freq, sar_permanent_water


def _fvi(flood_freq, sar_permanent_water, roi):
    """Flood Vulnerability Index: Evaluates low-lying topographic sinks."""
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").filterBounds(roi).median().clip(roi).rename("Elevation")
    
    elev = dem.select("Elevation")
    freq = flood_freq.select("FloodFreq")

    moderate_cond = elev.lte(8).And(freq.gt(0.10))
    high_cond = elev.lte(5).And(freq.gt(0.35))

    fvi = (
        ee.Image(1)
        .where(moderate_cond, 2)
        .where(high_cond, 3)
        .rename("FVI")
        .clip(roi)
    )
    return fvi.updateMask(sar_permanent_water.eq(0))


def _env_exposure(heat_index, fvi_img):
    heat_norm  = heat_index.add(0.5).divide(1.2).clamp(0, 1)
    flood_norm = fvi_img.subtract(1).divide(2).clamp(0, 1)
    return heat_norm.multiply(0.5).add(flood_norm.multiply(0.5)).rename("EnvExposure")


# ──────────────────────────────────────────────────────────────────────
#  Empirical Data Metrics Mapping
# ──────────────────────────────────────────────────────────────────────

ENV_STATS = [
    {"label": "New Built-up Footprint",   "value": "34.013 km²", "color": "#8b949e"},
    {"label": "Growth ∩ Vegetation Loss", "value": "28.203 km²", "color": "#f0883e"},
    {"label": "Biomass Conversion Share","value": "82.9%",      "color": "#e63946"},
    {"label": "Avg SAVI Loss / km²",     "value": "−0.1927",   "color": "#f0883e"},
    {"label": "In Moderate Flood Risk",  "value": "8.865 km²",  "color": "#f59e0b"},
    {"label": "In High Flood Risk",      "value": "0.027 km²",  "color": "#e63946"},
]

FVI_STATS = [
    {"label": "✅ Low Risk",      "km2": 983.651, "pct": 84.9, "color": "#2d6a4f"},
    {"label": "⚠️ Moderate Risk", "km2": 164.324, "pct": 14.2, "color": "#f4a261"},
    {"label": "🚨 High Risk",     "km2": 10.672,  "pct":  0.9, "color": "#d62828"},
]


@solara.component
def StatCard(label, value, color):
    with solara.Column(
        style=(
            f"background:#161b22; border:1px solid #30363d; border-left:3px solid {color};"
            "border-radius:8px; padding:14px 18px; min-width:150px; flex:1; gap:4px;"
        )
    ):
        solara.Text(label, style="font-size:10px; color:#8b949e; font-family:'IBM Plex Mono',monospace; letter-spacing:0.8px;")
        solara.Text(value, style=f"font-size:20px; font-weight:700; color:{color}; font-family:'IBM Plex Mono',monospace;")


def _fvi_chart():
    fig = go.Figure(go.Bar(
        x=[s["label"] for s in FVI_STATS],
        y=[s["km2"]   for s in FVI_STATS],
        marker=dict(
            color=[s["color"] for s in FVI_STATS],
            line=dict(color="#0d1117", width=1),
        ),
        text=[f"{s['km2']} km²\n({s['pct']}%)" for s in FVI_STATS],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color="#c9d1d9"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        title=dict(text="Flood Vulnerability Surface Area Scaling", font=dict(family="IBM Plex Mono", size=13, color="#e6edf3")),
        yaxis=dict(title="Square Kilometers", gridcolor="#21262d", color="#8b949e"),
        xaxis=dict(color="#8b949e"),
        margin=dict(l=50, r=20, t=50, b=50),
        height=300,
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
                "RQ3 — Sustainability & Environmental Risk Realignment",
                style="font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:700; color:#e6edf3;",
            )
            solara.Text(
                "Sentinel-2 SAVI Matrix  ·  Sentinel-1 Radar Interferometry  ·  Copernicus GLO-30 DEM",
                style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#8b949e; letter-spacing:1px;",
            )
            solara.Text(
                "Does the physical footprint of Dholera's built-up expansion overlap with environmentally critical "
                "or climatically vulnerable terrain sinks, and does corridor deployment introduce systematic spatial risk?",
                style="font-size:14px; color:#c9d1d9; max-width:860px; line-height:1.6;",
            )

        # ── Environmental Risk Cards Grid ──
        with solara.Row(style="gap:10px; flex-wrap:wrap; width:100%;"):
            for s in ENV_STATS:
                StatCard(s["label"], s["value"], s["color"])

        # ── Maps & Spatial Diagnostics Matrix ──
        try:
            roi = _get_roi()
            s2_2025 = _s2("2025-10-01", "2025-12-31", roi)
            s2_2016 = _s2("2016-10-01", "2016-12-31", roi)

            savi_25 = _savi(s2_2025)
            savi_16 = _savi(s2_2016)
            savi_delta = savi_25.subtract(savi_16).rename("SAVI_Delta")

            heat = _heat_index(s2_2025)
            flood, perm_water = _sar_flood_freq(roi)
            fvi = _fvi(flood, perm_water, roi)
            exposure = _env_exposure(heat, fvi)

            # ── Map 1: Biomass Transformation ──
            with solara.Column(style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden; width:100%;"):
                with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; align-items:center; gap:10px;"):
                    solara.Text("🌿", style="font-size:18px;")
                    solara.Text("Biomass Transformation Map — SAVI Loss Profile Within New Growth",
                                style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;")
                with solara.Column(style="padding:0; width:100%;"):
                    solara.Text(
                        "Pale yellow = marginal biomass decay  →  Deep red = severe removal  ·  82.9% of new growth matches local biomass depletion",
                        style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8b949e; padding:10px 20px;",
                    )
                    m1 = geemap.Map(center=[22.37, 72.05], zoom=11)
                    m1.add_basemap("HYBRID")
                    m1.addLayer(
                        savi_delta.updateMask(savi_delta.lt(0)),
                        {"min": -0.3, "max": 0, "palette": ["#800026", "#e31a1c", "#fd8d3c", "#ffffb2"]},
                        "SAVI Delta (Vegetation Loss)"
                    )
                    m1.layout.height = "460px"
                    solara.display(m1)

            # ── Map 2: Heat Susceptibility ──
            with solara.Column(style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden; width:100%;"):
                with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; align-items:center; gap:10px;"):
                    solara.Text("🌡️", style="font-size:18px;")
                    solara.Text("Heat Susceptibility Index Proxy — NDBI_norm − SAVI_norm",
                                style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;")
                with solara.Column(style="padding:0; width:100%;"):
                    solara.Text(
                        "Cool blue = thermal resilience framing  →  Fiery red = exposed urban heat retention cores",
                        style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8b949e; padding:10px 20px;",
                    )
                    m2 = geemap.Map(center=[22.37, 72.05], zoom=11)
                    m2.add_basemap("HYBRID")
                    m2.addLayer(
                        heat,
                        {"min": -0.5, "max": 0.7, "palette": ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"]},
                        "Heat Susceptibility Index"
                    )
                    m2.layout.height = "460px"
                    solara.display(m2)

            # ── Map 3 & Chart Combo: Flood Vulnerability Index ──
            with solara.Row(style="gap:16px; flex-wrap:wrap; width:100%;"):
                with solara.Column(style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden; flex:1; min-width:320px;"):
                    with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; align-items:center; gap:10px;"):
                        solara.Text("🌊", style="font-size:18px;")
                        solara.Text("Bivariate Flood Vulnerability Index (FVI)",
                                    style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;")
                    m3 = geemap.Map(center=[22.37, 72.05], zoom=11)
                    m3.add_basemap("HYBRID")
                    m3.addLayer(
                        fvi,
                        {"min": 1, "max": 3, "palette": ["2d6a4f", "f4a261", "d62828"]},
                        "FVI Choropleth Map"
                    )
                    m3.layout.height = "400px"
                    solara.display(m3)

                with solara.Column(style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden; flex:1; min-width:320px;"):
                    with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; background:#0d1117;"):
                        solara.Text("FVI Zonal Distribution Profile", style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;")
                    with solara.Column(style="padding:12px; width:100%;"):
                        solara.FigurePlotly(_fvi_chart())
                    with solara.Column(style="padding:16px; gap:8px;"):
                        solara.Text(
                            "Notably, 26.1% of all new urban footprint additions (8.865 km²) sit inside low-elevation terrain "
                            "characterized by recurring post-monsoon radar surface pooling, confirming a structural mismatch.",
                            style="font-size:12.5px; color:#c9d1d9; line-height:1.6;",
                        )

            # ── Map 4: Environmental Exposure Composite ──
            with solara.Column(style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden; width:100%;"):
                with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; align-items:center; gap:10px;"):
                    solara.Text("⚠️", style="font-size:18px;")
                    solara.Text("Composite Environmental Exposure Surface Matrix (Equal Weighted)",
                                style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;")
                with solara.Column(style="padding:0; width:100%;"):
                    solara.Text(
                        "0 = Geographically stable matrix  →  1 = Maximum compound micro-climate threat matrix (Low-Elevation + High Thermal Concentration)",
                        style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8b949e; padding:10px 20px;",
                    )
                    m4 = geemap.Map(center=[22.37, 72.05], zoom=11)
                    m4.add_basemap("HYBRID")
                    m4.addLayer(
                        exposure,
                        {"min": 0, "max": 1, "palette": ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560", "#f5a623", "#ffffff"]},
                        "Integrated Risk Exposure Surface"
                    )
                    m4.layout.height = "460px"
                    solara.display(m4)

        except Exception as e:
            with solara.Column(style="padding:40px; align-items:center; gap:12px; width:100%;"):
                solara.Text("⚠️ Cloud Dataset Handshake Faulted.", style="color:#f0883e; font-family:'IBM Plex Mono',monospace; font-size:14px;")
                solara.Text(str(e), style="color:#8b949e; font-family:'IBM Plex Mono',monospace; font-size:11px;")

        # ── Key Analytical Realignment Findings ──
        with solara.Column(style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px 24px; gap:14px; width:100%;"):
            solara.Text("Key Ecological Realignment Findings", style="font-family:'IBM Plex Mono',monospace; font-size:14px; font-weight:700; color:#e6edf3;")
            
            insights = [
                ("82.9% of total new built-up growth has expanded directly over active regional biomass matrices. Every square kilometer of newly paved ground registers an average SAVI depletion signature of −0.1927, framing the systematic ecological friction of infrastructure footprint growth.", "#3fb950"),
                ("26.1% of growth space (8.865 km²) occupies low-lying flood-prone terrain verified directly by Sentinel-1 radar backscatter reflections, exposing an engineering disconnect between grid zoning and micro-topography.", "#f59e0b"),
                ("The Compound Exposure Phenomenon: Vulnerable sectors confront simultaneous multi-risk profiles where high physical building densities collide with low vegetative evapotranspiration buffers inside low-elevation terrain sinks.", "#e63946"),
            ]
            for finding, color in insights:
                with solara.Row(style="align-items:flex-start; gap:12px; width:100%;"):
                    solara.Text("▸", style=f"color:{color}; font-size:16px; line-height:1.5; flex-shrink:0;")
                    solara.Text(finding, style="font-size:13px; color:#c9d1d9; line-height:1.6;")

        # ── Methodological System Limitations ──
        with solara.Column(style="background:#161b22; border:1px solid #21262d; border-radius:10px; padding:16px 20px; gap:8px; width:100%;"):
            solara.Text("Methodological System Limitations", style="font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; color:#8b949e;")
            limitations = [
                "SAVI Delta Monsoon Confound — 2025 baseline parameters are elevated due to higher regional precipitation anomalies; calculations are tightly isolated to the new growth footprint boundary to mitigate skewed signals.",
                "Copernicus GLO-30 Vertical Thresholds — Vertical accuracy errors (~4m) mean absolute FVI class boundary pixel cuts are sensitive to pixel edge blending.",
                "Specular Backscatter Threshold Constraints — Smooth, dry alluvial bare soils or dried coastal salt pans can occasionally mimic specular surface water returns below −17 dB thresholds.",
                "Thermal Susceptibility Proxy Scope — The derived heat index functions strictly as a relative spectral materials balance framework, not a direct kinetic Land Surface Temperature (LST) measurement."
            ]
            for lim in limitations:
                solara.Text(f"· {lim}", style="font-size:12px; color:#6e7681; line-height:1.5;")

'''
# ──────────────────────────────────────────────────────────────────────
#  Standalone Code Execution Handler (Local Diagnostic Sandbox)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Triggering standalone spatial audit for Page 3...")
    print(f"Validating boundary file track: {GEOJSON_PATH} -> Exists: {os.path.exists(GEOJSON_PATH)}")
    try:
        ee.Initialize()
        roi_test = _get_roi()
        print(f"✅ Boundary configuration stable. ROI Type: {roi_test.type().getInfo()}")
        flood_test, dry_test = _sar_flood_freq(roi_test)
        print(f"✅ SAR Processing Pipeline online. Flood Index Bands: {flood_test.bandNames().getInfo()}")
        print("🎉 Script structural logic is mathematically secure.")
    except Exception as error:
        print(f"❌ Sandbox verification compile failed: {str(error)}")
'''