"""
Page 3 — RQ3: Sustainability & Environmental Risk
Vegetation Loss · Heat Susceptibility · Flood Vulnerability · Env Exposure
"""

import solara
import geemap
import ee
import plotly.graph_objects as go


# ─────────────────────────────────────────
#  GEE Layer Builders
# ─────────────────────────────────────────

ROI_ASSET = "projects/ee-YOUR_PROJECT/assets/Dholera_Taluk"   # ← replace


def _get_roi():
    return ee.FeatureCollection(ROI_ASSET).geometry()


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
    """SAR-based inundation frequency (monsoon 2025), permanent water excluded."""
    s1_monsoon = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate("2025-06-01", "2025-10-31")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )
    s1_dry = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate("2025-03-01", "2025-05-31")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    def water_mask(img):
        smoothed = img.focal_mean(radius=1, kernelType="square", units="pixels")
        return smoothed.lt(-17).rename("water")

    perm_water = s1_dry.map(water_mask).mean().gt(0.5)

    flood_freq = (
        s1_monsoon.map(water_mask)
        .mean()
        .where(perm_water, 0)
        .rename("FloodFreq")
        .clip(roi)
    )
    return flood_freq


def _fvi(flood_freq, roi):
    """Flood Vulnerability Index: 3-class bivariate (terrain + SAR)."""
    dem = ee.Image("COPERNICUS/DEM/GLO30").select("DEM").clip(roi)
    low_risk  = ee.Image(1)
    moderate  = ee.Image(2).where(
        dem.gt(8).Or(flood_freq.lt(0.10)), ee.Image(1)
    )
    high_risk = ee.Image(3).where(
        dem.gt(5).Or(flood_freq.lt(0.35)), ee.Image(1)
    )
    fvi = low_risk \
        .where(dem.lte(8).And(flood_freq.gt(0.10)), 2) \
        .where(dem.lte(5).And(flood_freq.gt(0.35)), 3) \
        .clip(roi).rename("FVI")
    return fvi


def _env_exposure(heat_index, fvi_img):
    heat_norm  = heat_index.add(0.5).divide(1.2).clamp(0, 1)
    flood_norm = fvi_img.subtract(1).divide(2).clamp(0, 1)
    return heat_norm.multiply(0.5).add(flood_norm.multiply(0.5)).rename("EnvExposure")


# ─────────────────────────────────────────
#  Stats
# ─────────────────────────────────────────

ENV_STATS = [
    {"label": "New Built-up Growth",     "value": "34.013 km²", "color": "#8b949e"},
    {"label": "Growth ∩ Veg Loss",       "value": "28.203 km²", "color": "#f0883e"},
    {"label": "Biomass Conversion Share","value": "82.9%",      "color": "#e63946"},
    {"label": "Avg SAVI Loss / km²",     "value": "−0.1927",   "color": "#f0883e"},
    {"label": "In Moderate Flood Risk",  "value": "8.865 km²",  "color": "#f59e0b"},
    {"label": "In High Flood Risk",      "value": "0.027 km²",  "color": "#e63946"},
]

FVI_STATS = [
    {"label": "✅ Low Risk",      "km2": 983.651, "pct": 84.9, "color": "#3fb950"},
    {"label": "⚠️ Moderate Risk", "km2": 164.324, "pct": 14.2, "color": "#f59e0b"},
    {"label": "🚨 High Risk",     "km2": 10.672,  "pct":  0.9, "color": "#e63946"},
]


@solara.component
def StatCard(label, value, color):
    with solara.Column(
        style=(
            f"background:#161b22; border:1px solid #30363d; border-left:3px solid {color};"
            "border-radius:8px; padding:14px 18px; min-width:150px; gap:4px;"
        )
    ):
        solara.Text(label, style="font-size:10px; color:#8b949e; font-family:'IBM Plex Mono',monospace; letter-spacing:0.8px;")
        solara.Text(value, style=f"font-size:20px; font-weight:700; color:{color}; font-family:'IBM Plex Mono',monospace;")


# ─────────────────────────────────────────
#  FVI Bar Chart
# ─────────────────────────────────────────

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
        title=dict(text="Flood Vulnerability Index (FVI) — Area by Class", font=dict(family="IBM Plex Mono", size=13, color="#e6edf3")),
        yaxis=dict(title="km²", gridcolor="#21262d", color="#8b949e"),
        xaxis=dict(color="#8b949e"),
        margin=dict(l=50, r=20, t=50, b=60),
        height=300,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────
#  Map Panel helper
# ─────────────────────────────────────────

@solara.component
def MapPanel(title, icon, map_obj):
    with solara.Column(
        style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden;"
    ):
        with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; align-items:center; gap:10px;"):
            solara.Text(icon, style="font-size:18px;")
            solara.Text(title, style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;")
        solara.display(map_obj)


# ─────────────────────────────────────────
#  Main Page Component
# ─────────────────────────────────────────

@solara.component
def Page():

    with solara.Column(style="padding:24px 32px; gap:24px; background:#0d1117; min-height:100vh;"):

        # ── Header ──────────────────────────────────────────────
        with solara.Column(style="gap:4px;"):
            solara.Text(
                "RQ3 — Sustainability & Environmental Risk",
                style="font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:700; color:#e6edf3;",
            )
            solara.Text(
                "Sentinel-2 SAVI  ·  Sentinel-1 SAR  ·  Copernicus GLO-30 DEM",
                style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#8b949e; letter-spacing:1px;",
            )
            solara.Text(
                "Does the spatial footprint of Dholera's built-up expansion overlap with environmentally sensitive "
                "or climatically vulnerable terrain, and does corridor development carry a quantifiable ecological cost?",
                style="font-size:14px; color:#c9d1d9; max-width:860px; line-height:1.6;",
            )

        # ── Stat Cards ───────────────────────────────────────────
        with solara.Row(style="gap:10px; flex-wrap:wrap;"):
            for s in ENV_STATS:
                StatCard(s["label"], s["value"], s["color"])

        # ── Maps ─────────────────────────────────────────────────
        try:
            roi      = _get_roi()
            s2_2025  = _s2("2025-10-01", "2025-12-31", roi)
            s2_2016  = _s2("2016-10-01", "2016-12-31", roi)

            savi_25  = _savi(s2_2025)
            savi_16  = _savi(s2_2016)
            savi_delta = savi_25.subtract(savi_16).rename("SAVI_Delta")

            heat     = _heat_index(s2_2025)
            flood    = _sar_flood_freq(roi)
            fvi      = _fvi(flood, roi)
            exposure = _env_exposure(heat, fvi)

            # ── Map 1: Biomass Transformation ───────────────────
            with solara.Column(
                style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden;"
            ):
                with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; align-items:center; gap:10px;"):
                    solara.Text("🌿", style="font-size:18px;")
                    solara.Text("Biomass Transformation Map — SAVI Loss Within New Growth",
                                style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;")
                with solara.Column(style="padding:0;"):
                    solara.Text(
                        "Pale yellow = marginal loss  →  Deep red = severe loss  ·  82.9% of new growth incurred vegetation loss",
                        style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8b949e; padding:10px 20px;",
                    )
                    m1 = geemap.Map(center=[22.35, 72.10], zoom=11)
                    m1.add_basemap("HYBRID")
                    m1.addLayer(
                        savi_delta.updateMask(savi_delta.lt(0)),
                        {"min": -0.3, "max": 0, "palette": ["8b0000", "d73027", "fc8d59", "fee090", "ffffbf"]},
                        "SAVI Delta (Vegetation Loss)"
                    )
                    m1.layout.height = "460px"
                    solara.display(m1)

            # ── Map 2: Heat Susceptibility ───────────────────────
            with solara.Column(
                style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden;"
            ):
                with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; align-items:center; gap:10px;"):
                    solara.Text("🌡️", style="font-size:18px;")
                    solara.Text("Heat Susceptibility Index — NDBI_norm − SAVI_norm",
                                style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;")
                with solara.Column(style="padding:0;"):
                    solara.Text(
                        "Cool blue = vegetated / resilient  →  Fiery red = built-up / heat-exposed",
                        style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8b949e; padding:10px 20px;",
                    )
                    m2 = geemap.Map(center=[22.35, 72.10], zoom=10)
                    m2.add_basemap("HYBRID")
                    m2.addLayer(
                        heat,
                        {"min": -0.5, "max": 0.7, "palette": ["313695", "74add1", "ffffbf", "f46d43", "a50026"]},
                        "Heat Susceptibility"
                    )
                    m2.layout.height = "460px"
                    solara.display(m2)

            # ── Map 3: FVI ───────────────────────────────────────
            with solara.Row(style="gap:16px; flex-wrap:wrap;"):
                with solara.Column(
                    style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden; flex:1; min-width:280px;"
                ):
                    with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; align-items:center; gap:10px;"):
                        solara.Text("🌊", style="font-size:18px;")
                        solara.Text("Flood Vulnerability Index (FVI)",
                                    style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;")
                    m3 = geemap.Map(center=[22.35, 72.10], zoom=10)
                    m3.add_basemap("HYBRID")
                    m3.addLayer(
                        fvi,
                        {"min": 1, "max": 3, "palette": ["2d6a4f", "f59e0b", "e63946"]},
                        "FVI (Low/Moderate/High)"
                    )
                    m3.layout.height = "400px"
                    solara.display(m3)

                # FVI stats chart beside the map
                with solara.Column(
                    style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden; flex:1; min-width:280px;"
                ):
                    with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d;"):
                        solara.Text("FVI Area Statistics", style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;")
                    with solara.Column(style="padding:12px;"):
                        solara.FigurePlotly(_fvi_chart())
                    with solara.Column(style="padding:8px 20px 16px; gap:8px;"):
                        solara.Text(
                            "26.1% of new growth (8.865 km²) sits in moderate flood-risk terrain.\n"
                            "This reflects the planned-corridor model of deploying infrastructure "
                            "across the full taluka regardless of micro-terrain suitability.",
                            style="font-size:12px; color:#c9d1d9; line-height:1.6;",
                        )

            # ── Map 4: Environmental Exposure ────────────────────
            with solara.Column(
                style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden;"
            ):
                with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; align-items:center; gap:10px;"):
                    solara.Text("⚠️", style="font-size:18px;")
                    solara.Text("Environmental Exposure Layer — 0.5 × Heat + 0.5 × Flood (Composite)",
                                style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;")
                with solara.Column(style="padding:0;"):
                    solara.Text(
                        "0 = No environmental exposure  →  1 = Maximum combined heat + flood risk  ·  Equal weighting treats both risks as co-equal",
                        style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8b949e; padding:10px 20px;",
                    )
                    m4 = geemap.Map(center=[22.35, 72.10], zoom=10)
                    m4.add_basemap("HYBRID")
                    m4.addLayer(
                        exposure,
                        {"min": 0, "max": 1, "palette": ["03071e", "370617", "6a040f", "9d0208", "d00000",
                                                           "dc2f02", "e85d04", "f48c06", "faa307", "ffba08"]},
                        "Environmental Exposure (0→1)"
                    )
                    m4.layout.height = "460px"
                    solara.display(m4)

        except Exception as e:
            with solara.Column(style="padding:40px; align-items:center; gap:12px;"):
                solara.Text(
                    "⚠️  Maps unavailable — GEE connection required.",
                    style="color:#f0883e; font-family:'IBM Plex Mono',monospace; font-size:14px;",
                )
                solara.Text(str(e), style="color:#8b949e; font-family:'IBM Plex Mono',monospace; font-size:11px;")

        # ── Key Findings ─────────────────────────────────────────
        with solara.Column(
            style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px 24px; gap:14px;"
        ):
            solara.Text("Key Findings", style="font-family:'IBM Plex Mono',monospace; font-size:14px; font-weight:700; color:#e6edf3;")
            for finding, color in [
                ("82.9% of new built-up growth came at the direct cost of local biomass. Each km² of new industrial footprint carries an average SAVI decline of −0.1927 — the ecological price tag of corridor expansion.", "#3fb950"),
                ("26.1% of new growth (8.865 km²) occupies moderate flood-risk terrain confirmed by SAR inundation evidence — terrain that is low-elevation and repeatedly inundated during monsoon.", "#f59e0b"),
                ("The Dual Exposure Problem: some zones face compound environmental risk — high built-up density with minimal vegetation buffer, situated on low-lying, repeatedly inundated terrain. This is a direct consequence of infrastructure-first development logic.", "#e63946"),
            ]:
                with solara.Row(style="align-items:flex-start; gap:12px;"):
                    solara.Text("▸", style=f"color:{color}; font-size:16px; line-height:1.5; flex-shrink:0;")
                    solara.Text(finding, style="font-size:13px; color:#c9d1d9; line-height:1.6;")

        # ── Limitations ──────────────────────────────────────────
        with solara.Column(
            style="background:#161b22; border:1px solid #21262d; border-radius:10px; padding:16px 20px; gap:8px;"
        ):
            solara.Text("Limitations", style="font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; color:#8b949e;")
            for lim in [
                "SAVI delta monsoon confound — 2025 SAVI elevated due to stronger monsoon; analysis scoped to new growth footprint to mitigate.",
                "GLO-30 vertical accuracy ~4 m; FVI thresholds at 5 m and 8 m introduce classification uncertainty at boundary pixels.",
                "SAR −17 dB threshold uniform; smooth bare soil and salt pans can produce false water signatures after permanent water exclusion.",
                "Heat index is a spectral proxy — not a direct LST measurement.",
            ]:
                solara.Text(f"· {lim}", style="font-size:12px; color:#6e7681; line-height:1.5;")
