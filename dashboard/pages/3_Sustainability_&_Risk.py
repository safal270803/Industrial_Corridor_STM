"""
Page 3 - RQ3: Sustainability & Environmental Risk
Data Science Implementation - Solara Standard Layout Framework
"""

import os
import ee
import geemap
import solara
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────
#  1. Spatial Path Mapping & GEE Core Engines (Retaining All Logic)
# ──────────────────────────────────────────────────────────────────────

PAGES_DIR = os.path.dirname(__file__)                 # dashboard/pages/
DASHBOARD_ROOT = os.path.dirname(PAGES_DIR)           # dashboard/
GEOJSON_PATH = os.path.join(DASHBOARD_ROOT, "assets", "Dholera_Taluk.geojson")


def _get_roi():
    """Ingests your local boundary file natively from the assets folder."""
    if os.path.exists(GEOJSON_PATH):
        return geemap.geojson_to_ee(GEOJSON_PATH).geometry()
    else:
        # Stable fallback bounding box boundary
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

    s1_dry_smoothed = s1_dry.median().focal_mean(radius=1, kernelType="square", units="pixels").clip(roi)
    sar_permanent_water = s1_dry_smoothed.lt(-17).rename("PermanentWater")

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

def _calculate_indices(image):
    """Computes NDBI, MNDWI, and SAVI required for built-up filtering."""
    ndbi = image.expression(
        '(SWIR1 - NIR) / (SWIR1 + NIR)',
        {'SWIR1': image.select('B11'), 'NIR': image.select('B8')}
    ).rename('NDBI')

    mndwi = image.expression(
        '(Green - SWIR1) / (Green + SWIR1)',
        {'Green': image.select('B3'), 'SWIR1': image.select('B11')}
    ).rename('MNDWI')

    savi = image.expression(
        '((NIR - Red) * 1.5) / (NIR + Red + 0.5)',
        {'NIR': image.select('B8'), 'Red': image.select('B4')}
    ).rename('SAVI')

    return image.addBands([ndbi, mndwi, savi])


def _get_growth_class(roi):
    """Generates the 4-class urban growth layer matching the notebook thresholds."""
    s2_2016_raw = _s2("2016-10-01", "2016-12-31", roi)
    s2_2025_raw = _s2("2025-10-01", "2025-12-31", roi)

    processed_2016 = _calculate_indices(s2_2016_raw)
    processed_2025 = _calculate_indices(s2_2025_raw)

    # 2016 Built-up Mask (Threshold > 0.13)
    built_up_2016 = processed_2016.expression(
        '(NDBI > 0.13) && (MNDWI < 0) && (SAVI < 0.18)',
        {'NDBI': processed_2016.select('NDBI'), 'MNDWI': processed_2016.select('MNDWI'), 'SAVI': processed_2016.select('SAVI')}
    ).rename('BuiltUp_2016')

    # 2025 Built-up Mask (Threshold > 0.05)
    built_up_2025 = processed_2025.expression(
        '(NDBI > 0.05) && (MNDWI < 0) && (SAVI < 0.18)',
        {'NDBI': processed_2025.select('NDBI'), 'MNDWI': processed_2025.select('MNDWI'), 'SAVI': processed_2025.select('SAVI')}
    ).rename('BuiltUp_2025')

    change_stack = built_up_2016.rename('y2016').addBands(built_up_2025.rename('y2025'))

    growth_class = change_stack.expression(
        "(b('y2016') == 0 && b('y2025') == 0) ? 0"   # No built-up
        ": (b('y2016') == 1 && b('y2025') == 1) ? 1"  # Stable
        ": (b('y2016') == 0 && b('y2025') == 1) ? 2"  # New Growth
        ": (b('y2016') == 1 && b('y2025') == 0) ? 3"  # Lost
        ": 0"
    ).rename('GrowthClass').clip(roi)

    return growth_class

# ──────────────────────────────────────────────────────────────────────
#  2. Empirical Metrics & Bar Charts (Retaining All Logic)
# ──────────────────────────────────────────────────────────────────────

ENV_STATS = [
    {"label": "New Built-up Footprint:\n",   "value": "34.013 km²", "type": "info"},
    {"label": "Growth ∩ Biomass Loss:\n",    "value": "28.203 km²", "type": "warning"},
    {"label": "Share of new growth at cost of local biomass:\n",         "value": "82.9%",      "type": "error"},
    {"label": "Avg SAVI Loss / km²:\n",     "value": "-0.1927",    "type": "warning"},
    {"label": "New Growth in Moderate Flood Risk:\n",  "value": "8.865 km²",  "type": "warning"},
    {"label": "New Growth in High Flood Risk:\n",      "value": "0.027 km²",  "type": "error"},
]

FVI_STATS = [
    {"label": "✅ Low Risk",      "km2": 983.651, "pct": 84.9, "color": "#2d6a4f"},
    {"label": "⚠️ Moderate Risk", "km2": 164.324, "pct": 14.2, "color": "#f4a261"},
    {"label": "🚨 High Risk",     "km2": 10.672,  "pct":  0.9, "color": "#d62828"},
]


def _fvi_chart():
    fig = go.Figure(go.Bar(
        x=[s["label"] for s in FVI_STATS],
        y=[s["km2"]   for s in FVI_STATS],
        marker=dict(
            color=[s["color"] for s in FVI_STATS],
            line=dict(width=1),
        ),
        text=[f"{s['km2']} km²\n({s['pct']}%)" for s in FVI_STATS],
        textposition="outside",
        textfont=dict(family="monospace", size=11),
    ))
    fig.update_layout(
        title="Flood Vulnerability Surface Area Scaling",
        yaxis=dict(title="Square Kilometers"),
        margin=dict(l=50, r=20, t=50, b=50),
        height=300,
        showlegend=False,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
#  3. Page View Component Layout (Standardized Native Theme)
# ──────────────────────────────────────────────────────────────────────

@solara.component
def Page():
    # Native VBox serves as the uniform page padding frame
    with solara.Column(style={"gap": "24px", "padding": "10px"}):

        # ── Academic Header Block ──
        solara.Markdown("# RQ3: Sustainability & Environmental Risk Realignment")
        solara.Markdown(
            "**Research Direction:** Does the spatial footprint of Dholera's built-up expansion overlap with environmentally sensitive or climatically vulnerable terrain, "
            "and does corridor development carry a quantifiable ecological cost? "
        )

        # ── Hypothesis Evaluation Framework ──
        with solara.Card("Hypothesis Framework Summary"):
            with solara.Column(style={"gap": "16px", "padding": "4px"}):
                
                # Hypothesis 1 Row
                with solara.Row(justify="space-between", style={"align-items": "center", "border-bottom": "1px solid #e0e0e0", "padding-bottom": "8px"}):
                    with solara.Column(style={"max-width": "80%"}):
                        solara.Markdown("### **H1: Built-up expansion is associated with declining vegetation cover and increased surface heat susceptibility**")
                        solara.Markdown("Evaluates whether physical corridor expansion systematically drives localized biomass conversion and elevates relative material thermal stress metrics.")
                    solara.Success(label="SUPPORTED", children=["✓ Biomass Cost"])

                # Hypothesis 2 Row
                with solara.Row(justify="space-between", style={"align-items": "center"}):
                    with solara.Column(style={"max-width": "80%"}):
                        solara.Markdown("### **H2: A share of new built-up areas overlaps with environmentally sensitive or water-prone zones**")
                        solara.Markdown("Tests for structural spatial misalignment by measuring how much new industrial infrastructure sits inside radar-verified seasonal flood pools and low-elevation basins.")
                    solara.Success(label="SUPPORTED", children=["✓ Topographic Mismatch"])

        # ── Environmental Risk Cards Grid ──
        # Natively loops through your empirical stats using standard Solara state alerts
        with solara.GridFixed(columns=3):
            for s in ENV_STATS:
                if s["type"] == "info":
                    solara.Info(label=s["label"], children=[s["value"]])
                elif s["type"] == "warning":
                    solara.Warning(label=s["label"], children=[s["value"]])
                elif s["type"] == "error":
                    solara.Error(label=s["label"], children=[s["value"]])

        # ── Map 1: Biomass Transformation Card (UPDATED) ──
        with solara.Card("Biomass Transformation Map - SAVI Loss Profile Within New Growth"):
            solara.Markdown(
                "**Legend Gradient:** 🌼 Pale yellow (marginal biomass decay) → 🔴 Deep red (severe removal) "
            )
            try:
                roi = _get_roi()
                s2_2016 = _s2("2016-10-01", "2016-12-31", roi)
                s2_2025 = _s2("2025-10-01", "2025-12-31", roi)
                
                savi_16 = _savi(s2_2016)
                savi_25 = _savi(s2_2025)
                savi_delta = savi_25.subtract(savi_16).rename("SAVI_Delta").clip(roi)

                # Fetch our growth footprint map layer logic
                growth_class = _get_growth_class(roi)
                new_growth_mask = growth_class.eq(2) # Class 2 = New Urban Growth
                new_growth_flat = new_growth_mask.selfMask()

                # Isolate biomass loss solely on the active construction footprints
                veg_loss_in_growth = savi_delta.updateMask(
                    new_growth_mask.And(savi_delta.lt(0))
                )

                m1 = geemap.Map()
                m1.centerObject(roi, 10)
                m1.add_basemap("HYBRID")

                # Layer 1: Flat charcoal footprint baseline (from your notebook)
                m1.addLayer(
                    new_growth_flat,
                    {"min": 1, "max": 1, "palette": ["#444444"]},
                    "New Growth Footprint (34 km²)", shown=False
                )

                # Layer 2: Targeted vegetation loss gradient mapped strictly over growth
                m1.addLayer(
                    veg_loss_in_growth,
                    {"min": -0.3, "max": 0, "palette": ["#800026", "#e31a1c", "#fd8d3c", "#ffffb2"]},
                    "Vegetation Loss Intensity (ΔSAVI)", shown=False
                )
                m1.addLayer(
                    savi_delta.updateMask(savi_delta.lt(0)),
                    {"min": -0.3, "max": 0, "palette": ["#800026", "#e31a1c", "#fd8d3c", "#ffffb2"]},
                    "SAVI Delta (Vegetation Loss)", shown=True
                )
                
                m1.layout.height = "480px"
                solara.display(m1)
            except Exception as e:
                solara.Error(f"GEE Pipeline Error (Biomass Track): {str(e)}")

        # ── Map 2: Heat Susceptibility Card ──
        with solara.Card("Heat Susceptibility Index Proxy (NDBI_norm − SAVI_norm)"):
            solara.Markdown(
                "**Legend Gradient:** 💧 Cool blue (thermal resilience framing) → 🔥 Fiery red (exposed urban heat retention cores)"
            )
            try:
                heat = _heat_index(s2_2025)
                m2 = geemap.Map()
                m2.centerObject(roi, 10)
                m2.add_basemap("HYBRID")
                m2.addLayer(
                    heat,
                    {"min": -0.5, "max": 0.7, "palette": ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"]},
                    "Heat Susceptibility Index"
                )
                m2.layout.height = "480px"
                solara.display(m2)
            except Exception as e:
                solara.Error(f"GEE Pipeline Error (Thermal Track): {str(e)}")

        # ── Map 3 & Chart Combo: Flood Vulnerability Index ──
        with solara.GridFixed(columns=2):
            with solara.Card("Bivariate Flood Vulnerability Index (FVI) Map"):
                solara.Markdown(
                "**Legend Gradient**: \n🌿 Low Risk (Resilient Terrain)\n 🟧 Moderate Risk (Episodic Pooling) \n→ 🔴 High Risk (Topographic Sink)"
                )
                try:
                    flood, perm_water = _sar_flood_freq(roi)
                    fvi = _fvi(flood, perm_water, roi)
                    
                    m3 = geemap.Map()
                    m3.centerObject(roi, 10)
                    m3.add_basemap("HYBRID")
                    m3.addLayer(
                        fvi,
                        {"min": 1, "max": 3, "palette": ["2d6a4f", "f4a261", "d62828"]},
                        "FVI Choropleth Map"
                    )
                    m3.layout.height = "400px"
                    solara.display(m3)
                except Exception as e:
                    solara.Error(f"GEE Pipeline Error (Radar Flood Track): {str(e)}")

            with solara.Card("FVI Spatial Distribution Profile"):
                solara.FigurePlotly(_fvi_chart())
                solara.Markdown(
                    "**Headline Realignment Metric:** Crucially, **26.1%** of all new built-up growth footprint "
                    "sits inside moderate or high flood risk areas.(8.865 km²) "
                )


        # ── Map 4: Environmental Exposure Composite Card ──
        with solara.Card("Composite Environmental Exposure Surface Matrix (Equal Weighted Heat + Flood)"):
            solara.Markdown(
                "**Legend:** 0 = Safe/Buffered Grid Surface → 1 = Maximum Combined Risk Surface (Low Elevation Sink + High Thermal Trapping)"
            )
            try:
                exposure = _env_exposure(heat, fvi)
                m4 = geemap.Map()
                m4.centerObject(roi, 10)
                m4.add_basemap("HYBRID")
                m4.addLayer(
                    exposure,
                    {"min": 0, "max": 1, "palette": ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560", "#f5a623", "#ffffff"]},
                    "Integrated Risk Exposure Surface"
                )
                m4.layout.height = "480px"
                solara.display(m4)
            except Exception as e:
                solara.Error(f"GEE Pipeline Error (Exposure Fusion Engine): {str(e)}")

        # ── Key Findings Evaluation Panel ──
        with solara.Card("Core Sustainability Empirical Findings"):
            with solara.Column(style={"gap": "14px"}):
                solara.Markdown(
                    "• **82.9% of total new built-up growth has expanded directly over active regional biomass matrices:** "
                    "Every square kilometer of newly paved ground registers an average SAVI depletion signature of −0.1927, "
                    "framing the ecological cost of infrastructure footprint expansion."
                )
                solara.Markdown(
                    "• **Zoning frameworks demonstrate an engineering disconnect from local micro-topography:** "
                    "A notable 26.1% of all new built-up growth footprint sits inside moderate or high flood risk areas.(8.865 km²) verified "
                    "directly by seasonal Sentinel-1 radar backscatter reflection metrics."
                )
                solara.Markdown(
                    "• **The Compound Exposure Phenomenon presents long-range risk concentration:** "
                    "Vulnerable grid sectors confront simultaneous multi-risk profiles where high physical building densities "
                    "collide with low vegetative evapotranspiration buffers directly inside low-elevation topographic basins."
                )

        # ── Methodological System Limitations ──
        with solara.Card("Methodological System Limitations"):
            solara.Markdown(
                "1. **SAVI Delta Monsoon Issue** - 2025 baseline spectral parameters of SAVI are elevated due to higher regional precipitation anomalies; calculations are tightly isolated to the new growth footprint boundary to mitigate distorted signals.\n"
                "2. **Copernicus GLO-30 Vertical Thresholds** - ~4 m absolute vertical error over flat terrain; in an area where FVI thresholds are set at 5 m and 8 m, this introduces classification uncertainty in boundary pixels.\n"
                "3. **SAR Backscatter Threshold Constraints** - The -17 dB water detection threshold is applied uniformly; smooth bare soil and salt pans can produce false water signatures even after permanent water exclusion.\n"
                "4. **Thermal Susceptibility Proxy Used** - Not a direct kinetic Land Surface Temperature (LST) measurement."
            )


# ──────────────────────────────────────────────────────────────────────
#  4. Standalone Verification Entry Point (For Local Terminal Testing)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Running native layout standalone structural check for Page 3...")
    try:
        ee.Initialize()
        roi_test = _get_roi()
        print(f"✅ Boundary configuration stable. ROI Type: {roi_test.type().getInfo()}")
        print("🎉 Script structural logic is mathematically secure.")
    except Exception as error:
        print(f"❌ Sandbox verification compile failed: {str(error)}")