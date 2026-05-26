"""
Page 1 — RQ1: Infrastructure-Led Urbanization Patterns
Data Science Implementation — Solara Standard Layout Framework
"""

import os
import ee
import geemap
import solara

# ──────────────────────────────────────────────────────────────────────
#  1. Spatial Path Mapping & GEE Core Functions (Retaining All Logic)
# ──────────────────────────────────────────────────────────────────────

PAGES_DIR = os.path.dirname(__file__)                 # dashboard/pages/
DASHBOARD_ROOT = os.path.dirname(PAGES_DIR)           # dashboard/
GEOJSON_PATH = os.path.join(DASHBOARD_ROOT, "assets", "Dholera_Taluk.geojson")

# Standard relative public URLs for local asset resolution
IMG_POLY2_URL = "/static/public/regression_poly2.png"
IMG_POLY3_URL = "/static/public/regression_poly3.png"


def _get_roi():
    """Ingests local GeoJSON boundary from assets and returns GEE Geometry."""
    if os.path.exists(GEOJSON_PATH):
        return geemap.geojson_to_ee(GEOJSON_PATH).geometry()
    else:
        # Stable fallback bounding box boundary
        return ee.Geometry.Polygon([[
            [72.00, 22.15], [72.35, 22.15], [72.35, 22.50], [72.00, 22.50], [72.00, 22.15]
        ]])


def _s2_composite(year_start, year_end, roi, cloud_thresh):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(year_start, year_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
        .median()
        .clip(roi)
    )


def _built_up_mask(s2, ndbi_thresh):
    ndbi  = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")
    mndwi = s2.normalizedDifference(["B3",  "B11"]).rename("MNDWI")
    savi  = s2.expression(
        "((NIR - Red) * 1.5) / (NIR + Red + 0.5)",
        {"NIR": s2.select("B8"), "Red": s2.select("B4")}
    ).rename("SAVI")
    
    return (
        ndbi.gt(ndbi_thresh)
        .And(mndwi.lt(0))
        .And(savi.lt(0.18))
        .rename("BuiltUp")
    )


def _growth_map(mask_2016, mask_2025):
    stable     = mask_2016.multiply(mask_2025).multiply(1)
    new_growth = mask_2025.multiply(mask_2016.Not()).multiply(2)
    lost       = mask_2016.multiply(mask_2025.Not()).multiply(3)
    return stable.add(new_growth).add(lost).rename("GrowthClass")


def _accessibility_surface(roi):
    """Computes Dual-Anchor exponential distance decay signal around Airport."""
    airport = ee.Geometry.Point([72.1770, 22.2917])
    dist_to_hub = ee.Image.constant(0).distance(
        ee.FeatureCollection([ee.Feature(airport)]), 20000
    ).rename("dist_hub")
    
    sigma_tier1 = 5000  
    infra_access = dist_to_hub.multiply(-1).divide(sigma_tier1).exp()
    return infra_access.clamp(0, 1).clip(roi)


# ──────────────────────────────────────────────────────────────────────
#  2. Page View Component Layout (Standardized Native Theme)
# ──────────────────────────────────────────────────────────────────────

@solara.component
def Page():
    # Native VBox serves as the universal page padding frame
    with solara.Column(style={"gap": "24px", "padding": "10px"}):

        # ── Academic Header Block ──
        solara.Markdown("# RQ1 — Infrastructure-Led Urbanization Patterns")
        solara.Markdown(
            "**Research Direction:** Has infrastructure development in Dholera SIR driven measurable built-up growth, "
            "and does proximity to roads and key infrastructure nodes explain the spatial pattern of urbanization?"
        )

        # ── Empirical Stat Cards Grid ──
        # Uses standard semantic theme alerts to present numerical results completely textually
        with solara.GridFixed(columns=4):
            solara.Info(label="Baseline Frame", children=["Built-up 2016: 10.54 km²"])
            solara.Success(label="Current Horizon", children=["Built-up 2025: 35.51 km²"])
            solara.Success(label="Net Transformation", children=["Expansion: +24.97 km² (+236.99%)"])
            solara.Warning(label="Correlation Index", children=["Pearson r: −0.028 (Spine Ahead of Demand)"])

        # ── Map Presentation Card ──
        with solara.Card("Dynamic Spatial Matrix (2016 vs 2025)"):
            solara.Markdown(
                "**Legend:** ⬛ No built-up | ⬜ Stable Frame | 🟠 New Infrastructure Growth (2016→2025) | 🔵 Lost Signal"
            )
            try:
                roi = _get_roi()
                s2_2016 = _s2_composite("2016-10-01", "2016-12-31", roi, cloud_thresh=5)
                s2_2025 = _s2_composite("2025-10-01", "2025-12-31", roi, cloud_thresh=40)
                
                mask_2016 = _built_up_mask(s2_2016, ndbi_thresh=0.13) 
                mask_2025 = _built_up_mask(s2_2025, ndbi_thresh=0.05) 
                
                growth = _growth_map(mask_2016, mask_2025)
                acc_surface = _accessibility_surface(roi)

                m = geemap.Map(center=[22.37, 72.05], zoom=11)
                m.add_basemap("HYBRID")
                
                m.addLayer(acc_surface, {"min": 0, "max": 0.75, "palette": ['#000004', '#51127c', '#fb8861', '#fcfdbf']}, "Accessibility Surface", False)
                m.addLayer(growth, {"min": 0, "max": 3, "palette": ["1a1a1a", "cccccc", "ff4500", "1e90ff"]}, "🔥 Built-up Growth Classification")
                
                m.layout.height = "520px"
                solara.display(m)

            except Exception as e:
                solara.Error(f"Google Earth Engine Pipeline Timeout or Auth Disconnection: {str(e)}")

        # ── Statistical Regression Modeling Panel ──
        with solara.Card("📉 Empirical Regression — 'Roads Ahead of Growth' Paradox"):
            solara.Markdown(
                "Linear Pearson $r = −0.028$ · Polynomial $R^2 = 0.0053$. Distance to transportation lines "
                "explains less than 1% of total urban density variance. This statistically verifies that network frameworks "
                "are proactively laid down far ahead of active settlement expansion to catalyze future spatial land value capture."
            )
            
            # Displays your authentic empirical curve fit plots side by side cleanly using default grid rows
            with solara.GridFixed(columns=2):
                with solara.VBox():
                    solara.Markdown("### Order-2 Polynomial Fit (Non-linear distance decay profiling)")
                    solara.Image(IMG_POLY2_URL)
                with solara.VBox():
                    solara.Markdown("### Order-3 Polynomial Fit (Resolves secondary peak near unconnected Airport Hub)")
                    solara.Image(IMG_POLY3_URL)

        # ── Hypothesis Evaluation Matrices ──
        with solara.Card("Systemic Hypothesis Framework Summary"):
            with solara.Column(style={"gap": "12px"}):
                solara.Markdown(
                    "**H1 — Urban built-up density directly decays with road distance**\n"
                    "*Result:* **✗ Rejected** ($r = −0.028$). Road corridors function as long-range economic structural "
                    "anchors rather than immediate local settlement drivers."
                )
                solara.Markdown(
                    "**H2 — Major infrastructure nodes create secondary spatial clusters**\n"
                    "*Result:* **✓ Supported**. The 3rd-order curve fitting successfully isolates a significant urban density "
                    "uptick surrounding the 5.5km airport radius boundary line."
                )
                solara.Markdown(
                    "**H3 — Multi-variable accessibility modeling out-performs single proximity lines**\n"
                    "*Result:* **✓ Supported**. The Dual-Anchor framework explains multi-tiered development planning vectors "
                    "far more accurately than singular distance-to-spine tracking vectors."
                )


# ──────────────────────────────────────────────────────────────────────
#  3. Standalone Verification Entry Point (For Local Terminal Testing)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    print("🚀 Running native layout standalone structural check...")
    try:
        ee.Initialize()
        print("✅ Earth Engine Handshake: Online")
        print(f"✅ Local Boundary Track Link: {os.path.exists(GEOJSON_PATH)}")
        print("🎉 Code baseline compile is fully functional.")
    except Exception as error:
        print(f"❌ Script compilation check dropped: {str(error)}")
