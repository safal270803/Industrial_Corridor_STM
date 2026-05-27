"""
Dholera SIR — Spatial Intelligence Dashboard
pages/__init__.py  →  Shared layout template + Master GEE authentication
"""

import os
import json
import solara
import ee

def _init_gee():
    """Initializes Google Earth Engine securely via Service Account secrets or local fallback."""
    try:
        sa_key_json = os.environ.get("EE_SERVICE_ACCOUNT_KEY", "")
        if sa_key_json:
            key_dict = json.loads(sa_key_json)
            credentials = ee.ServiceAccountCredentials(
                client_email=key_dict["client_email"], 
                key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials=credentials)
            print("✅ GEE Initialized successfully via Service Account.")
        else:
            ee.Initialize()
            print("✅ GEE Initialized successfully via local credentials.")
    except Exception as e:
        print(f"⚠️ GEE System Initialization Failure: {e}")

_init_gee()


@solara.component
def Layout(children=[]):
    """Universal app shell template with customized typography updates."""
    
    # 1. Map your customized uppercase title route dictionary
    # Solara checks custom page naming keys through standard layout configs
    solara.lab.theme.sidebar_title = "INFRASTRUCTURE-LED URBANIZATION"
    
    with solara.AppLayout(navigation=True):
        
        # ── Global App Header Bar ──
        with solara.AppBar():
            with solara.Row(justify="space-between", style={"align-items": "center", "width": "100%", "padding-right": "10px"}):
                '''
                # Left Side Header Context
                with solara.Row(style={"align-items": "center", "gap": "12px"}):
                    solara.Text("🛰️", style={"font-size": "24px"})
                    with solara.Column(style={"gap": "2px", "color": "#ffffff"}):
                        solara.Markdown("### Dholera: Spatial Intelligence Dashboard")
                        solara.Text(
                            "Google Earth Engine · Sentinel-2 · VIIRS · SAR Imagery · GLO-30 DEM", 
                            style={"font-size": "11px", "opacity": "0.85"}
                        )
                '''
                # Right Side Grey Box Custom Branding — Refactored to display white text safely via clear css overrides
                with solara.Column(style={
                    "background-color": "rgba(255, 255, 255, 0.15)", 
                    "border-radius": "6px", 
                    "padding": "6px 14px",
                    "align-items": "flex-end",
                    "gap": "2px"
                }):
                    solara.Markdown(
                        "**🛰️ Dholera: Spatial Intelligence Dashboard**", 
                        style={"color": "#ffffff !important", "font-size": "12px", "margin": "0"}
                    )
                    solara.Text(
                        "Google Earth Engine · Sentinel-2 · VIIRS · SAR Imagery · GLO-30 DEM", 
                        style={"color": "#ffffff", "font-size": "10px", "opacity": "0.9"}
                    )

        # ── Reactive Page Canvas Container ──
        with solara.Padding(24):
            solara.Column(children=children, style={"width": "100%"})