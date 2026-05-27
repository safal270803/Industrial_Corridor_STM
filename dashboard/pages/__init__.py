"""
Dholera SIR — Spatial Intelligence Dashboard
pages/__init__.py  →  Shared layout template + Master GEE authentication
"""

import os
import json
import solara
import ee

title = "INFRASTRUCTURE-LED URBANIZATION" 

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
    """
    Universal app shell template with structural alignment fixes.
    """
    # 1. Force the sidebar navigation configuration to use your custom uppercase title
    solara.lab.theme.sidebar_title = "INFRASTRUCTURE-LED URBANIZATION"
    
    with solara.AppLayout(navigation=True):
        
        # ── Global App Header Bar ──
        with solara.AppBar():
            # FIX: We use a raw HTML-style flex wrapper with explicit dimensions to bypass Solara's AppBar row scaling constraints
            with solara.Div(style={
                "display": "flex",
                "align-items": "center",
                "justify-content": "flex-end",
                "width": "100%",
                "min-width": "100%",
                "padding": "0 16px 0 0",
                "box-sizing": "border-box"
            }):
                
                
                # Right Side Grey Box Custom Branding (Aligned perfectly with zero overflow)
                with solara.Column(style={
                    "background-color": "rgba(255, 255, 255, 0.12)", 
                    "border-radius": "6px", 
                    "padding": "6px 16px",
                    "align-items": "flex-end",
                    "justify-content": "center",
                    "gap": "1px",
                    "margin-left": "auto"
                }):
                    solara.Markdown(
                        "**🛰️DHOLERA: SPATIAL INTELLIGENCE DASHBOARD**", 
                        style={"color": "#ffffff !important", "font-size": "11px", "margin": "0", "letter-spacing": "0.5px"}
                    )
                    solara.Text(
                        "Infrastructure-Led Urbanization Patterns ", 
                        style={"color": "#ffffff", "font-size": "10px", "opacity": "0.9"}
                    )
                    # New subtext row added here:
                    solara.Text(
                        "Google Earth Engine · Sentinel-2 · VIIRS · SAR Imagery · GLO-30 DEM", 
                        style={"color": "#ffffff", "font-size": "9px", "opacity": "0.75"}
                    )

        # ── Reactive Page Canvas Container ──
        with solara.Padding(24):
            solara.Column(children=children, style={"width": "100%"})

# ─── ADD THIS AT THE ABSOLUTE BOTTOM OF YOUR pages/__init__.py ───

# 1. Define a basic placeholder component for the root tab
@solara.component
def RootPage():
    with solara.Div(style={"padding": "24px"}):
        solara.Markdown("## Welcome to the Spatial Intelligence Dashboard")
        solara.Markdown("Select a module from the menu above to begin exploring.")

# 2. Tell Solara explicitly to name the root tab "INFRASTRUCTURE-LED URBANIZATION"
# and link your other physical python files to the navbar
routes = [
    solara.Route(path="/", component=RootPage, label="INFRASTRUCTURE-LED URBANIZATION"),
    solara.Route(path="1-infrastructure-growth", label="INFRASTRUCTURE & GROWTH"),
    solara.Route(path="2-economic-utilization", label="ECONOMIC UTILIZATION"),
    solara.Route(path="3-sustainability-risk", label="SUSTAINABILITY & RISK"),
]