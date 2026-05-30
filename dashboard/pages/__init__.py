"""
Dholera SIR — Spatial Intelligence Dashboard
pages/__init__.py  →  Shared layout template + Master GEE authentication
"""

import os
import json
import solara
import ee

title = "INFRASTRUCTURE-LED URBANIZATION" 

from google.oauth2 import service_account

def _init_gee_globally():
    print("🚀 TARGET ACQUIRED: Launching Earth Engine Handshake inside pages/__init__.py...")
    sa_key_json = os.environ.get("EE_SERVICE_ACCOUNT_KEY", "")
    
    if not sa_key_json:
        print("❌ CRITICAL ERROR: EE_SERVICE_ACCOUNT_KEY environment secret is empty!")
        return

    try:
        # Load the key dictionary safely
        key_dict = json.loads(sa_key_json)
        
        # Explicitly declare Google cloud authorization scopes
        scopes = [
            'https://www.googleapis.com/auth/earthengine', 
            'https://www.googleapis.com/auth/cloud-platform'
        ]
        
        # Authenticate using standard Google infrastructure tools
        credentials = service_account.Credentials.from_service_account_info(key_dict, scopes=scopes)
        ee.Initialize(credentials=credentials)
        print("✅ SUCCESS: Google Earth Engine authenticated globally for all dashboard tabs!")
    except Exception as e:
        print(f"❌ HANDSHAKE EXCEPTION: Initialization aborted due to: {str(e)}")

# Force the handshake execution immediately when Solara registers the multi-page folder
_init_gee_globally()

# Create a master layout wrapper so Solara binds the multi-page menu tabs perfectly
@solara.component
def Layout(children):
    return solara.AppLayout(children=children)


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