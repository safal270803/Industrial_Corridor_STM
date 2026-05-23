"""
Dholera SIR — Spatial Intelligence Dashboard
pages/__init__.py  →  Shared layout template + GEE authentication
"""

import os
import json
import solara
import ee


# ─────────────────────────────────────────
#  GEE Authentication (runs once at startup)
# ─────────────────────────────────────────

def _init_gee():
    """Authenticate with GEE using Service Account stored in HF Secrets."""
    try:
        sa_email = os.environ.get("GEE_SERVICE_ACCOUNT_EMAIL", "")
        sa_key_json = os.environ.get("GEE_SERVICE_ACCOUNT_KEY", "")

        if sa_email and sa_key_json:
            key_dict = json.loads(sa_key_json)
            credentials = ee.ServiceAccountCredentials(sa_email, key_data=json.dumps(key_dict))
            ee.Initialize(credentials)
        else:
            # Fallback: try default credentials (works locally if you ran `earthengine authenticate`)
            ee.Initialize()

        print("✅ GEE Initialized successfully.")
    except Exception as e:
        print(f"⚠️  GEE init failed: {e}")


_init_gee()


# ─────────────────────────────────────────
#  Shared Layout Component
# ─────────────────────────────────────────

@solara.component
def Layout(children=[]):
    """
    Universal dashboard shell — wraps every page with a consistent
    header and navigation sidebar (Solara builds the sidebar automatically
    from the pages/ folder; we just provide the top banner).
    """
    with solara.Column(style="min-height: 100vh; background: #0d1117;"):

        # ── Top Banner ──────────────────────────────────────────
        with solara.Row(
            style=(
                "background: linear-gradient(90deg, #0d1117 0%, #161b22 100%);"
                "border-bottom: 1px solid #30363d;"
                "padding: 16px 32px;"
                "align-items: center;"
                "gap: 16px;"
            )
        ):
            solara.Text(
                "🛰️",
                style="font-size: 28px; line-height: 1;",
            )
            with solara.Column(style="gap: 0px;"):
                solara.Text(
                    "Dholera SIR — Spatial Intelligence Dashboard",
                    style=(
                        "font-family: 'IBM Plex Mono', monospace;"
                        "font-size: 18px;"
                        "font-weight: 700;"
                        "color: #e6edf3;"
                        "letter-spacing: 0.5px;"
                    ),
                )
                solara.Text(
                    "Google Earth Engine  ·  Sentinel-2  ·  VIIRS  ·  SAR",
                    style=(
                        "font-family: 'IBM Plex Mono', monospace;"
                        "font-size: 11px;"
                        "color: #8b949e;"
                        "letter-spacing: 1px;"
                    ),
                )

        # ── Page Content ────────────────────────────────────────
        with solara.Column(style="flex: 1; padding: 0;"):
            solara.display(*children)
