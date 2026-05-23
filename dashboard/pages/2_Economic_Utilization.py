"""
Page 2 — RQ2: Economic Utilization & Cluster Typology
VIIRS Nighttime Lights · IUR · Ghost Grid
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


def _cluster_map(s2, viirs_roi):
    ndbi  = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")
    mndwi = s2.normalizedDifference(["B3",  "B11"]).rename("MNDWI")
    savi  = s2.expression(
        "((NIR - RED) * 1.5) / (NIR + RED + 0.5)",
        {"NIR": s2.select("B8"), "RED": s2.select("B4")}
    ).rename("SAVI")

    built_up = ndbi.gt(0.05).And(mndwi.lt(0)).And(savi.lt(0.18))
    is_active_night = viirs_roi.gt(0.6)
    is_soil_disturbed = (
        savi.gt(0.18).And(savi.lt(0.3))
        .And(ndbi.gt(0.05))
        .And(mndwi.lt(-0.4))
    )

    return (
        ee.Image(0)
        .where(is_soil_disturbed,                              3)
        .where(built_up.And(is_active_night.Not()),            2)
        .where(built_up.And(is_active_night),                  1)
        .rename("ClusterType")
    )


# ─────────────────────────────────────────
#  Stats (hardcoded from notebook run)
# ─────────────────────────────────────────

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
            "border-radius:8px; padding:16px 20px; min-width:180px; gap:4px;"
        )
    ):
        solara.Text(label, style="font-size:12px; color:#8b949e; font-family:'IBM Plex Mono',monospace;")
        solara.Text(value, style=f"font-size:22px; font-weight:700; color:{color}; font-family:'IBM Plex Mono',monospace;")
        solara.Text(sub,   style="font-size:11px; color:#8b949e; font-family:'IBM Plex Mono',monospace;")


# ─────────────────────────────────────────
#  IUR Donut Chart
# ─────────────────────────────────────────

def _iur_chart():
    labels = [s["label"] for s in CLUSTER_STATS]
    values = [s["pct"]   for s in CLUSTER_STATS]
    colors = [s["color"] for s in CLUSTER_STATS]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
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
            text="Industrial Utilisation Ratio (IUR) — Dholera SIR 2025",
            font=dict(family="IBM Plex Mono", size=13, color="#e6edf3"),
        ),
        legend=dict(
            bgcolor="#161b22", bordercolor="#30363d",
            font=dict(family="IBM Plex Mono", color="#c9d1d9", size=11),
            orientation="v",
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=360,
    )
    return fig


# ─────────────────────────────────────────
#  Bar Chart — Area by Class
# ─────────────────────────────────────────

def _area_chart():
    labels = [s["label"] for s in CLUSTER_STATS]
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
        title=dict(text="Area by Cluster Class (km²)", font=dict(family="IBM Plex Mono", size=13, color="#e6edf3")),
        yaxis=dict(title="km²", gridcolor="#21262d", color="#8b949e"),
        xaxis=dict(color="#8b949e"),
        margin=dict(l=50, r=20, t=50, b=80),
        height=320,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────
#  Main Page Component
# ─────────────────────────────────────────

@solara.component
def Page():

    with solara.Column(style="padding:24px 32px; gap:24px; background:#0d1117; min-height:100vh;"):

        # ── Header ──────────────────────────────────────────────
        with solara.Column(style="gap:4px;"):
            solara.Text(
                "RQ2 — Economic Utilization & Cluster Typology",
                style="font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:700; color:#e6edf3;",
            )
            solara.Text(
                "VIIRS Nighttime Lights (Oct 2025–Mar 2026)  ·  Sentinel-2 Built-up Mask",
                style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#8b949e; letter-spacing:1px;",
            )
            solara.Text(
                "What proportion of Dholera's built-up footprint is economically active, "
                "and how is industrial utilisation spatially distributed across the corridor?",
                style="font-size:14px; color:#c9d1d9; max-width:860px; line-height:1.6;",
            )

        # ── IUR Hero Stat ────────────────────────────────────────
        with solara.Row(style="gap:12px; flex-wrap:wrap; align-items:stretch;"):
            with solara.Column(
                style=(
                    "background:linear-gradient(135deg,#161b22 0%,#1c2128 100%);"
                    "border:1px solid #e63946; border-radius:10px; padding:24px 32px;"
                    "align-items:center; justify-content:center; min-width:180px; gap:4px;"
                )
            ):
                solara.Text("Industrial Utilisation Ratio",
                            style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8b949e; letter-spacing:1.5px;")
                solara.Text(f"{IUR}%",
                            style="font-family:'IBM Plex Mono',monospace; font-size:48px; font-weight:700; color:#e63946;")
                solara.Text(f"of {TOTAL_KM2} km² total built-up",
                            style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8b949e;")

            with solara.Column(style="gap:12px; flex:1; min-width:240px;"):
                for s in CLUSTER_STATS:
                    StatCard(s["label"], f"{s['km2']} km²", f"{s['pct']}% of built-up", s["color"])

        # ── Map ──────────────────────────────────────────────────
        with solara.Column(
            style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden;"
        ):
            with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d; align-items:center; gap:12px;"):
                solara.Text("🌃", style="font-size:18px;")
                solara.Text(
                    "Cluster Typology Map — Active / Dormant / Under-Construction",
                    style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;",
                )

            try:
                roi = _get_roi()
                s2  = _s2_2025(roi)
                viirs_roi = _viirs(roi)
                cluster = _cluster_map(s2, viirs_roi)

                m = geemap.Map(center=[22.35, 72.10], zoom=10)
                m.add_basemap("HYBRID")

                m.addLayer(
                    viirs_roi,
                    {"min": 0, "max": 20, "palette": ["000000", "3d0066", "ff6600", "ffff99"]},
                    "VIIRS Nightlights (Inferno)", shown=False
                )
                m.addLayer(
                    cluster,
                    {"min": 0, "max": 3,
                     "palette": ["1a1a1a", "e63946", "6a0572", "f4a261"]},
                    "Cluster Typology (RQ2)"
                )
                m.add_legend(
                    title="Cluster Typology",
                    legend_dict={
                        "⬛ Background":            "1a1a1a",
                        "🔴 Active Industrial":     "e63946",
                        "🟣 Dormant / Speculative": "6a0572",
                        "🟠 Under-Construction":    "f4a261",
                    },
                    position="bottomright"
                )
                m.layout.height = "520px"
                solara.display(m)

            except Exception as e:
                with solara.Column(style="padding:40px; align-items:center;"):
                    solara.Text(
                        f"⚠️  Map unavailable — GEE connection required.\n{e}",
                        style="color:#f0883e; font-family:'IBM Plex Mono',monospace; font-size:12px;",
                    )

        # ── Charts Row ───────────────────────────────────────────
        with solara.Row(style="gap:16px; flex-wrap:wrap;"):
            with solara.Column(
                style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden; flex:1; min-width:300px;"
            ):
                with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d;"):
                    solara.Text("IUR Breakdown", style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;")
                with solara.Column(style="padding:12px;"):
                    solara.FigurePlotly(_iur_chart())

            with solara.Column(
                style="background:#161b22; border:1px solid #30363d; border-radius:10px; overflow:hidden; flex:1; min-width:300px;"
            ):
                with solara.Row(style="padding:14px 20px; border-bottom:1px solid #30363d;"):
                    solara.Text("Area by Class", style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:#e6edf3;")
                with solara.Column(style="padding:12px;"):
                    solara.FigurePlotly(_area_chart())

        # ── Key Findings ─────────────────────────────────────────
        with solara.Column(
            style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px 24px; gap:14px;"
        ):
            solara.Text("Key Findings", style="font-family:'IBM Plex Mono',monospace; font-size:14px; font-weight:700; color:#e6edf3;")
            for finding, color in [
                ("Physical infrastructure has outpaced economic activation. An IUR of 35.2% means nearly two-thirds of built-up land shows no measurable nighttime radiance.", "#e63946"),
                ("The 31.1% Dormant share is not failure — it is intent. Roads in Dholera activate land value for future investors, not current factories. This is the 'Ghost Grid' pattern.", "#6a0572"),
                ("33.7% Under-Construction confirms a pipeline nearly as large as the active footprint. The corridor is in a state of hyper-transformation.", "#f4a261"),
            ]:
                with solara.Row(style="align-items:flex-start; gap:12px;"):
                    solara.Text("▸", style=f"color:{color}; font-size:16px; line-height:1.5; flex-shrink:0;")
                    solara.Text(finding, style="font-size:13px; color:#c9d1d9; line-height:1.6;")
