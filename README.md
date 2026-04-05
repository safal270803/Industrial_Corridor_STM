# Industrial_Corridor_STM
## RQ1 — Built-up Growth & Infrastructure Accessibility: Dholera SIR (2016–2025)

**Research Question:** Has infrastructure development in Dholera SIR driven measurable built-up growth, and does proximity to roads and key infrastructure nodes explain the spatial pattern of urbanization?

---

### What This Notebook Does

This notebook uses **Google Earth Engine (GEE)** + **Sentinel-2** satellite imagery to detect, quantify, and spatially analyze built-up land cover change across Dholera Taluka between 2016 and 2025.

---

### Setup

```bash
pip install geemap earthengine-api geopandas matplotlib seaborn scikit-learn
```

```python
import geemap
geemap.ee_initialize()   # Requires GEE authentication
```

---

### Data Requirements

| File | Location |
|---|---|
| Dholera Taluka boundary | `data/processed/Dholera_Taluk.geojson` |
| Major roads (OSM-extracted) | `data/Processed/important_roads.geojson` |
| Active infrastructure nodes | `data/Processed/dholera_active_infra.geojson` |
| Pre-sampled points (fast path) | `data/processed/dholera_points_2025.csv` |

> Roads were extracted from OSM via QGIS using highway filter: `motorway|trunk|primary|secondary|tertiary`

---

### Analytical Pipeline

#### Stage 1 — Sentinel-2 True Color (Study Area)

Oct–Dec composites were used for both years — lower SWIR soil reflectance in this season reduces false positives compared to summer or monsoon imagery.

**2025 Baseline (Oct–Dec 2025)**
<!-- Screenshot: S2 True Color 2025 — geemap layer "S2 True Color (2025)" centered on Dholera ROI -->
![S2 True Color 2025](output/screenshot/s2_truecolor_2025.png)

**2016 Baseline (Oct–Dec 2016)**
<!-- Screenshot: S2 True Color 2016 — geemap layer "S2 True Color (2016)" centered on Dholera ROI -->
![S2 True Color 2016](output/s2_truecolor_2016.png)

---

#### Stage 2 - Spectral Indices

Three indices are computed per image to isolate built-up land from confounders.

| Index | Formula | Purpose |
|---|---|---|
| NDBI | `(SWIR1 - NIR) / (SWIR1 + NIR)` | Detect built-up surfaces |
| MNDWI | `(Green - SWIR1) / (Green + SWIR1)` | Exclude water / salt pans |
| SAVI | `((NIR - Red) × 1.5) / (NIR + Red + 0.5)` | Exclude vegetation |

**NDBI Map (2025)** — Red/orange areas indicate residential or commercial zones
<!-- Screenshot: Map_NDBI layer with colorbar, centered on ROI -->
![NDBI 2025](output/ndbi_2025.png)

**MNDWI Map (2025)** — Deep blue pixels = lakes, rivers, or reservoirs
<!-- Screenshot: Map_MNDWI layer with colorbar, centered on ROI -->
![MNDWI 2025](output/mndwi_2025.png)

**SAVI Map (2025)** — Vibrant green = healthy biomass / vegetation
<!-- Screenshot: Map_SAVI layer with colorbar, centered on ROI -->
![SAVI 2025](output/savi_2025.png)

---

#### Stage 3 — Built-up Masks

Multi-index thresholds applied to isolate built-up pixels (white = built-up, black = everything else).

| Index | 2025 Threshold | 2016 Threshold |
|---|---|---|
| NDBI | > 0.05 | > 0.13 |
| MNDWI | < 0 | < 0 |
| SAVI | < 0.18 | < 0.18 |

> The stricter 2016 NDBI threshold (`0.13`) accounts for lower radiometric contrast in early Sentinel-2 data.

**Built-up Mask — 2025**
<!-- Screenshot: "Clean Built-up Layer" (2025) on top of S2 True Color, geemap -->
![Built-up Mask 2025](output/builtup_mask_2025.png)

**Built-up Mask — 2016**
<!-- Screenshot: "Clean Built-up Layer" (2016) on top of S2 True Color, geemap -->
![Built-up Mask 2016](output/builtup_mask_2016.png)

---

#### Stage 4 — Built-up Growth Heatmap (2016 → 2025)

Pixel-wise comparison of both masks generates a 4-class change map.

| Color | Class | Meaning |
|---|---|---|
| ⬛ Dark | 0 | No built-up in either year |
| ⬜ Gray | 1 | Stable — built-up in both 2016 & 2025 |
| 🔴 Orange-Red | 2 | New growth — urbanized between 2016 and 2025 |
| 🔵 Blue | 3 | Lost — was built-up in 2016, not in 2025 (QA) |

<!-- Screenshot: "Built-up Growth Heatmap (2016–2025)" layer with legend, geemap -->
![Growth Heatmap](output/growth_heatmap_2016_2025.png)

**Area Change Summary (fill in after running the notebook):**

| Metric | Value (km²) |
|---|---|
| Total Built-up 2016 | ___ |
| Total Built-up 2025 | ___ |
| Stable Built-up | ___ |
| New Growth | ___ |
| Lost Built-up | ___ |
| **Net Change** | ___ |
| **% Change** | ___ % |

---

#### Stage 5 — Road Proximity Analysis

2,000 random sample points generated inside the ROI. Each point is attributed with:
- `dist_m` — distance to major road network
- `builtup_density` — 250m focal mean of the 2025 built-up mask

A **7.5 km buffer** is applied — this threshold captures the airport, the key infrastructure anchor at the edge of the study zone.

**Sample Points on Built-up Layer**
<!-- Screenshot: geemap map showing cyan "Top Urban Hotspots" dots overlaid on the white built-up mask -->
![Sample Points](output/sample_points_builtup.png)

**Regression Plot — Distance vs. Built-up Density**

2nd-order polynomial fit (order=2) models the decay of built-up density with road distance. Order-3 is also noted to capture the secondary density spike near the airport (which currently lacks direct road connectivity as of March 2026).

<!-- Screenshot: Seaborn regplot output — "Dholera SIR: Urban Activation Signal" -->
![Regression Plot](output/regression_distance_vs_density.png)

---

#### Stage 6 — Master Accessibility Surface

A weighted fused raster combining road and infrastructure accessibility:

- **Road layer** — sigmoid decay by road hierarchy (motorway → tertiary), inflection at 3km
- **Infrastructure layer** — exponential decay per node (Tier 1: airport/factory σ=5km; Tier 2: power/solar σ=2km), following Weber's agglomeration logic
- **Final fusion** — `road_access × 0.4 + infra_access × 0.5`

**Road Network (color-coded by hierarchy)**
<!-- Screenshot: geemap map with color-coded road vectors overlaid — Red=Motorway, Orange=Trunk, Yellow=Primary, Green=Secondary, Magenta=Tertiary -->
![Road Network](output/road_network_hierarchy.png)

**Master Accessibility Surface**
<!-- Screenshot: "Master Accessibility Surface" heatmap layer with colorbar (dark=low, bright yellow=high), geemap -->
![Master Accessibility Surface](output/master_accessibility_surface.png)

---

### Key Outputs

| Output | Description |
|---|---|
| Built-up masks (2016, 2025) | Binary rasters, exportable as GeoTIFF to Google Drive |
| Growth heatmap | 4-class change raster with legend |
| Area change report | km² stats: stable, new growth, lost, net, % change |
| Regression plot | Polynomial fit: distance from roads vs. built-up density |
| Master accessibility surface | Fused road + infrastructure heatmap raster |

---

*This notebook is RQ1 of a broader multi-RQ research project on Dholera SIR's urban transformation.*