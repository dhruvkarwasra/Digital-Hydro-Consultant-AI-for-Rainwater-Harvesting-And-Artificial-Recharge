import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# Load your dataset
df = pd.read_parquet("INDIA_HYDROLOGY_FINAL_LABELED.parquet")

# Load India boundary shapefile
india = gpd.read_file(r"india\gadm41_IND_0.shp")
# 1. NEW: High-Contrast "Neon" Color Mapping
zone_colors = {
    "Zone 1: Dual System (High AR + High RWH)": "#00FFAA",  # Vivid Teal/Green
    "Zone 2: Primary Recharge (Favorable Soil/Terrain)": "#00AAFF",  # Bright Blue
    "Zone 3: Surface Storage Priority (Impermeable Soil)": "#FFBB00",  # Sharp Amber
    "Zone 4: Storage Priority (Hilly/Rugged Terrain - AR Unsafe)": "#FF3333",  # High-Alert Red
    "Zone 5: Mixed/Moderate Potential": "#AAAAAA",  # Solid Light Gray
}

# 2. NEW: Setup Dark Background
bg_color = "#121212"
fig, ax = plt.subplots(figsize=(10, 11))
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# Make the country border slightly darker and thicker to frame the bright data
india.plot(ax=ax, color="none", edgecolor="#444444", linewidth=1.0)

for zone, group in df.groupby("EXPERT_ZONE"):
    ax.scatter(
        group["LONGITUDE"],
        group["LATITUDE"],
        c=zone_colors[zone],
        s=15,  # Slightly smaller points to prevent them from blurring together
        label=zone,
        alpha=1.0,  # REMOVED TRANSPARENCY for maximum punch
        linewidths=0,
    )

# 3. NEW: Style the Legend and Title for Dark Mode
legend = ax.legend(
    loc="lower left",
    fontsize=9,
    title="Suitability Zone",
    facecolor="#1E1E1E",  # Dark gray legend box
    edgecolor="#444444",
    labelcolor="white",
)
legend.get_title().set_color("white")
legend.get_title().set_weight("bold")

ax.set_title(
    "Fig. 2 — Spatial Distribution of Suitability Zones\n"
    "INDIA_HYDROLOGY_FINAL_LABELED (~4,936 grid cells)",
    fontsize=12,
    color="white",
    weight="bold",
    pad=15,
)

ax.axis("off")
plt.tight_layout()

# 4. NEW: Pass the facecolor to savefig so the dark background exports correctly
plt.savefig(
    "zone_distribution_map_high_contrast.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=bg_color,
)
print(
    "✅ High-contrast map successfully saved to zone_distribution_map_high_contrast.png"
)
