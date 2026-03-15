import pandas as pd

# Configuration
RAINFALL_FILE = "rainfallDataSmartFeatures.parquet"
SOIL_FILE = "india_soil_grid_final.csv"
OUTPUT_FILE = "final_capstone_dataset.parquet"

print("⏳ Loading datasets...")
# Load Rainfall (Time Series)
df_rain = pd.read_parquet(RAINFALL_FILE)
print(f"   Rainfall Rows: {len(df_rain)}")

# Load Soil (Static)
df_soil = pd.read_csv(SOIL_FILE)
print(f"   Soil Grids:    {len(df_soil)}")

# 1. Round Coordinates to ensure perfect matching
# (Floats can be tricky: 23.5000001 != 23.5)
print("⚙️ Aligning coordinates...")
df_rain["LATITUDE"] = df_rain["LATITUDE"].round(2)
df_rain["LONGITUDE"] = df_rain["LONGITUDE"].round(2)
df_soil["LATITUDE"] = df_soil["LATITUDE"].round(2)
df_soil["LONGITUDE"] = df_soil["LONGITUDE"].round(2)

# 2. Merge (Left Join on Lat/Lon)
print("🔗 Merging...")
df_final = pd.merge(df_rain, df_soil, on=["LATITUDE", "LONGITUDE"], how="left")

# 3. Handle Water Bodies / Missing Data
# If a grid was 'Water Body', structure is NaN. We fill it with "None".
values_to_fill = {
    "soil_label": "Land",  # Default assumption
    "recommended_structure": "None",  # No structure for water bodies
    "avg_sand_pct": 0.0,
    "avg_clay_pct": 0.0,
    "infiltration_rate_mm_hr": 0.0,
}
df_final.fillna(value=values_to_fill, inplace=True)

# 4. Save
print(f"💾 Saving to {OUTPUT_FILE}...")
df_final.to_parquet(OUTPUT_FILE, index=False)

print("\n🎉 SUCCESS! Dataset is ML-ready.")
print(f"Total Rows: {len(df_final)}")
print(f"Columns: {list(df_final.columns)}")
