import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = Path("Rainfall Data 25x25")
OUTPUT_FILE = "rainfallDataBaseline.parquet"


def calculate_max_dry_spell(ds_year):
    """
    Vectorized calculation of max consecutive dry days (< 2.5mm)
    over a 3D grid (Time, Lat, Lon).
    """
    is_dry = ds_year < 2.5

    # Cumulative count of dry days
    cum_dry = is_dry.cumsum(dim="TIME")

    # Identify where the dry spell was broken and forward-fill that value
    # to create a 'reset' baseline
    broken_spells = cum_dry.where(~is_dry)
    reset = broken_spells.ffill(dim="TIME").fillna(0)

    # Subtracting the reset baseline gives the current consecutive streak
    consecutive = cum_dry - reset
    return consecutive.max(dim="TIME")


def get_monsoon_stats(ds_year):
    """
    Vectorized calculation of monsoon duration (10% to 90% cumulative rainfall)
    over a 3D grid.
    """
    total = ds_year.sum(dim="TIME")
    cum_sum = ds_year.cumsum(dim="TIME")

    # argmax(dim='TIME') returns the index of the first True value along the time axis
    start_day = (cum_sum >= 0.1 * total).argmax(dim="TIME")
    end_day = (cum_sum >= 0.9 * total).argmax(dim="TIME")

    duration = end_day - start_day
    # Return 0 where total rainfall is 0 to avoid nonsense numbers in deserts
    return duration.where(total > 0, 0)


def preprocess_rainfall():
    print("🚀 Starting Preprocessing (1991-2025)...")

    file_list = [DATA_DIR / f"RF25_ind{year}_rfp25.nc" for year in range(1991, 2026)]
    valid_files = [str(f) for f in file_list if f.exists()]

    if not valid_files:
        print("❌ No files found! Please check the folder: 'Rainfall Data 25x25'")
        return

    # Load data with 1-year chunks
    ds = xr.open_mfdataset(valid_files, combine="by_coords", chunks={"TIME": 366})
    rainfall = ds.RAINFALL

    print("🧠 Calculating annual and seasonal averages...")
    annual_avg = rainfall.groupby("TIME.year").sum().mean(dim="year")
    p95_rain = rainfall.quantile(0.95, dim="TIME")
    avg_rainy_days = (rainfall > 2.5).groupby("TIME.year").sum().mean(dim="year")
    monthly_avg = rainfall.groupby("TIME.month").mean(dim="TIME") * 30.44

    print("⏳ Calculating sequence-based stats (Max Dry Spell & Monsoon Duration)...")
    # We use .map() here to pass the DataArray chunks to our custom functions
    max_dry = (
        rainfall.groupby("TIME.year").map(calculate_max_dry_spell).mean(dim="year")
    )
    monsoon_dur = rainfall.groupby("TIME.year").map(get_monsoon_stats).mean(dim="year")

    print("🧹 Formatting into final dataframe...")
    df_main = annual_avg.to_dataframe(name="annual_avg_mm").reset_index()

    # Add Monthly Columns
    for m in range(1, 13):
        m_col = f"month_{m}_avg_mm"
        m_data = monthly_avg.sel(month=m).to_dataframe(name=m_col).reset_index()
        df_main = df_main.merge(
            m_data[["LATITUDE", "LONGITUDE", m_col]], on=["LATITUDE", "LONGITUDE"]
        )

    # Adding remaining features
    # (Using the index from annual_avg ensures lat/lon alignment)
    df_main["p95_design_storm_mm"] = p95_rain.values.flatten()
    df_main["avg_rainy_days"] = avg_rainy_days.values.flatten()
    df_main["max_consecutive_dry_days"] = max_dry.values.flatten()
    df_main["monsoon_duration_days"] = monsoon_dur.values.flatten()

    # Save to Parquet
    df_main.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ SUCCESS: Baseline file created: {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess_rainfall()
