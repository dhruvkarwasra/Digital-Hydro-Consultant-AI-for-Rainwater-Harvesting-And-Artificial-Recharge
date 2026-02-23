import xarray as xr
import numpy as np
from pathlib import Path
from scipy.stats import linregress

# --- Configuration ---
DATA_DIR = Path("Rainfall Data 25x25")
OUTPUT_FILE = "rainfallDataSmartFeatures.parquet"

# --- Helper Functions ---


def calculate_slope_vectorized(y_data):
    """
    Calculates the linear trend slope for a time series.
    Designed to be used with xarray.apply_ufunc.
    """
    # Create x-axis (years 0 to N)
    x = np.arange(y_data.shape[-1])

    # If a pixel is all NaNs (ocean) or all zeros, return NaN
    if np.isnan(y_data).all() or np.all(y_data == 0):
        return np.nan

    # Calculate slope (vectorized linear regression is complex, so we loop
    # over the core dimension or use polyfit for speed on small arrays)
    if np.std(y_data) == 0:
        return 0.0
    slope, _, _, _, _ = linregress(x, y_data)
    return slope


def calculate_max_dry_spell(ds_year):
    """
    Vectorized calculation of max consecutive dry days (< 2.5mm).
    """
    is_dry = ds_year < 2.5
    cum_dry = is_dry.cumsum(dim="TIME")
    broken_spells = cum_dry.where(~is_dry)
    reset = broken_spells.ffill(dim="TIME").fillna(0)
    consecutive = cum_dry - reset
    return consecutive.max(dim="TIME")


def get_trend_grid(data_array):
    """
    Wrapper to apply slope calculation safely across Dask chunks.
    FIX: Forces 'year' dimension to be a single chunk to prevent ValueErrors.
    """
    # Force the core dimension 'year' to be contiguous (1 chunk)
    # We keep lat/lon chunked to save memory
    data_array = data_array.chunk({"year": -1})

    return xr.apply_ufunc(
        calculate_slope_vectorized,
        data_array,
        input_core_dims=[["year"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


# --- Main Pipeline ---


def preprocess_rainfall():
    print("🚀 Starting Advanced Feature Extraction (1991-2025)...")

    # 1. Load Data
    file_list = sorted(
        [DATA_DIR / f"RF25_ind{year}_rfp25.nc" for year in range(1991, 2026)]
    )
    valid_files = [str(f) for f in file_list if f.exists()]

    if not valid_files:
        print("❌ No files found! Check path.")
        return

    # Chunk by time to fit in memory
    ds = xr.open_mfdataset(
        valid_files,
        combine="by_coords",
        chunks={"TIME": -1, "LATITUDE": 50, "LONGITUDE": 50},
    )
    rainfall = ds.RAINFALL

    print("🧠 Computing Time-Series Metrics (Yearly Aggregates)...")

    # A. Yearly Totals & Averages
    yearly_sum = rainfall.groupby("TIME.year").sum()
    yearly_p95 = rainfall.groupby("TIME.year").quantile(0.95, dim="TIME")

    # B. Yearly Rainy Days Count (> 2.5mm)
    is_rainy_day = rainfall > 2.5
    yearly_rainy_days = is_rainy_day.groupby("TIME.year").sum()

    # C. Yearly Simple Daily Intensity Index (SDII)
    # Avoid division by zero
    yearly_sdii = yearly_sum / yearly_rainy_days.where(yearly_rainy_days > 0)

    # D. Yearly Max Dry Spells (Complex Sequence)
    # We map the dry spell function over each year
    print("⏳ calculating dry spell sequences (this is heavy)...")
    yearly_max_dry = rainfall.groupby("TIME.year").map(calculate_max_dry_spell)

    # --- Trend Analysis (The "ML" Part) ---
    print("📈 Extracting Trends and Slopes (1991-2025)...")

    # We use the corrected helper function here
    slope_dry_days = get_trend_grid(yearly_max_dry)
    slope_rainy_days = get_trend_grid(yearly_rainy_days)
    slope_sdii = get_trend_grid(yearly_sdii)
    slope_p95 = get_trend_grid(yearly_p95)

    # --- Baseline Averages ---
    print("📊 Computing Final Baselines...")

    # 35-Year Averages
    avg_annual_rain = yearly_sum.mean(dim="year")
    avg_max_dry = yearly_max_dry.mean(dim="year")
    avg_p95_daily = yearly_p95.mean(dim="year")

    # Recent 10-Year Averages (for "Climate Shift" detection)
    recent_years = yearly_sum.sel(year=slice(2016, 2025))
    avg_recent_rain = recent_years.mean(dim="year")

    # Reliability (Coefficient of Variation)
    cv_reliability = yearly_sum.std(dim="year") / avg_annual_rain

    # --- Formatting for Output ---
    print("🧹 Flattening to Table...")

    # Base DataFrame
    df = avg_annual_rain.to_dataframe(name="annual_avg_mm").reset_index()

    # Add Feature Columns (using .values.flatten() to safeguard index alignment)
    # Note: We compute the Dask arrays to numpy before flattening to ensure speed
    print("   -> Fetching computed values...")
    df["recent_10yr_avg_mm"] = avg_recent_rain.compute().values.flatten()
    df["cv_reliability"] = cv_reliability.compute().values.flatten()
    df["avg_max_dry_days"] = avg_max_dry.compute().values.flatten()
    df["p95_daily_mm"] = avg_p95_daily.compute().values.flatten()

    # Trend Features (These were the ones causing issues)
    print("   -> Computing Slopes...")
    df["trend_dry_days_per_year"] = slope_dry_days.compute().values.flatten()
    df["trend_rainy_days_per_year"] = slope_rainy_days.compute().values.flatten()
    df["trend_sdii_intensity"] = slope_sdii.compute().values.flatten()
    df["trend_p95_intensity"] = slope_p95.compute().values.flatten()

    # --- Derived Engineering Features ---
    # 15-min Peak Intensity using 1/3 Power Law: P_15min = P_24hr * (0.25/24)^(1/3)
    power_law_factor = (0.25 / 24) ** (1 / 3)  # ~0.218
    df["design_15min_intensity_mm"] = df["p95_daily_mm"] * power_law_factor

    # Cleanup: Remove ocean pixels (NaNs) to save space
    df = df.dropna(subset=["annual_avg_mm"])

    # Save
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ SUCCESS: ML Feature Store created at {OUTPUT_FILE}")
    print(f"📊 Features extracted: {list(df.columns)}")


if __name__ == "__main__":
    preprocess_rainfall()
