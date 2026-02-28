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
    Returns 0.0 if data is flat or invalid.
    """
    # Create x-axis (years 0 to N)
    x = np.arange(y_data.shape[-1])

    # Safety Check: If a pixel is all NaNs (ocean) or all zeros
    if np.isnan(y_data).all() or np.all(y_data == 0):
        return np.nan

    # If variance is 0 (e.g., all values are the same), slope is 0
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
    CRITICAL FIX: Forces 'year' dimension to be a single chunk.
    """
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
    print("🚀 Starting Final Smart Feature Extraction (1991-2025)...")

    file_list = sorted(
        [DATA_DIR / f"RF25_ind{year}_rfp25.nc" for year in range(1991, 2026)]
    )
    valid_files = [str(f) for f in file_list if f.exists()]

    if not valid_files:
        print("❌ No files found! Check path.")
        return

    # Load Data (Chunked spatially to save RAM)
    ds = xr.open_mfdataset(
        valid_files,
        combine="by_coords",
        chunks={"TIME": -1, "LATITUDE": 50, "LONGITUDE": 50},
    )
    rainfall = ds.RAINFALL

    print("🧠 Computing Time-Series Metrics...")

    # 1. Yearly Aggregates
    yearly_sum = rainfall.groupby("TIME.year").sum()

    # P95 (For Filter Efficiency Design)
    yearly_p95 = rainfall.groupby("TIME.year").quantile(0.95, dim="TIME")

    # Peak Daily (For Overflow Safety Design) - NEW CRITICAL FEATURE
    yearly_peak = rainfall.groupby("TIME.year").max(dim="TIME")

    # Rainy Days (> 2.5mm)
    is_rainy_day = rainfall > 2.5
    yearly_rainy_days = is_rainy_day.groupby("TIME.year").sum()

    # Simple Daily Intensity Index (SDII)
    # FIX: We fillna(0) to prevent NaN trends in dry years
    yearly_sdii = (yearly_sum / yearly_rainy_days.where(yearly_rainy_days > 0)).fillna(
        0
    )

    # Max Dry Spells
    print("⏳ Calculating dry spell sequences...")
    yearly_max_dry = rainfall.groupby("TIME.year").map(calculate_max_dry_spell)

    # 2. Trend Analysis
    print("📈 Extracting Trends (Slopes)...")
    slope_dry_days = get_trend_grid(yearly_max_dry)
    slope_rainy_days = get_trend_grid(yearly_rainy_days)
    slope_sdii = get_trend_grid(yearly_sdii)
    slope_p95 = get_trend_grid(yearly_p95)
    slope_peak = get_trend_grid(yearly_peak)  # Trend of extreme storms

    # 3. Computing Baselines (Averages)
    print("📊 Computing Final Baselines...")
    avg_annual_rain = yearly_sum.mean(dim="year")
    avg_max_dry = yearly_max_dry.mean(dim="year")
    avg_p95_daily = yearly_p95.mean(dim="year")
    avg_peak_daily = yearly_peak.mean(dim="year")  # Average "Worst Day of the Year"

    # Recent 10-Year Averages
    recent_years = yearly_sum.sel(year=slice(2016, 2025))
    avg_recent_rain = recent_years.mean(dim="year")

    # Reliability (CV)
    cv_reliability = yearly_sum.std(dim="year") / avg_annual_rain

    # 4. Flattening to Table
    print("🧹 Creating Final Table...")

    # We create the DataFrame and explicitly compute() Dask arrays to prevent lazy loading errors
    df = avg_annual_rain.to_dataframe(name="annual_avg_mm").reset_index()

    # Helper to add columns safely
    def add_col(name, dask_array):
        df[name] = dask_array.compute().values.flatten()

    add_col("recent_10yr_avg_mm", avg_recent_rain)
    add_col("cv_reliability", cv_reliability)
    add_col("avg_max_dry_days", avg_max_dry)
    add_col("p95_daily_mm", avg_p95_daily)
    add_col("peak_daily_mm", avg_peak_daily)  # The critical metric for overflow

    # Trends
    add_col("trend_dry_days_per_year", slope_dry_days)
    add_col("trend_rainy_days_per_year", slope_rainy_days)
    add_col("trend_sdii_intensity", slope_sdii)
    add_col("trend_p95_intensity", slope_p95)
    add_col("trend_peak_intensity", slope_peak)  # Is the "worst day" getting worse?

    # 5. Derived Engineering Features (Power Law Approximation)
    # Factor = (0.25 hours / 24 hours) ^ (1/3) ≈ 0.218
    power_law_factor = 0.218

    # For Filter Sizing (Efficiency): Uses P95 (Standard heavy rain)
    df["design_15min_filter_intensity_mm"] = df["p95_daily_mm"] * power_law_factor

    # For Overflow Sizing (Safety): Uses Peak Daily (Extreme rain)
    # This roughly matches the CPWD "Peak Intensity" requirements
    df["design_15min_overflow_intensity_mm"] = df["peak_daily_mm"] * power_law_factor

    # Cleanup
    df = df.dropna(subset=["annual_avg_mm"])

    # Save
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ SUCCESS: Final ML Feature Store created at {OUTPUT_FILE}")
    print(f"📊 Features extracted: {len(df.columns)} columns")


if __name__ == "__main__":
    preprocess_rainfall()
