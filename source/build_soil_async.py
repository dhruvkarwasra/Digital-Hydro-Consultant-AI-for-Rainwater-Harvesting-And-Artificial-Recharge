import asyncio
import aiohttp
import pandas as pd
import sys
import random
from pathlib import Path

# --- Configuration ---
INPUT_RAINFALL_FILE = "rainfallDataSmartFeatures.parquet"
OUTPUT_SOIL_FILE = "india_soil_grid_final.csv"

# CONCURRENCY CONTROL
# We process N grids at a time. Each grid launches 5 requests.
# GRID_CONCURRENCY = 4  --> Means 4 * 5 = 20 active connections (Safe for Hotspot)
GRID_CONCURRENCY = 4
SAVE_EVERY_N_GRIDS = 20

# Random User Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
]


def get_sampling_points(lat, lon):
    """Returns 5 points: Center + 4 Corners (approx 12 km offset)"""
    offset = 0.1
    return [
        (lat, lon),  # Center
        (lat + offset, lon + offset),  # NE
        (lat - offset, lon - offset),  # SW
        (lat + offset, lon - offset),  # NW
        (lat - offset, lon + offset),  # SE
    ]


async def fetch_single_probe(session, lat, lon):
    """Fetches data for ONE distinct point."""
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}&property=clay&property=sand&depth=0-5cm&depth=5-15cm&depth=15-30cm"

    retries = 3
    for attempt in range(retries):
        try:
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            async with session.get(url, headers=headers, timeout=20) as response:
                if response.status == 200:
                    data = await response.json()

                    # Check if layers exist (If empty list -> Water/Ice/Rock)
                    if not data.get("properties", {}).get("layers"):
                        return "WATER", 0.0, 0.0

                    soil_values = {}
                    for layer in data["properties"]["layers"]:
                        name = layer["name"]
                        vals = [
                            d["values"]["mean"]
                            for d in layer["depths"]
                            if d["values"]["mean"] is not None
                        ]
                        if vals:
                            # Average the depths & divide by 10 (API returns ints)
                            soil_values[name] = (sum(vals) / len(vals)) / 10.0

                    if "sand" in soil_values and "clay" in soil_values:
                        return "LAND", soil_values["sand"], soil_values["clay"]

                    return "WATER", 0.0, 0.0  # Valid response but missing params

                elif response.status >= 500:
                    await asyncio.sleep(1)  # Server hiccup
                    continue
                else:
                    return "ERROR", None, None  # 404/Client Error

        except Exception:
            await asyncio.sleep(1)  # Network jitter

    return "TIMEOUT", None, None


async def process_grid(session, lat, lon, semaphore):
    """
    Orchestrates the 5 probes for a single grid.
    Returns the Averaged Soil Data.
    """
    async with semaphore:  # Wait for slot
        points = get_sampling_points(lat, lon)

        # Fire all 5 probes concurrently
        tasks = [fetch_single_probe(session, p[0], p[1]) for p in points]
        results = await asyncio.gather(*tasks)

        valid_sand = []
        valid_clay = []
        water_count = 0
        error_count = 0

        for status, s, c in results:
            if status == "LAND":
                valid_sand.append(s)
                valid_clay.append(c)
            elif status == "WATER":
                water_count += 1
            else:
                error_count += 1

        # --- DECISION LOGIC ---

        # 1. If we found ANY soil, we prioritize it (Handles Himalayan Valleys)
        if valid_sand:
            avg_sand = sum(valid_sand) / len(valid_sand)
            avg_clay = sum(valid_clay) / len(valid_clay)
            return lat, lon, avg_sand, avg_clay, "Land"

        # 2. If it was mostly errors (Network fail), we skip to retry later
        if error_count > 2:
            return lat, lon, 0.0, 0.0, "RETRY"

        # 3. If mostly Water/Ice and no soil found -> True Water Body
        return lat, lon, 0.0, 0.0, "Water Body"


def determine_structure(sand, clay):
    if sand > 60:
        return "Recharge Pit", 30.0
    elif clay > 35:
        return "Recharge Shaft", 3.0
    else:
        return "Recharge Trench", 15.0


async def main():
    print("🚀 Starting 5-Point Multi-Probe Scraper...")

    # Load Input
    if not Path(INPUT_RAINFALL_FILE).exists():
        print("❌ Input missing.")
        return
    df = pd.read_parquet(INPUT_RAINFALL_FILE)
    coords = df[["LATITUDE", "LONGITUDE"]].drop_duplicates().values.tolist()

    # Resume Logic
    processed_keys = set()
    if Path(OUTPUT_SOIL_FILE).exists():
        try:
            df_exist = pd.read_csv(OUTPUT_SOIL_FILE)
            for _, r in df_exist.iterrows():
                processed_keys.add(f"{r['LATITUDE']:.4f}_{r['LONGITUDE']:.4f}")
            print(f"🔄 Resuming... {len(processed_keys)} grids done.")
        except:  # noqa: E722
            pass

    todo = [c for c in coords if f"{c[0]:.4f}_{c[1]:.4f}" not in processed_keys]
    print(f"📋 Grids remaining: {len(todo)} (Total Requests: {len(todo) * 5})")

    semaphore = asyncio.Semaphore(GRID_CONCURRENCY)
    buffer = []

    async with aiohttp.ClientSession() as session:
        # Create a task for every grid
        # Note: We create them all, but semaphore limits execution
        tasks = []
        for i, (lat, lon) in enumerate(todo):
            tasks.append(
                asyncio.create_task(process_grid(session, lat, lon, semaphore))
            )

            # Save/Flush every N grids
            if len(tasks) >= SAVE_EVERY_N_GRIDS or i == len(todo) - 1:
                batch_results = await asyncio.gather(*tasks)

                for res in batch_results:
                    lat, lon, sand, clay, label = res

                    if label == "RETRY":
                        print(f"❌ Network Fail: {lat}, {lon}")
                        continue

                    struct, rate = "None", 0.0
                    if label == "Land":
                        struct, rate = determine_structure(sand, clay)
                        print(
                            f"✅ {lat:.2f}, {lon:.2f} | Avg Sand: {sand:.1f}% | Avg Clay: {clay:.1f}%"
                        )
                    else:
                        print(f"🔹 {lat:.2f}, {lon:.2f} | Water/Glacier")

                    buffer.append(
                        {
                            "LATITUDE": lat,
                            "LONGITUDE": lon,
                            "avg_sand_pct": round(sand, 2),
                            "avg_clay_pct": round(clay, 2),
                            "soil_label": label,
                            "infiltration_rate_mm_hr": rate,
                            "recommended_structure": struct,
                        }
                    )

                # Save
                if buffer:
                    pd.DataFrame(buffer).to_csv(
                        OUTPUT_SOIL_FILE,
                        mode="a",
                        header=not Path(OUTPUT_SOIL_FILE).exists(),
                        index=False,
                    )
                    print(
                        f"💾 Saved batch. Total Progress: {len(processed_keys) + len(buffer) + (i - len(buffer))}/{len(coords)}"
                    )
                    buffer = []

                tasks = []  # Clear completed tasks
                await asyncio.sleep(1)  # Cooldown


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
