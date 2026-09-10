from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PERSISTENCE_RASTER = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "persistence"
    / "heat_pocket_persistence_percentage.tif"
)

BOUNDARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "boundaries"
    / "gba_wards_369.geojson"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "persistence"
)

OUTPUT_CSV = OUTPUT_DIR / "ward_persistence_summary.csv"

TARGET_CRS = "EPSG:32643"

PIXEL_AREA_HA = 0.09

PERSISTENCE_THRESHOLDS = [20, 50, 60]


# ============================================================
# LOAD DATA
# ============================================================

def load_persistence_raster():
    """Load the pixel-level persistence percentage raster."""

    if not PERSISTENCE_RASTER.exists():
        raise FileNotFoundError(
            f"Persistence raster not found:\n{PERSISTENCE_RASTER}"
        )

    src = rasterio.open(PERSISTENCE_RASTER)

    persistence = src.read(1).astype(np.float32)

    return src, persistence


def load_wards():
    """Load the 369 Bengaluru ward boundaries."""

    if not BOUNDARY_PATH.exists():
        raise FileNotFoundError(
            f"Ward boundary file not found:\n{BOUNDARY_PATH}"
        )

    wards = gpd.read_file(BOUNDARY_PATH)

    print(f"Wards loaded: {len(wards)}")
    print(f"Original CRS: {wards.crs}")

    if wards.crs is None:
        raise ValueError("Ward boundary CRS is missing.")

    wards = wards.to_crs(TARGET_CRS)

    print(f"Projected CRS: {wards.crs}")

    return wards


# ============================================================
# WARD IDENTIFIER
# ============================================================

def create_ward_identifier(wards):
    """
    Create a unique ward identifier.

    Ward IDs may repeat across corporations, so both
    Corporation and ward_id are used.
    """

    if "Corporation" not in wards.columns:
        raise ValueError("Expected 'Corporation' column not found.")

    if "ward_id" not in wards.columns:
        raise ValueError("Expected 'ward_id' column not found.")

    wards["ward_id"] = wards["ward_id"].astype(str)
    wards["Corporation"] = wards["Corporation"].astype(str)

    wards["ward_key"] = (
        wards["Corporation"].str.strip()
        + "_"
        + wards["ward_id"].str.strip()
    )

    return wards


# ============================================================
# PIXEL CALCULATIONS
# ============================================================

def calculate_ward_statistics(wards, persistence, transform):
    """
    Calculate persistence statistics for every ward.
    """

    results = []

    total_raster_pixels = persistence.size

    print(f"Raster pixels: {total_raster_pixels:,}")
    print()

    for index, ward in wards.iterrows():

        geometry = [ward.geometry]

        # Pixels whose centers fall inside the ward.
        mask = geometry_mask(
            geometry,
            transform=transform,
            invert=True,
            out_shape=persistence.shape,
            all_touched=False,
        )

        values = persistence[mask]

        # Remove nodata / NaN values.
        values = values[np.isfinite(values)]

        if len(values) == 0:
            print(
                f"WARNING: No valid raster pixels for "
                f"{ward['ward_key']}"
            )
            continue

        ward_pixel_count = len(values)

        # ----------------------------------------------------
        # Basic persistence statistics
        # ----------------------------------------------------

        mean_persistence = float(np.mean(values))
        median_persistence = float(np.median(values))
        max_persistence = float(np.max(values))

        # ----------------------------------------------------
        # Threshold-based statistics
        # ----------------------------------------------------

        threshold_stats = {}

        for threshold in PERSISTENCE_THRESHOLDS:

            persistent_pixels = np.sum(values >= threshold)

            persistent_area_ha = (
                persistent_pixels * PIXEL_AREA_HA
            )

            persistent_percentage_of_ward = (
                persistent_pixels / ward_pixel_count * 100
            )

            threshold_stats[threshold] = {
                "pixels": int(persistent_pixels),
                "area_ha": float(persistent_area_ha),
                "percentage_of_ward": float(
                    persistent_percentage_of_ward
                ),
            }

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "ward_key": ward["ward_key"],
            "Corporation": ward["Corporation"],
            "ward_id": ward["ward_id"],
            "ward_name": (
                ward["ward_name"]
                if "ward_name" in wards.columns
                else ""
            ),

            "valid_raster_pixels": ward_pixel_count,

            "mean_persistence_pct": mean_persistence,
            "median_persistence_pct": median_persistence,
            "max_persistence_pct": max_persistence,

            "persistent_pixels_ge_20pct":
                threshold_stats[20]["pixels"],

            "persistent_area_ge_20pct_ha":
                threshold_stats[20]["area_ha"],

            "persistent_coverage_ge_20pct":
                threshold_stats[20]["percentage_of_ward"],

            "persistent_pixels_ge_50pct":
                threshold_stats[50]["pixels"],

            "persistent_area_ge_50pct_ha":
                threshold_stats[50]["area_ha"],

            "persistent_coverage_ge_50pct":
                threshold_stats[50]["percentage_of_ward"],

            "persistent_pixels_ge_60pct":
                threshold_stats[60]["pixels"],

            "persistent_area_ge_60pct_ha":
                threshold_stats[60]["area_ha"],

            "persistent_coverage_ge_60pct":
                threshold_stats[60]["percentage_of_ward"],
        }

        results.append(result)

    return pd.DataFrame(results)


# ============================================================
# RANKINGS
# ============================================================

def add_rankings(df):
    """Add ward rankings based on persistent heat."""

    df = df.copy()

    df["rank_ge_20pct_area"] = (
        df["persistent_area_ge_20pct_ha"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    df["rank_ge_50pct_area"] = (
        df["persistent_area_ge_50pct_ha"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    df["rank_ge_60pct_area"] = (
        df["persistent_area_ge_60pct_ha"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    df = df.sort_values(
        by="persistent_area_ge_50pct_ha",
        ascending=False
    ).reset_index(drop=True)

    return df


# ============================================================
# SAVE
# ============================================================

def save_results(df):
    """Save ward-level persistence statistics."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)

    print()
    print("Saved:")
    print(f"  {OUTPUT_CSV}")


# ============================================================
# MAIN
# ============================================================

def run_ward_persistence_analysis():

    print("=" * 70)
    print("WARD-LEVEL PERSISTENCE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load raster
    # --------------------------------------------------------

    print()
    print("Loading persistence raster...")

    src, persistence = load_persistence_raster()

    print(f"Raster CRS: {src.crs}")
    print(f"Raster size: {src.width} x {src.height}")
    print(f"Resolution: {src.res}")

    if src.crs.to_string() != TARGET_CRS:
        raise ValueError(
            f"Unexpected raster CRS: {src.crs}. "
            f"Expected {TARGET_CRS}."
        )

    # --------------------------------------------------------
    # Load wards
    # --------------------------------------------------------

    print()
    print("Loading ward boundaries...")

    wards = load_wards()

    wards = create_ward_identifier(wards)

    # --------------------------------------------------------
    # Validate ward count
    # --------------------------------------------------------

    if len(wards) != 369:
        raise ValueError(
            f"Expected 369 wards, but found {len(wards)}."
        )

    if wards["ward_key"].nunique() != 369:
        raise ValueError(
            "Ward identifiers are not unique."
        )

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    print()
    print("Calculating ward-level persistence statistics...")
    print()

    df = calculate_ward_statistics(
        wards,
        persistence,
        src.transform,
    )

    # --------------------------------------------------------
    # Rankings
    # --------------------------------------------------------

    df = add_rankings(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(df)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("WARD-LEVEL PERSISTENCE ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(f"Wards processed: {len(df)}")

    print()
    print("Top 10 wards by >=50% persistence area:")

    columns_to_show = [
        "ward_name",
        "Corporation",
        "ward_id",
        "persistent_area_ge_50pct_ha",
        "persistent_coverage_ge_50pct",
        "mean_persistence_pct",
        "max_persistence_pct",
    ]

    print(
        df[columns_to_show]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("DONE.")


if __name__ == "__main__":
    run_ward_persistence_analysis()