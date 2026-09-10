from pathlib import Path
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_LANDSAT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landsat"
)

HEAT_POCKET_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "heat_pockets"
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
    / "processed"
    / "features"
)

OUTPUT_CSV = OUTPUT_DIR / "ward_temporal_ml_dataset.csv"

TARGET_CRS = "EPSG:32643"

PIXEL_AREA_HA = 0.09


# ============================================================
# REQUIRED FILES
# ============================================================

REQUIRED_FEATURES = [
    "lst_celsius.tif",
    "ndvi.tif",
    "ndbi.tif",
    "ndwi.tif",
]


# ============================================================
# LOAD WARDS
# ============================================================

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

    if len(wards) != 369:
        raise ValueError(
            f"Expected 369 wards, found {len(wards)}."
        )

    # Ensure unique ward identifier.
    wards["ward_id"] = wards["ward_id"].astype(str)
    wards["Corporation"] = wards["Corporation"].astype(str)

    wards["ward_key"] = (
        wards["Corporation"].str.strip()
        + "_"
        + wards["ward_id"].str.strip()
    )

    if wards["ward_key"].nunique() != 369:
        raise ValueError(
            "Ward keys are not unique."
        )

    return wards


# ============================================================
# SCENE DISCOVERY
# ============================================================

def find_scene_dates():
    """Find all processed Landsat scene dates."""

    if not PROCESSED_LANDSAT_DIR.exists():
        raise FileNotFoundError(
            f"Processed Landsat directory not found:\n"
            f"{PROCESSED_LANDSAT_DIR}"
        )

    scene_dirs = sorted(
        [
            p
            for p in PROCESSED_LANDSAT_DIR.iterdir()
            if p.is_dir()
        ]
    )

    scene_dates = []

    for scene_dir in scene_dirs:

        try:
            datetime.strptime(
                scene_dir.name,
                "%Y-%m-%d"
            )
        except ValueError:
            continue

        missing = [
            feature
            for feature in REQUIRED_FEATURES
            if not (scene_dir / feature).exists()
        ]

        if missing:
            raise ValueError(
                f"Scene {scene_dir.name} is missing: {missing}"
            )

        scene_dates.append(scene_dir.name)

    if len(scene_dates) != 76:
        raise ValueError(
            f"Expected 76 scenes, found {len(scene_dates)}."
        )

    return sorted(scene_dates)


# ============================================================
# RASTER READING
# ============================================================

def read_raster(path):
    """Read a raster as float32."""

    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs

    return array, transform, crs


# ============================================================
# WARD STATISTICS
# ============================================================

def calculate_scene_ward_features(
    scene_date,
    wards,
):
    """
    Calculate ward-level statistics for one observation.

    Returns one row per ward.
    """

    scene_dir = PROCESSED_LANDSAT_DIR / scene_date

    lst, transform, crs = read_raster(
        scene_dir / "lst_celsius.tif"
    )

    ndvi, _, _ = read_raster(
        scene_dir / "ndvi.tif"
    )

    ndbi, _, _ = read_raster(
        scene_dir / "ndbi.tif"
    )

    ndwi, _, _ = read_raster(
        scene_dir / "ndwi.tif"
    )

    # Validate raster consistency.
    if crs.to_string() != TARGET_CRS:
        raise ValueError(
            f"{scene_date}: unexpected CRS {crs}"
        )

    if not (
        lst.shape
        == ndvi.shape
        == ndbi.shape
        == ndwi.shape
    ):
        raise ValueError(
            f"{scene_date}: raster shapes do not match."
        )

    # Heat-pocket raster.
    heat_pocket_path = (
        HEAT_POCKET_DIR
        / scene_date
        / "heat_pockets.tif"
    )

    if not heat_pocket_path.exists():
        raise FileNotFoundError(
            f"Heat-pocket raster missing:\n"
            f"{heat_pocket_path}"
        )

    heat_pockets, heat_transform, heat_crs = (
        read_raster(heat_pocket_path)
    )

    if heat_crs.to_string() != TARGET_CRS:
        raise ValueError(
            f"{scene_date}: heat-pocket CRS mismatch."
        )

    if (
        heat_pockets.shape != lst.shape
        or heat_transform != transform
    ):
        raise ValueError(
            f"{scene_date}: heat-pocket raster "
            f"is not aligned with LST raster."
        )

    results = []

    for _, ward in wards.iterrows():

        mask = geometry_mask(
            [ward.geometry],
            transform=transform,
            invert=True,
            out_shape=lst.shape,
            all_touched=False,
        )

        # --------------------------------------------
        # Valid LST pixels
        # --------------------------------------------

        valid_lst = mask & np.isfinite(lst)

        lst_values = lst[valid_lst]

        if len(lst_values) == 0:
            continue

        # --------------------------------------------
        # Spectral features
        # --------------------------------------------

        ndvi_values = ndvi[valid_lst]
        ndbi_values = ndbi[valid_lst]
        ndwi_values = ndwi[valid_lst]

        ndvi_values = ndvi_values[np.isfinite(ndvi_values)]
        ndbi_values = ndbi_values[np.isfinite(ndbi_values)]
        ndwi_values = ndwi_values[np.isfinite(ndwi_values)]

        # --------------------------------------------
        # Heat-pocket features
        # --------------------------------------------

        heat_values = heat_pockets[valid_lst]

        hot_pixels = np.sum(heat_values > 0)

        heat_pocket_area_ha = (
            hot_pixels * PIXEL_AREA_HA
        )

        heat_pocket_coverage_pct = (
            hot_pixels
            / len(lst_values)
            * 100
        )

        # --------------------------------------------
        # Result
        # --------------------------------------------

        result = {
            "date": scene_date,

            "ward_key": ward["ward_key"],
            "Corporation": ward["Corporation"],
            "ward_id": ward["ward_id"],
            "ward_name": (
                ward["ward_name"]
                if "ward_name" in wards.columns
                else ""
            ),

            "valid_pixels": int(len(lst_values)),

            "mean_lst_celsius":
                float(np.mean(lst_values)),

            "median_lst_celsius":
                float(np.median(lst_values)),

            "std_lst_celsius":
                float(np.std(lst_values)),

            "mean_ndvi":
                float(np.mean(ndvi_values))
                if len(ndvi_values) > 0
                else np.nan,

            "mean_ndbi":
                float(np.mean(ndbi_values))
                if len(ndbi_values) > 0
                else np.nan,

            "mean_ndwi":
                float(np.mean(ndwi_values))
                if len(ndwi_values) > 0
                else np.nan,

            "hot_pixels":
                int(hot_pixels),

            "heat_pocket_area_ha":
                float(heat_pocket_area_ha),

            "heat_pocket_coverage_pct":
                float(heat_pocket_coverage_pct),
        }

        results.append(result)

    return pd.DataFrame(results)


# ============================================================
# BUILD CURRENT OBSERVATION DATASET
# ============================================================

def build_current_observation_dataset(
    scene_dates,
    wards,
):
    """Build ward-level features for all observations."""

    all_frames = []

    print()
    print("Building ward-level observation dataset...")
    print()

    for i, scene_date in enumerate(scene_dates, start=1):

        print(
            f"[{i:02d}/{len(scene_dates)}] "
            f"Processing {scene_date}"
        )

        scene_df = calculate_scene_ward_features(
            scene_date,
            wards,
        )

        all_frames.append(scene_df)

    df = pd.concat(
        all_frames,
        ignore_index=True,
    )

    return df


# ============================================================
# TEMPORAL FEATURES
# ============================================================

def add_temporal_features(df):
    """
    Add lagged and future temporal information.

    All rows are sorted chronologically within each ward.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["ward_key", "date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Previous observation features
    # --------------------------------------------------------

    df["previous_lst_celsius"] = (
        df.groupby("ward_key")[
            "mean_lst_celsius"
        ].shift(1)
    )

    df["previous_heat_pocket_coverage_pct"] = (
        df.groupby("ward_key")[
            "heat_pocket_coverage_pct"
        ].shift(1)
    )

    # --------------------------------------------------------
    # Time since previous observation
    # --------------------------------------------------------

    previous_date = (
        df.groupby("ward_key")["date"].shift(1)
    )

    df["days_since_previous_observation"] = (
        df["date"] - previous_date
    ).dt.days

    # --------------------------------------------------------
    # Next observation target
    # --------------------------------------------------------

    df["next_observation_lst_celsius"] = (
        df.groupby("ward_key")[
            "mean_lst_celsius"
        ].shift(-1)
    )

    next_date = (
        df.groupby("ward_key")["date"].shift(-1)
    )

    df["next_observation_date"] = next_date

    df["days_to_next_observation"] = (
        next_date - df["date"]
    ).dt.days

    return df


# ============================================================
# HISTORICAL HEAT-POCKET PERSISTENCE
# ============================================================

def add_historical_persistence(df):
    """
    Calculate cumulative historical heat-pocket frequency
    using only observations BEFORE the current observation.

    This prevents future information from entering the model.
    """

    df = df.copy()

    df = df.sort_values(
        ["ward_key", "date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Previous heat-pocket observations
    # --------------------------------------------------------

    previous_hot_coverage = (
        df.groupby("ward_key")[
            "heat_pocket_coverage_pct"
        ].shift(1)
    )

    # --------------------------------------------------------
    # Cumulative historical mean
    #
    # This uses only rows before the current observation.
    # --------------------------------------------------------

    cumulative_sum = (
        df.groupby("ward_key")[
            "heat_pocket_coverage_pct"
        ]
        .transform(
            lambda x:
            x.shift(1).expanding().sum()
        )
    )

    cumulative_count = (
        df.groupby("ward_key")[
            "heat_pocket_coverage_pct"
        ]
        .transform(
            lambda x:
            x.shift(1).expanding().count()
        )
    )

    df["historical_heat_pocket_coverage_pct"] = (
        cumulative_sum / cumulative_count
    )

    # --------------------------------------------------------
    # Historical maximum
    # --------------------------------------------------------

    df["historical_max_heat_pocket_coverage_pct"] = (
        df.groupby("ward_key")[
            "heat_pocket_coverage_pct"
        ]
        .transform(
            lambda x:
            x.shift(1).expanding().max()
        )
    )

    # --------------------------------------------------------
    # Historical observation count
    # --------------------------------------------------------

    df["historical_observation_count"] = (
        df.groupby("ward_key").cumcount()
    )

    # Avoid unused variable warnings / keep explicit meaning.
    df["previous_heat_pocket_coverage_pct"] = (
        previous_hot_coverage
    )

    return df


# ============================================================
# FINAL DATASET
# ============================================================

def prepare_final_dataset(df):
    """
    Prepare the final supervised-learning dataset.

    The first observation has no historical features.
    The last observation has no next-observation target.
    """

    df = df.copy()

    # We need:
    # - previous observation
    # - historical persistence
    # - next observation target

    df = df[
        df["previous_lst_celsius"].notna()
        & df[
            "next_observation_lst_celsius"
        ].notna()
        & df[
            "days_to_next_observation"
        ].notna()
    ].copy()

    # Remove impossible temporal gaps.
    df = df[
        df["days_to_next_observation"] > 0
    ].copy()

    df = df.sort_values(
        ["date", "ward_key"]
    ).reset_index(drop=True)

    return df


# ============================================================
# SAVE
# ============================================================

def save_dataset(df):
    """Save the ML-ready dataset."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print("Saved:")
    print(f"  {OUTPUT_CSV}")


# ============================================================
# MAIN
# ============================================================

def run_feature_engineering():

    print("=" * 70)
    print("WARD-LEVEL TEMPORAL FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load wards
    # --------------------------------------------------------

    wards = load_wards()

    # --------------------------------------------------------
    # Find scenes
    # --------------------------------------------------------

    scene_dates = find_scene_dates()

    print()
    print(f"Scenes found: {len(scene_dates)}")
    print(
        f"First observation: {scene_dates[0]}"
    )
    print(
        f"Last observation: {scene_dates[-1]}"
    )

    # --------------------------------------------------------
    # Current-observation features
    # --------------------------------------------------------

    df = build_current_observation_dataset(
        scene_dates,
        wards,
    )

    print()
    print(
        f"Current observation rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    print()
    print("Adding temporal features...")

    df = add_temporal_features(df)

    # --------------------------------------------------------
    # Historical persistence
    # --------------------------------------------------------

    print(
        "Adding historical persistence features..."
    )

    df = add_historical_persistence(df)

    # --------------------------------------------------------
    # Final supervised dataset
    # --------------------------------------------------------

    df = prepare_final_dataset(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(df)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print()
    print(f"Final ML rows: {len(df):,}")
    print(
        f"Unique wards: {df['ward_key'].nunique()}"
    )
    print(
        f"Unique observations: {df['date'].nunique()}"
    )

    print()
    print("Date range:")
    print(
        f"  {df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    print()
    print("Columns:")
    for column in df.columns:
        print(f"  - {column}")

    print()
    print("Target summary:")
    print(
        df[
            "next_observation_lst_celsius"
        ].describe()
    )

    print()
    print("DONE.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_feature_engineering()