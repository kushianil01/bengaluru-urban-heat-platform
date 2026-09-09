from pathlib import Path
import sys
import time

import pandas as pd
import requests
from pystac_client import Client
import planetary_computer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

INVENTORY_FILE = OUTPUT_DIR / "landsat_inventory.csv"
MONTHLY_SELECTION_FILE = OUTPUT_DIR / "landsat_monthly_selection.csv"

LANDSAT_RAW_DIR = OUTPUT_DIR / "landsat"


# ============================================================
# STUDY AREA
# ============================================================

# Bengaluru study-area bounding box
# [west, south, east, north]

BBOX = [
    77.35,
    12.75,
    77.85,
    13.15,
]


# ============================================================
# STUDY PERIOD
# ============================================================

START_DATE = "2016-01-01T00:00:00Z"
END_DATE = "2026-09-02T23:59:59Z"


# ============================================================
# LANDSAT CONFIGURATION
# ============================================================

LANDSAT_COLLECTION = "landsat-c2-l2"

LANDSAT_PATH = 144
LANDSAT_ROW = 51

MAX_CLOUD_COVER = 20

SATELLITES = {
    "landsat-8": "Landsat 8",
    "landsat-9": "Landsat 9",
}


# ============================================================
# DOWNLOAD CONFIGURATION
# ============================================================

# Actual Planetary Computer STAC asset names.
#
# red      -> Landsat red band
# green    -> Landsat green band
# nir08    -> Landsat 8/9 near-infrared band
# swir16   -> Landsat 8/9 SWIR1 band
# swir22   -> Landsat 8/9 SWIR2 band
# lwir11   -> Landsat thermal band
# qa_pixel -> Pixel quality / cloud / shadow information
# mtl.txt  -> Landsat metadata

REQUIRED_ASSETS = [
    "red",
    "green",
    "nir08",
    "swir16",
    "swir22",
    "lwir11",
    "qa_pixel",
    "mtl.txt",
]

DOWNLOAD_TIMEOUT = 120
DOWNLOAD_RETRIES = 3
RETRY_WAIT_SECONDS = 5


# ============================================================
# PLANETARY COMPUTER
# ============================================================

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_satellite(item):
    """
    Determine whether a scene belongs to Landsat 8 or Landsat 9.
    """

    platform = item.properties.get("platform", "")
    platform = str(platform).lower()

    if "landsat-9" in platform:
        return "Landsat 9"

    if "landsat-8" in platform:
        return "Landsat 8"

    # Fallback based on item ID
    item_id = item.id.lower()

    if item_id.startswith("lc09"):
        return "Landsat 9"

    if item_id.startswith("lc08"):
        return "Landsat 8"

    return "Unknown"


def get_cloud_cover(item):
    """
    Extract cloud cover percentage from STAC metadata.
    """

    value = item.properties.get("eo:cloud_cover")

    if value is None:
        value = item.properties.get("landsat:cloud_cover_land")

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_path_row(item):
    """
    Extract Landsat WRS path and row.

    STAC metadata may return these values as strings or integers,
    so they are normalized to integers.
    """

    path = item.properties.get("landsat:wrs_path")
    row = item.properties.get("landsat:wrs_row")

    try:
        path = int(path)
    except (TypeError, ValueError):
        path = None

    try:
        row = int(row)
    except (TypeError, ValueError):
        row = None

    return path, row


def get_acquisition_date(item):
    """
    Extract acquisition datetime and return date only.
    """

    datetime_value = item.properties.get("datetime")

    if datetime_value is None:
        return None

    return pd.to_datetime(
        datetime_value,
        utc=True
    ).strftime("%Y-%m-%d")


# ============================================================
# SEARCH LANDSAT
# ============================================================

def search_landsat():

    print("=" * 70)
    print("BENGALURU LANDSAT INVENTORY")
    print("=" * 70)

    print(f"Study period : {START_DATE[:10]} → {END_DATE[:10]}")
    print(f"WRS path     : {LANDSAT_PATH}")
    print(f"WRS row      : {LANDSAT_ROW}")
    print(f"Cloud limit  : {MAX_CLOUD_COVER}%")
    print(f"BBOX         : {BBOX}")
    print()

    print("Connecting to Microsoft Planetary Computer...")

    catalog = Client.open(STAC_URL)

    print("Connected successfully.")
    print()

    print("Searching Landsat Collection 2 Level-2 scenes...")

    search = catalog.search(
        collections=[LANDSAT_COLLECTION],
        bbox=BBOX,
        datetime=f"{START_DATE}/{END_DATE}",
        max_items=10000,
    )

    items = list(search.items())

    print(f"Scenes returned by catalog: {len(items)}")
    print()

    records = []

    # ========================================================
    # FILTER SCENES
    # ========================================================

    for item in items:

        path, row = get_path_row(item)

        # ----------------------------------------------------
        # Filter WRS path / row
        # ----------------------------------------------------

        if path != LANDSAT_PATH or row != LANDSAT_ROW:
            continue

        # ----------------------------------------------------
        # Filter satellite
        # ----------------------------------------------------

        satellite = get_satellite(item)

        if satellite not in SATELLITES.values():
            continue

        # ----------------------------------------------------
        # Filter cloud cover
        # ----------------------------------------------------

        cloud_cover = get_cloud_cover(item)

        if cloud_cover is None:
            continue

        if cloud_cover > MAX_CLOUD_COVER:
            continue

        # ----------------------------------------------------
        # Acquisition date
        # ----------------------------------------------------

        acquisition_date = get_acquisition_date(item)

        if acquisition_date is None:
            continue

        # ----------------------------------------------------
        # Store scene
        # ----------------------------------------------------

        records.append(
            {
                "entity_id": item.id,
                "display_id": item.id,
                "acquisition_date": acquisition_date,
                "year": int(acquisition_date[:4]),
                "month": int(acquisition_date[5:7]),
                "month_label": acquisition_date[:7],
                "satellite": satellite,
                "cloud_cover": cloud_cover,
                "wrs_path": path,
                "wrs_row": row,
            }
        )

    # ========================================================
    # CREATE INVENTORY DATAFRAME
    # ========================================================

    df = pd.DataFrame(records)

    if df.empty:

        print("No matching Landsat scenes were found.")

        return

    # Remove duplicate scene IDs
    df = df.drop_duplicates(
        subset="entity_id"
    )

    # Sort chronologically and by cloud cover
    df = df.sort_values(
        by=[
            "acquisition_date",
            "cloud_cover",
            "satellite",
        ]
    ).reset_index(drop=True)

    # ========================================================
    # SAVE FULL INVENTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        INVENTORY_FILE,
        index=False
    )

    # ========================================================
    # MONTHLY SCENE SELECTION
    # ========================================================

    print("=" * 70)
    print("SELECTING ONE BEST SCENE PER MONTH")
    print("=" * 70)

    # Sort so the first scene within each month has:
    # 1. Lowest cloud cover
    # 2. Earliest acquisition date
    # 3. Stable satellite ordering

    monthly = df.sort_values(
        by=[
            "year",
            "month",
            "cloud_cover",
            "acquisition_date",
            "satellite",
        ]
    ).copy()

    # Select the lowest-cloud scene for each calendar month
    monthly = (
        monthly
        .groupby(
            "month_label",
            as_index=False
        )
        .first()
    )

    # Sort final monthly dataset chronologically
    monthly = monthly.sort_values(
        by="month_label"
    ).reset_index(drop=True)

    # ========================================================
    # SAVE MONTHLY SELECTION
    # ========================================================

    monthly.to_csv(
        MONTHLY_SELECTION_FILE,
        index=False
    )

    # ========================================================
    # INVENTORY SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FULL INVENTORY")
    print("=" * 70)

    print(
        f"Total candidate scenes : {len(df)}"
    )

    print(
        f"Date range             : "
        f"{df['acquisition_date'].min()} → "
        f"{df['acquisition_date'].max()}"
    )

    print(
        f"Best cloud cover       : "
        f"{df['cloud_cover'].min():.2f}%"
    )

    print(
        f"Average cloud cover    : "
        f"{df['cloud_cover'].mean():.2f}%"
    )

    print()

    print("Scenes by satellite:")

    print(
        df["satellite"]
        .value_counts()
        .to_string()
    )

    print()

    print("Scenes by year:")

    print(
        df["year"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================
    # MONTHLY SELECTION SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("MONTHLY SELECTION")
    print("=" * 70)

    print(
        f"Months with selected scenes : {len(monthly)}"
    )

    print(
        f"Selection date range        : "
        f"{monthly['month_label'].min()} → "
        f"{monthly['month_label'].max()}"
    )

    print()

    print("Selected scenes:")

    for _, row in monthly.iterrows():

        print(
            f"{row['month_label']} | "
            f"{row['acquisition_date']} | "
            f"{row['satellite']} | "
            f"{row['cloud_cover']:.2f}% | "
            f"{row['entity_id']}"
        )

    # ========================================================
    # MISSING MONTHS
    # ========================================================

    start_period = pd.Period(
        START_DATE[:7],
        freq="M"
    )

    end_period = pd.Period(
        END_DATE[:7],
        freq="M"
    )

    expected_months = pd.period_range(
        start=start_period,
        end=end_period,
        freq="M"
    )

    selected_months = set(
        pd.Period(
            value,
            freq="M"
        )
        for value in monthly["month_label"]
    )

    missing_months = [
        str(month)
        for month in expected_months
        if month not in selected_months
    ]

    print()
    print("=" * 70)
    print("MISSING MONTHS")
    print("=" * 70)

    if missing_months:

        print(
            f"Months without a qualifying scene: "
            f"{len(missing_months)}"
        )

        for month in missing_months:
            print(month)

    else:

        print("No months are missing.")

    # ========================================================
    # MONTHLY SATELLITE DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print("SELECTED SCENES BY SATELLITE")
    print("=" * 70)

    print(
        monthly["satellite"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"Full inventory   : {INVENTORY_FILE}"
    )

    print(
        f"Monthly selection: {MONTHLY_SELECTION_FILE}"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def get_selected_items():

    if not MONTHLY_SELECTION_FILE.exists():

        print()
        print("Monthly selection file does not exist:")
        print(MONTHLY_SELECTION_FILE)
        print()
        print("Run the inventory first.")
        return []

    monthly = pd.read_csv(
        MONTHLY_SELECTION_FILE
    )

    if monthly.empty:

        print("Monthly selection file is empty.")
        return []

    print()
    print("=" * 70)
    print("LOADING SELECTED SCENES")
    print("=" * 70)

    print(
        f"Selected scenes to download: {len(monthly)}"
    )

    catalog = Client.open(STAC_URL)

    selected_items = []

    for index, row in monthly.iterrows():

        entity_id = row["entity_id"]

        print(
            f"[{index + 1}/{len(monthly)}] "
            f"Finding {entity_id}..."
        )

        search = catalog.search(
            collections=[LANDSAT_COLLECTION],
            ids=[entity_id],
        )

        items = list(search.items())

        if not items:

            print(
                f"  WARNING: Scene not found: {entity_id}"
            )

            continue

        selected_items.append(items[0])

    print()
    print(
        f"Scenes successfully located: "
        f"{len(selected_items)}/{len(monthly)}"
    )

    return selected_items


def download_asset(session, asset_href, output_path):

    # --------------------------------------------------------
    # Skip files that already exist
    # --------------------------------------------------------

    if output_path.exists():

        print(
            f"    EXISTS: {output_path.name}"
        )

        return True

    # --------------------------------------------------------
    # Retry failed downloads
    # --------------------------------------------------------

    for attempt in range(
        1,
        DOWNLOAD_RETRIES + 1
    ):

        try:

            print(
                f"    Downloading: {output_path.name} "
                f"(attempt {attempt}/{DOWNLOAD_RETRIES})"
            )

            with session.get(
                asset_href,
                stream=True,
                timeout=DOWNLOAD_TIMEOUT,
            ) as response:

                response.raise_for_status()

                with open(
                    output_path,
                    "wb"
                ) as file:

                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):

                        if chunk:
                            file.write(chunk)

            print(
                f"    SUCCESS: {output_path.name}"
            )

            return True

        except Exception as error:

            print(
                f"    FAILED: {output_path.name}"
            )

            print(
                f"    Error: {error}"
            )

            # Remove incomplete file
            if output_path.exists():

                try:
                    output_path.unlink()

                except OSError:
                    pass

            if attempt < DOWNLOAD_RETRIES:

                print(
                    f"    Retrying in "
                    f"{RETRY_WAIT_SECONDS} seconds..."
                )

                time.sleep(
                    RETRY_WAIT_SECONDS
                )

    return False


# ============================================================
# DOWNLOAD SELECTED LANDSAT SCENES
# ============================================================

def download_selected_scenes():

    print()
    print("=" * 70)
    print("LANDSAT DOWNLOAD")
    print("=" * 70)

    print()
    print("Required assets:")

    for asset in REQUIRED_ASSETS:

        print(
            f"  - {asset}"
        )

    print()

    selected_items = get_selected_items()

    if not selected_items:

        print(
            "No scenes available for download."
        )

        return

    LANDSAT_RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    session = requests.Session()

    successful_scenes = 0
    failed_scenes = 0

    total_scenes = len(
        selected_items
    )

    for scene_number, item in enumerate(
        selected_items,
        start=1
    ):

        acquisition_date = get_acquisition_date(
            item
        )

        if acquisition_date is None:

            print()
            print(
                f"Skipping scene with no acquisition date: "
                f"{item.id}"
            )

            failed_scenes += 1

            continue

        scene_dir = (
            LANDSAT_RAW_DIR /
            acquisition_date
        )

        scene_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print()
        print("=" * 70)
        print(
            f"SCENE {scene_number}/{total_scenes}"
        )
        print("=" * 70)

        print(
            f"Scene ID       : {item.id}"
        )

        print(
            f"Acquisition    : {acquisition_date}"
        )

        print(
            f"Satellite      : {get_satellite(item)}"
        )

        cloud_cover = get_cloud_cover(
            item
        )

        if cloud_cover is not None:

            print(
                f"Cloud cover    : {cloud_cover:.2f}%"
            )

        print(
            f"Output folder  : {scene_dir}"
        )

        # ----------------------------------------------------
        # Sign Planetary Computer assets
        # ----------------------------------------------------

        signed_item = planetary_computer.sign(
            item
        )

        # ----------------------------------------------------
        # Check required assets
        # ----------------------------------------------------

        missing_assets = [
            asset
            for asset in REQUIRED_ASSETS
            if asset not in signed_item.assets
        ]

        if missing_assets:

            print()
            print(
                "WARNING: Required assets missing:"
            )

            for asset in missing_assets:

                print(
                    f"  - {asset}"
                )

            print()
            print(
                "Available assets:"
            )

            for asset_name in signed_item.assets:

                print(
                    f"  - {asset_name}"
                )

            failed_scenes += 1

            continue

        # ----------------------------------------------------
        # Download assets
        # ----------------------------------------------------

        scene_success = True

        for asset_name in REQUIRED_ASSETS:

            asset = signed_item.assets[
                asset_name
            ]

            output_path = (
                scene_dir /
                asset_name
            )

            success = download_asset(
                session=session,
                asset_href=asset.href,
                output_path=output_path,
            )

            if not success:

                scene_success = False

        if scene_success:

            successful_scenes += 1

            print()
            print(
                f"SCENE COMPLETE: {acquisition_date}"
            )

        else:

            failed_scenes += 1

            print()
            print(
                f"SCENE FAILED: {acquisition_date}"
            )

    session.close()

    # ========================================================
    # DOWNLOAD SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print(
        f"Scenes attempted : {total_scenes}"
    )

    print(
        f"Scenes complete  : {successful_scenes}"
    )

    print(
        f"Scenes failed    : {failed_scenes}"
    )

    print(
        f"Raw data folder  : {LANDSAT_RAW_DIR}"
    )

    print()

    if failed_scenes == 0:

        print(
            "ALL SELECTED SCENES DOWNLOADED SUCCESSFULLY."
        )

    else:

        print(
            "Some scenes failed. "
            "You can safely run the download command again."
        )

    print()
    print("=" * 70)
    print("DOWNLOAD DONE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Normal mode:
    #
    # python landsat_acquisition.py
    #
    # Creates/updates:
    # - landsat_inventory.csv
    # - landsat_monthly_selection.csv
    # --------------------------------------------------------

    if "--download" not in sys.argv:

        search_landsat()

    # --------------------------------------------------------
    # Download mode:
    #
    # python landsat_acquisition.py --download
    #
    # Uses the existing monthly selection CSV and downloads
    # the selected Landsat scenes.
    # --------------------------------------------------------

    else:

        download_selected_scenes()