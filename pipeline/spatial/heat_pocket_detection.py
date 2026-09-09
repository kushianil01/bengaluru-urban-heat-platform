from pathlib import Path
import argparse
import traceback

import numpy as np
import rasterio
from rasterio.mask import mask
from scipy import ndimage
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_LANDSAT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landsat"
)

HEAT_POCKETS_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "heat_pockets"
)


# ============================================================
# HEAT-POCKET PARAMETERS
# ============================================================

HEAT_PERCENTILE = 95

# 8-connectivity means diagonal pixels can belong
# to the same connected component.
CONNECTIVITY = 8

# Minimum number of pixels required for a
# retained heat pocket.
MIN_POCKET_PIXELS = 20


# ============================================================
# FUNCTIONS
# ============================================================

def read_lst(path):
    """Read LST GeoTIFF."""

    with rasterio.open(path) as src:

        lst = src.read(1).astype(np.float32)

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

        nodata = src.nodata

    return (
        lst,
        profile,
        transform,
        crs,
        nodata
    )


def calculate_threshold(lst):
    """
    Calculate the scene-specific 95th percentile
    using only valid LST pixels.
    """

    valid_lst = lst[
        np.isfinite(lst)
    ]

    if valid_lst.size == 0:

        raise ValueError(
            "No valid LST pixels found."
        )

    threshold = np.percentile(
        valid_lst,
        HEAT_PERCENTILE
    )

    return float(threshold)


def create_hot_pixel_mask(
    lst,
    threshold
):
    """Identify pixels at or above the heat threshold."""

    hot_pixels = (
        np.isfinite(lst)
        & (lst >= threshold)
    )

    return hot_pixels


def create_connectivity_structure():
    """Create 8- or 4-connectivity structure."""

    if CONNECTIVITY == 8:

        return np.ones(
            (3, 3),
            dtype=np.uint8
        )

    return np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        dtype=np.uint8
    )


def detect_heat_pockets(
    hot_pixel_mask
):
    """
    Label connected hot-pixel components
    and remove components smaller than
    the minimum pocket size.
    """

    structure = (
        create_connectivity_structure()
    )

    labels, number_of_components = (
        ndimage.label(
            hot_pixel_mask,
            structure=structure
        )
    )

    if number_of_components == 0:

        return (
            np.zeros(
                hot_pixel_mask.shape,
                dtype=np.uint32
            ),
            0,
            0
        )

    component_sizes = (
        np.bincount(
            labels.ravel()
        )
    )

    valid_components = (
        component_sizes
        >= MIN_POCKET_PIXELS
    )

    # Background label 0 must not
    # become a heat pocket.
    valid_components[0] = False

    retained_mask = (
        valid_components[labels]
    )

    retained_labels = np.zeros(
        labels.shape,
        dtype=np.uint32
    )

    retained_labels[
        retained_mask
    ] = labels[
        retained_mask
    ]

    retained_component_count = (
        np.count_nonzero(
            valid_components
        )
    )

    removed_component_count = (
        number_of_components
        - retained_component_count
    )

    return (
        retained_labels,
        retained_component_count,
        removed_component_count
    )


def calculate_pocket_statistics(
    labels,
    lst,
    transform
):
    """
    Calculate statistics for each retained
    heat pocket.
    """

    pixel_area_m2 = (
        abs(transform.a)
        * abs(transform.e)
    )

    pixel_area_ha = (
        pixel_area_m2 / 10000.0
    )

    records = []

    pocket_labels = np.unique(
        labels
    )

    pocket_labels = pocket_labels[
        pocket_labels > 0
    ]

    for pocket_id in pocket_labels:

        pocket_mask = (
            labels == pocket_id
        )

        pixel_count = np.count_nonzero(
            pocket_mask
        )

        area_ha = (
            pixel_count
            * pixel_area_ha
        )

        pocket_lst = lst[
            pocket_mask
        ]

        valid_pocket_lst = (
            pocket_lst[
                np.isfinite(
                    pocket_lst
                )
            ]
        )

        if valid_pocket_lst.size > 0:

            mean_lst = float(
                np.mean(
                    valid_pocket_lst
                )
            )

            max_lst = float(
                np.max(
                    valid_pocket_lst
                )
            )

        else:

            mean_lst = np.nan

            max_lst = np.nan

        records.append(
            {
                "pocket_id": int(
                    pocket_id
                ),
                "pixel_count": int(
                    pixel_count
                ),
                "area_ha": float(
                    area_ha
                ),
                "mean_lst_celsius": mean_lst,
                "max_lst_celsius": max_lst
            }
        )

    return records


def save_heat_pocket_raster(
    output_path,
    labels,
    profile,
    transform
):
    """
    Save labeled heat pockets as GeoTIFF.

    0 = background
    >0 = heat-pocket component ID
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_profile = profile.copy()

    output_profile.update(
        driver="GTiff",
        dtype="uint32",
        count=1,
        nodata=0,
        transform=transform,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER"
    )

    with rasterio.open(
        output_path,
        "w",
        **output_profile
    ) as dst:

        dst.write(
            labels.astype(
                np.uint32
            ),
            1
        )

        dst.set_band_description(
            1,
            "Heat Pocket Component ID"
        )


def save_statistics_csv(
    output_path,
    records
):
    """Save individual pocket statistics."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe = pd.DataFrame(
        records
    )

    dataframe.to_csv(
        output_path,
        index=False
    )


def outputs_exist(scene_date):
    """Check whether this scene has already been processed."""

    scene_output_dir = (
        HEAT_POCKETS_OUTPUT_DIR
        / scene_date
    )

    raster_output = (
        scene_output_dir
        / "heat_pockets.tif"
    )

    csv_output = (
        scene_output_dir
        / "heat_pocket_statistics.csv"
    )

    return (
        raster_output.exists()
        and csv_output.exists()
    )


# ============================================================
# SINGLE SCENE PROCESSING
# ============================================================

def process_scene(scene_date):

    lst_path = (
        PROCESSED_LANDSAT_DIR
        / scene_date
        / "lst_celsius.tif"
    )

    if not lst_path.exists():

        raise FileNotFoundError(
            f"LST file not found:\n{lst_path}"
        )

    print("=" * 70)

    print(
        f"HEAT-POCKET DETECTION: "
        f"{scene_date}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 1. Read LST
    # --------------------------------------------------------

    print(
        "\n[1/5] Reading LST raster..."
    )

    (
        lst,
        profile,
        transform,
        crs,
        nodata
    ) = read_lst(
        lst_path
    )

    valid_pixels = np.isfinite(
        lst
    )

    valid_pixel_count = (
        np.count_nonzero(
            valid_pixels
        )
    )

    print(
        f"Raster size: "
        f"{lst.shape[1]} x "
        f"{lst.shape[0]}"
    )

    print(
        f"CRS: {crs}"
    )

    print(
        f"Valid LST pixels: "
        f"{valid_pixel_count:,}"
    )

    if valid_pixel_count == 0:

        raise ValueError(
            "Scene contains no valid LST pixels."
        )

    # --------------------------------------------------------
    # 2. Calculate threshold
    # --------------------------------------------------------

    print(
        "\n[2/5] Calculating "
        f"{HEAT_PERCENTILE}th percentile..."
    )

    threshold = calculate_threshold(
        lst
    )

    print(
        f"Heat threshold: "
        f"{threshold:.3f} °C"
    )

    # --------------------------------------------------------
    # 3. Identify hot pixels
    # --------------------------------------------------------

    print(
        "\n[3/5] Identifying hot pixels..."
    )

    hot_pixel_mask = (
        create_hot_pixel_mask(
            lst,
            threshold
        )
    )

    hot_pixel_count = (
        np.count_nonzero(
            hot_pixel_mask
        )
    )

    hot_pixel_percentage = (
        hot_pixel_count
        / valid_pixel_count
        * 100
    )

    print(
        f"Hot pixels: "
        f"{hot_pixel_count:,}"
    )

    print(
        f"Hot pixels among valid pixels: "
        f"{hot_pixel_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # 4. Detect connected components
    # --------------------------------------------------------

    print(
        "\n[4/5] Detecting connected "
        "heat pockets..."
    )

    (
        labels,
        retained_count,
        removed_count
    ) = detect_heat_pockets(
        hot_pixel_mask
    )

    print(
        f"Retained heat pockets: "
        f"{retained_count}"
    )

    print(
        f"Removed tiny components: "
        f"{removed_count}"
    )

    # --------------------------------------------------------
    # 5. Statistics + save
    # --------------------------------------------------------

    print(
        "\n[5/5] Calculating pocket "
        "statistics and saving..."
    )

    records = (
        calculate_pocket_statistics(
            labels,
            lst,
            transform
        )
    )

    output_scene_dir = (
        HEAT_POCKETS_OUTPUT_DIR
        / scene_date
    )

    output_scene_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    raster_output = (
        output_scene_dir
        / "heat_pockets.tif"
    )

    csv_output = (
        output_scene_dir
        / "heat_pocket_statistics.csv"
    )

    save_heat_pocket_raster(
        raster_output,
        labels,
        profile,
        transform
    )

    save_statistics_csv(
        csv_output,
        records
    )

    total_pocket_area_ha = sum(
        record["area_ha"]
        for record in records
    )

    largest_pocket_area_ha = max(
        (
            record["area_ha"]
            for record in records
        ),
        default=0.0
    )

    # --------------------------------------------------------
    # Return master summary information
    # --------------------------------------------------------

    summary = {
        "date": scene_date,
        "heat_percentile": HEAT_PERCENTILE,
        "heat_threshold_celsius": threshold,
        "valid_pixels": valid_pixel_count,
        "hot_pixels": hot_pixel_count,
        "hot_pixel_percentage": (
            hot_pixel_percentage
        ),
        "total_components": (
            retained_count
            + removed_count
        ),
        "retained_pockets": (
            retained_count
        ),
        "removed_tiny_components": (
            removed_count
        ),
        "total_heat_pocket_area_ha": (
            total_pocket_area_ha
        ),
        "largest_heat_pocket_ha": (
            largest_pocket_area_ha
        )
    }

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HEAT-POCKET DETECTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDate: {scene_date}"
    )

    print(
        f"95th percentile threshold: "
        f"{threshold:.3f} °C"
    )

    print(
        f"Hot pixels: "
        f"{hot_pixel_count:,}"
    )

    print(
        f"Retained pockets: "
        f"{retained_count}"
    )

    print(
        f"Total heat-pocket area: "
        f"{total_pocket_area_ha:.2f} ha"
    )

    print(
        f"Largest heat pocket: "
        f"{largest_pocket_area_ha:.2f} ha"
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {raster_output}"
    )

    print(
        f"  {csv_output}"
    )

    print(
        "\nDONE."
    )

    return summary


# ============================================================
# BATCH PROCESSING
# ============================================================

def process_all_scenes():

    print("=" * 70)
    print(
        "BATCH HEAT-POCKET DETECTION"
    )
    print("=" * 70)

    # Find all processed Landsat scene folders.
    scene_dirs = sorted(
        [
            path
            for path in PROCESSED_LANDSAT_DIR.iterdir()
            if path.is_dir()
        ]
    )

    if not scene_dirs:

        raise RuntimeError(
            "No processed Landsat scene "
            "directories were found."
        )

    total = len(scene_dirs)

    print(
        f"\nProcessed scenes found: "
        f"{total}"
    )

    successful = []

    skipped = []

    failed = []

    summaries = []

    for index, scene_dir in enumerate(
        scene_dirs,
        start=1
    ):

        scene_date = scene_dir.name

        print(
            "\n\n"
            + "#" * 70
        )

        print(
            f"SCENE {index}/{total}: "
            f"{scene_date}"
        )

        print(
            "#" * 70
        )

        # ----------------------------------------------------
        # Skip completed scenes
        # ----------------------------------------------------

        if outputs_exist(
            scene_date
        ):

            print(
                "\nSKIPPING: "
                "heat-pocket outputs "
                "already exist."
            )

            skipped.append(
                scene_date
            )

            continue

        # ----------------------------------------------------
        # Process scene
        # ----------------------------------------------------

        try:

            summary = process_scene(
                scene_date
            )

            successful.append(
                scene_date
            )

            summaries.append(
                summary
            )

        except Exception as exc:

            print(
                "\n"
                + "!" * 70
            )

            print(
                f"FAILED: {scene_date}"
            )

            print(
                f"Error: {exc}"
            )

            print(
                "!" * 70
            )

            traceback.print_exc()

            failed.append(
                (
                    scene_date,
                    str(exc)
                )
            )

    # ========================================================
    # MASTER SUMMARY
    # ========================================================

    if summaries:

        summary_dataframe = (
            pd.DataFrame(
                summaries
            )
        )

        summary_path = (
            HEAT_POCKETS_OUTPUT_DIR
            / "heat_pocket_summary.csv"
        )

        summary_dataframe.to_csv(
            summary_path,
            index=False
        )

        print(
            "\nMaster summary saved:"
        )

        print(
            summary_path
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "BATCH PROCESSING SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal scenes found: "
        f"{total}"
    )

    print(
        f"Successfully processed: "
        f"{len(successful)}"
    )

    print(
        f"Already processed/skipped: "
        f"{len(skipped)}"
    )

    print(
        f"Failed: "
        f"{len(failed)}"
    )

    if skipped:

        print(
            "\nSkipped scenes:"
        )

        for scene_date in skipped:

            print(
                f"  - {scene_date}"
            )

    if failed:

        print(
            "\nFailed scenes:"
        )

        for scene_date, error in failed:

            print(
                f"  - {scene_date}: "
                f"{error}"
            )

    print(
        "\n"
        + "=" * 70
    )

    if failed:

        print(
            "BATCH COMPLETED WITH FAILURES"
        )

    else:

        print(
            "ALL SCENES COMPLETED SUCCESSFULLY"
        )

    print(
        "=" * 70
    )


# ============================================================
# COMMAND LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Detect Landsat-based heat "
            "pockets in Bengaluru."
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--date",
        help=(
            "Process one scene. "
            "Format: YYYY-MM-DD"
        )
    )

    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Process all available "
            "processed Landsat scenes."
        )
    )

    args = parser.parse_args()

    if args.date:

        process_scene(
            args.date
        )

    elif args.all:

        process_all_scenes()