from pathlib import Path
import argparse

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HEAT_POCKETS_DIR = (
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

PERSISTENCE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "persistence"
)


# ============================================================
# ANALYSIS PARAMETERS
# ============================================================

TARGET_CRS = "EPSG:32643"

PIXEL_SIZE = 30.0

EXPECTED_SCENE_COUNT = 76


# ============================================================
# FIND HEAT-POCKET RASTERS
# ============================================================

def find_heat_pocket_rasters():
    """
    Find all scene-level heat-pocket rasters.
    """

    raster_paths = sorted(
        HEAT_POCKETS_DIR.glob(
            "*/heat_pockets.tif"
        )
    )

    if not raster_paths:

        raise RuntimeError(
            "No heat-pocket rasters were found."
        )

    return raster_paths


# ============================================================
# LOAD GBA BOUNDARY
# ============================================================

def load_gba_boundary():
    """
    Load the 369-ward GBA boundary and reproject
    it to the Landsat analysis CRS.
    """

    if not BOUNDARY_PATH.exists():

        raise FileNotFoundError(
            f"GBA boundary not found:\n"
            f"{BOUNDARY_PATH}"
        )

    gdf = gpd.read_file(
        BOUNDARY_PATH
    )

    if gdf.empty:

        raise ValueError(
            "GBA boundary file contains no features."
        )

    print(
        f"GBA features loaded: "
        f"{len(gdf)}"
    )

    gdf = gdf.to_crs(
        TARGET_CRS
    )

    print(
        f"GBA CRS: "
        f"{gdf.crs}"
    )

    return gdf


# ============================================================
# DETERMINE COMMON GRID
# ============================================================

def determine_common_grid(
    reference_raster,
    gdf
):
    """
    Create a common 30 m grid covering the GBA.

    The grid is aligned to the existing Landsat raster
    origin so that no heat-pocket pixels are resampled.
    """

    with rasterio.open(
        reference_raster
    ) as src:

        raster_transform = src.transform
        raster_crs = src.crs
        raster_res = src.res

    if raster_crs != rasterio.crs.CRS.from_string(
        TARGET_CRS
    ):

        raise ValueError(
            f"Reference raster CRS is "
            f"{raster_crs}, expected "
            f"{TARGET_CRS}."
        )

    if not np.isclose(
        raster_res[0],
        PIXEL_SIZE
    ) or not np.isclose(
        abs(raster_res[1]),
        PIXEL_SIZE
    ):

        raise ValueError(
            f"Reference raster resolution "
            f"is {raster_res}, expected "
            f"{PIXEL_SIZE} x {PIXEL_SIZE} m."
        )

    min_x, min_y, max_x, max_y = (
        gdf.total_bounds
    )

    # Existing Landsat grid origin.
    origin_x = raster_transform.c
    origin_y = raster_transform.f

    # Snap the GBA extent outward to the existing
    # 30 m Landsat grid.
    min_col = int(
        np.floor(
            (min_x - origin_x)
            / PIXEL_SIZE
        )
    )

    max_col = int(
        np.ceil(
            (max_x - origin_x)
            / PIXEL_SIZE
        )
    )

    min_row = int(
        np.ceil(
            (origin_y - max_y)
            / PIXEL_SIZE
        )
    )

    max_row = int(
        np.floor(
            (origin_y - min_y)
            / PIXEL_SIZE
        )
    )

    width = (
        max_col
        - min_col
    )

    height = (
        max_row
        - min_row
    )

    if width <= 0 or height <= 0:

        raise ValueError(
            "Calculated GBA grid has "
            "invalid dimensions."
        )

    grid_origin_x = (
        origin_x
        + min_col * PIXEL_SIZE
    )

    grid_origin_y = (
        origin_y
        - min_row * PIXEL_SIZE
    )

    transform = from_origin(
        grid_origin_x,
        grid_origin_y,
        PIXEL_SIZE,
        PIXEL_SIZE
    )

    print(
        "\nCommon persistence grid:"
    )

    print(
        f"  CRS: {TARGET_CRS}"
    )

    print(
        f"  Resolution: "
        f"{PIXEL_SIZE} x {PIXEL_SIZE} m"
    )

    print(
        f"  Width: {width}"
    )

    print(
        f"  Height: {height}"
    )

    print(
        f"  Origin: "
        f"{grid_origin_x:.2f}, "
        f"{grid_origin_y:.2f}"
    )

    print(
        f"  Bounds: "
        f"left={grid_origin_x:.2f}, "
        f"bottom="
        f"{grid_origin_y - height * PIXEL_SIZE:.2f}, "
        f"right="
        f"{grid_origin_x + width * PIXEL_SIZE:.2f}, "
        f"top={grid_origin_y:.2f}"
    )

    return (
        transform,
        width,
        height
    )


# ============================================================
# CREATE GBA MASK
# ============================================================

def create_gba_mask(
    gdf,
    transform,
    width,
    height
):
    """
    Create a boolean mask for the GBA.

    True  = pixel is inside GBA
    False = pixel is outside GBA
    """

    geometries = [
        geometry
        for geometry in gdf.geometry
        if geometry is not None
        and not geometry.is_empty
    ]

    if not geometries:

        raise ValueError(
            "No valid geometries found "
            "in GBA boundary."
        )

    mask_array = geometry_mask(
        geometries,
        out_shape=(
            height,
            width
        ),
        transform=transform,
        invert=True,
        all_touched=False
    )

    return mask_array


# ============================================================
# CHECK RASTER GRID ALIGNMENT
# ============================================================

def check_raster_alignment(
    raster_paths,
    transform,
    width,
    height
):
    """
    Check whether every heat-pocket raster uses the same
    30 m grid origin and can be placed into the common grid
    without reprojection/resampling.
    """

    common_origin_x = transform.c
    common_origin_y = transform.f

    aligned = []
    misaligned = []

    print(
        "\nChecking heat-pocket raster alignment..."
    )

    for raster_path in raster_paths:

        with rasterio.open(
            raster_path
        ) as src:

            raster_transform = src.transform

            same_crs = (
                src.crs
                == rasterio.crs.CRS.from_string(
                    TARGET_CRS
                )
            )

            same_resolution = (
                np.isclose(
                    src.res[0],
                    PIXEL_SIZE
                )
                and
                np.isclose(
                    abs(src.res[1]),
                    PIXEL_SIZE
                )
            )

            # Check whether the raster starts on the
            # same global 30 m grid.
            x_offset = (
                (
                    raster_transform.c
                    - common_origin_x
                )
                / PIXEL_SIZE
            )

            y_offset = (
                (
                    common_origin_y
                    - raster_transform.f
                )
                / PIXEL_SIZE
            )

            x_aligned = np.isclose(
                x_offset,
                round(x_offset),
                atol=1e-6
            )

            y_aligned = np.isclose(
                y_offset,
                round(y_offset),
                atol=1e-6
            )

            is_aligned = (
                same_crs
                and same_resolution
                and x_aligned
                and y_aligned
            )

            if is_aligned:

                aligned.append(
                    raster_path.parent.name
                )

            else:

                misaligned.append(
                    raster_path.parent.name
                )

    print(
        f"Aligned rasters: "
        f"{len(aligned)}"
    )

    print(
        f"Misaligned rasters: "
        f"{len(misaligned)}"
    )

    if misaligned:

        print(
            "\nMisaligned scenes:"
        )

        for scene_date in misaligned:

            print(
                f"  - {scene_date}"
            )

        raise ValueError(
            "Some heat-pocket rasters are "
            "not aligned to the common grid."
        )

    return aligned


# ============================================================
# BUILD PIXEL PERSISTENCE
# ============================================================

def calculate_pixel_persistence(
    raster_paths,
    gba_mask,
    transform,
    width,
    height
):
    """
    Count how many observations identify each pixel
    as belonging to a retained heat pocket.

    No resampling is performed.

    Pixels outside the GBA are masked out.
    """

    persistence_count = np.zeros(
        (height, width),
        dtype=np.uint16
    )

    total_scenes = len(
        raster_paths
    )

    print(
        "\nCalculating pixel-level persistence..."
    )

    for index, raster_path in enumerate(
        raster_paths,
        start=1
    ):

        scene_date = (
            raster_path.parent.name
        )

        print(
            f"  [{index}/{total_scenes}] "
            f"{scene_date}"
        )

        with rasterio.open(
            raster_path
        ) as src:

            raster = src.read(
                1
            )

            raster_transform = (
                src.transform
            )

            # Determine where this raster overlaps
            # the common persistence grid.
            col_offset = int(
                round(
                    (
                        raster_transform.c
                        - transform.c
                    )
                    / PIXEL_SIZE
                )
            )

            row_offset = int(
                round(
                    (
                        transform.f
                        - raster_transform.f
                    )
                    / PIXEL_SIZE
                )
            )

            row_start = row_offset
            col_start = col_offset

            row_end = min(
                row_start + src.height,
                height
            )

            col_end = min(
                col_start + src.width,
                width
            )

            source_row_start = 0
            source_col_start = 0

            if row_start < 0:

                source_row_start = -row_start
                row_start = 0

            if col_start < 0:

                source_col_start = -col_start
                col_start = 0

            source_row_end = (
                source_row_start
                + (
                    row_end
                    - row_start
                )
            )

            source_col_end = (
                source_col_start
                + (
                    col_end
                    - col_start
                )
            )

            if (
                row_end <= row_start
                or col_end <= col_start
            ):

                continue

            source_data = raster[
                source_row_start:source_row_end,
                source_col_start:source_col_end
            ]

            source_hot = (
                source_data > 0
            )

            target_mask = gba_mask[
                row_start:row_end,
                col_start:col_end
            ]

            persistence_count[
                row_start:row_end,
                col_start:col_end
            ] += (
                source_hot
                & target_mask
            ).astype(
                np.uint16
            )

    # Pixels outside the GBA are represented as 0
    # in the count raster, but will receive nodata
    # when the persistence percentage is saved.
    persistence_percentage = (
        persistence_count.astype(
            np.float32
        )
        / total_scenes
        * 100.0
    )

    persistence_percentage[
        ~gba_mask
    ] = np.nan

    persistence_count[
        ~gba_mask
    ] = 0

    return (
        persistence_count,
        persistence_percentage
    )


# ============================================================
# SAVE RASTER
# ============================================================

def save_persistence_raster(
    output_path,
    data,
    transform,
    width,
    height,
    dtype,
    nodata,
    description
):
    """
    Save a persistence raster.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER"
    }

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            data,
            1
        )

        dst.set_band_description(
            1,
            description
        )


# ============================================================
# MAIN PERSISTENCE PIPELINE
# ============================================================

def run_persistence_analysis():

    print("=" * 70)
    print(
        "BENGALURU HEAT-POCKET PERSISTENCE ANALYSIS"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Find heat-pocket rasters
    # --------------------------------------------------------

    raster_paths = (
        find_heat_pocket_rasters()
    )

    print(
        f"\nHeat-pocket rasters found: "
        f"{len(raster_paths)}"
    )

    if len(raster_paths) != EXPECTED_SCENE_COUNT:

        raise ValueError(
            f"Expected {EXPECTED_SCENE_COUNT} "
            f"heat-pocket rasters, but found "
            f"{len(raster_paths)}."
        )

    # --------------------------------------------------------
    # 2. Load GBA boundary
    # --------------------------------------------------------

    gdf = load_gba_boundary()

    # --------------------------------------------------------
    # 3. Determine common grid
    # --------------------------------------------------------

    reference_raster = (
        raster_paths[0]
    )

    (
        transform,
        width,
        height
    ) = determine_common_grid(
        reference_raster,
        gdf
    )

    # --------------------------------------------------------
    # 4. Create GBA mask
    # --------------------------------------------------------

    print(
        "\nCreating GBA spatial mask..."
    )

    gba_mask = create_gba_mask(
        gdf,
        transform,
        width,
        height
    )

    gba_pixel_count = (
        np.count_nonzero(
            gba_mask
        )
    )

    gba_area_ha = (
        gba_pixel_count
        * PIXEL_SIZE
        * PIXEL_SIZE
        / 10000.0
    )

    print(
        f"GBA pixels: "
        f"{gba_pixel_count:,}"
    )

    print(
        f"Approximate GBA raster area: "
        f"{gba_area_ha:.2f} ha"
    )

    # --------------------------------------------------------
    # 5. Check all raster alignment
    # --------------------------------------------------------

    check_raster_alignment(
        raster_paths,
        transform,
        width,
        height
    )

    # --------------------------------------------------------
    # 6. Calculate persistence
    # --------------------------------------------------------

    (
        persistence_count,
        persistence_percentage
    ) = calculate_pixel_persistence(
        raster_paths,
        gba_mask,
        transform,
        width,
        height
    )

    # --------------------------------------------------------
    # 7. Save outputs
    # --------------------------------------------------------

    PERSISTENCE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    count_output = (
        PERSISTENCE_OUTPUT_DIR
        / "heat_pocket_persistence_count.tif"
    )

    percentage_output = (
        PERSISTENCE_OUTPUT_DIR
        / "heat_pocket_persistence_percentage.tif"
    )

    save_persistence_raster(
        count_output,
        persistence_count,
        transform,
        width,
        height,
        "uint16",
        0,
        "Heat Pocket Persistence Count"
    )

    save_persistence_raster(
        percentage_output,
        persistence_percentage,
        transform,
        width,
        height,
        "float32",
        np.nan,
        "Heat Pocket Persistence Percentage"
    )

    # --------------------------------------------------------
    # 8. Print summary
    # --------------------------------------------------------

    valid_persistence = (
        persistence_count[
            gba_mask
        ]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PERSISTENCE ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nObservations: "
        f"{len(raster_paths)}"
    )

    print(
        f"GBA pixels: "
        f"{gba_pixel_count:,}"
    )

    print(
        f"Maximum persistence count: "
        f"{valid_persistence.max()}"
    )

    print(
        f"Minimum persistence count: "
        f"{valid_persistence.min()}"
    )

    print(
        f"Mean persistence count: "
        f"{valid_persistence.mean():.2f}"
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {count_output}"
    )

    print(
        f"  {percentage_output}"
    )

    print(
        "\nDONE."
    )


# ============================================================
# COMMAND LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Calculate pixel-level temporal "
            "persistence of Landsat heat pockets "
            "across Bengaluru."
        )
    )

    parser.parse_args()

    run_persistence_analysis()