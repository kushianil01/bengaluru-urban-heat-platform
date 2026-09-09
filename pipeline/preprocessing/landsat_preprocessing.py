from pathlib import Path
import argparse
import traceback

import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_LANDSAT_DIR = PROJECT_ROOT / "data" / "raw" / "landsat"

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "boundaries"
    / "gba_wards_369.geojson"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "landsat"


# ============================================================
# LANDSAT COLLECTION 2 SCALE FACTORS
# ============================================================

SR_SCALE = 0.0000275
SR_OFFSET = -0.2

ST_SCALE = 0.00341802
ST_OFFSET_KELVIN = 149.0


# ============================================================
# QA_PIXEL BITS
# ============================================================

# Landsat 8/9 QA_PIXEL:
# Bit 0 = Fill
# Bit 1 = Dilated Cloud
# Bit 2 = Cirrus
# Bit 3 = Cloud
# Bit 4 = Cloud Shadow
# Bit 5 = Snow

QA_BITS_TO_MASK = [0, 1, 2, 3, 4, 5]


# ============================================================
# HELPERS
# ============================================================

def read_band(path):
    """Read a single-band raster as float32."""

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    return data, profile, transform, crs


def create_qa_mask(qa):
    """
    Return True for pixels that should be masked.

    Pixels are masked when any selected QA bits are set.
    """

    mask_array = np.zeros(qa.shape, dtype=bool)

    for bit in QA_BITS_TO_MASK:
        mask_array |= (qa & (1 << bit)) != 0

    return mask_array


def scale_reflectance(dn):
    """Convert Landsat Collection 2 surface reflectance DN."""

    result = dn * SR_SCALE + SR_OFFSET

    result[dn == 0] = np.nan

    return result


def scale_lst(dn):
    """Convert Landsat Collection 2 surface temperature DN to Celsius."""

    result_kelvin = (
        dn * ST_SCALE
        + ST_OFFSET_KELVIN
    )

    result_celsius = result_kelvin - 273.15

    result_celsius[dn == 0] = np.nan

    return result_celsius


def safe_normalized_difference(a, b):
    """
    Calculate:

        (A - B) / (A + B)

    safely.
    """

    denominator = a + b

    result = np.full(
        a.shape,
        np.nan,
        dtype=np.float32
    )

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-6)
    )

    result[valid] = (
        (a[valid] - b[valid])
        / denominator[valid]
    )

    return result


def clip_array_to_boundary(
    array,
    source_profile,
    boundary_gdf
):
    """
    Clip an array to the GBA boundary.
    """

    profile = source_profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan
    )

    from rasterio.io import MemoryFile

    with MemoryFile() as memfile:

        with memfile.open(**profile) as temp_src:

            temp_src.write(
                array.astype(np.float32),
                1
            )

            clipped, clipped_transform = mask(
                temp_src,
                boundary_gdf.geometry,
                crop=True,
                nodata=np.nan
            )

    return clipped[0], clipped_transform


def save_raster(
    path,
    array,
    transform,
    crs,
    description
):
    """Save a single-band float32 GeoTIFF."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    height, width = array.shape

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "BIGTIFF": "IF_SAFER"
    }

    with rasterio.open(path, "w", **profile) as dst:

        dst.write(
            array.astype(np.float32),
            1
        )

        dst.set_band_description(
            1,
            description
        )


def print_statistics(name, array):
    """Print useful raster statistics."""

    valid = array[np.isfinite(array)]

    if valid.size == 0:

        print(
            f"{name}: NO VALID PIXELS"
        )

        return

    print(
        f"{name}: "
        f"min={np.nanmin(valid):.3f}, "
        f"max={np.nanmax(valid):.3f}, "
        f"mean={np.nanmean(valid):.3f}, "
        f"median={np.nanmedian(valid):.3f}, "
        f"valid_pixels={valid.size:,}"
    )


def outputs_exist(scene_date):
    """Check whether all expected outputs already exist."""

    scene_output_dir = (
        OUTPUT_DIR / scene_date
    )

    required_outputs = [
        scene_output_dir / "lst_celsius.tif",
        scene_output_dir / "ndvi.tif",
        scene_output_dir / "ndbi.tif",
        scene_output_dir / "ndwi.tif",
    ]

    return all(
        path.exists()
        for path in required_outputs
    )


# ============================================================
# SINGLE SCENE PREPROCESSING
# ============================================================

def preprocess_scene(scene_date):

    scene_dir = (
        RAW_LANDSAT_DIR / scene_date
    )

    if not scene_dir.exists():

        raise FileNotFoundError(
            f"Scene directory not found:\n{scene_dir}"
        )

    print("=" * 70)
    print(
        f"PROCESSING LANDSAT SCENE: "
        f"{scene_date}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    red_path = scene_dir / "red"
    green_path = scene_dir / "green"
    nir_path = scene_dir / "nir08"
    swir16_path = scene_dir / "swir16"
    lwir_path = scene_dir / "lwir11"
    qa_path = scene_dir / "qa_pixel"

    required_files = [
        red_path,
        green_path,
        nir_path,
        swir16_path,
        lwir_path,
        qa_path,
        BOUNDARY_FILE
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Read raster bands
    # --------------------------------------------------------

    print("\n[1/7] Reading Landsat bands...")

    red, profile, transform, crs = read_band(
        red_path
    )

    green, _, _, _ = read_band(
        green_path
    )

    nir, _, _, _ = read_band(
        nir_path
    )

    swir16, _, _, _ = read_band(
        swir16_path
    )

    lwir, _, _, _ = read_band(
        lwir_path
    )

    qa, _, _, _ = read_band(
        qa_path
    )

    qa = qa.astype(np.uint16)

    print(
        f"Raster size: "
        f"{red.shape[1]} x {red.shape[0]}"
    )

    print(
        f"CRS: {crs}"
    )

    print(
        f"Resolution: {transform.a} m"
    )

    # --------------------------------------------------------
    # QA MASK
    # --------------------------------------------------------

    print(
        "\n[2/7] Applying "
        "QA_PIXEL cloud/shadow/snow mask..."
    )

    qa_mask = create_qa_mask(
        qa
    )

    total_pixels = qa_mask.size

    masked_pixels = np.count_nonzero(
        qa_mask
    )

    masked_percentage = (
        masked_pixels
        / total_pixels
        * 100
    )

    print(
        f"QA-masked pixels: "
        f"{masked_pixels:,} / "
        f"{total_pixels:,} "
        f"({masked_percentage:.2f}%)"
    )

    # --------------------------------------------------------
    # SCALE REFLECTANCE
    # --------------------------------------------------------

    print(
        "\n[3/7] Scaling surface reflectance..."
    )

    red = scale_reflectance(red)
    green = scale_reflectance(green)
    nir = scale_reflectance(nir)
    swir16 = scale_reflectance(swir16)

    red[qa_mask] = np.nan
    green[qa_mask] = np.nan
    nir[qa_mask] = np.nan
    swir16[qa_mask] = np.nan

    # --------------------------------------------------------
    # SCALE LST
    # --------------------------------------------------------

    print(
        "\n[4/7] Converting surface "
        "temperature to Celsius..."
    )

    lst = scale_lst(lwir)

    lst[qa_mask] = np.nan

    print_statistics(
        "LST (°C)",
        lst
    )

    # --------------------------------------------------------
    # CALCULATE INDICES
    # --------------------------------------------------------

    print(
        "\n[5/7] Calculating "
        "NDVI, NDBI and NDWI..."
    )

    # NDVI
    ndvi = safe_normalized_difference(
        nir,
        red
    )

    # NDBI
    ndbi = safe_normalized_difference(
        swir16,
        nir
    )

    # NDWI
    ndwi = safe_normalized_difference(
        green,
        nir
    )

    # Keep indices within their
    # mathematical range.
    ndvi[
        (ndvi < -1)
        | (ndvi > 1)
    ] = np.nan

    ndbi[
        (ndbi < -1)
        | (ndbi > 1)
    ] = np.nan

    ndwi[
        (ndwi < -1)
        | (ndwi > 1)
    ] = np.nan

    print_statistics(
        "NDVI",
        ndvi
    )

    print_statistics(
        "NDBI",
        ndbi
    )

    print_statistics(
        "NDWI",
        ndwi
    )

    # --------------------------------------------------------
    # LOAD GBA BOUNDARY
    # --------------------------------------------------------

    print(
        "\n[6/7] Clipping to "
        "Bengaluru GBA boundary..."
    )

    gdf = gpd.read_file(
        BOUNDARY_FILE
    )

    if gdf.empty:

        raise ValueError(
            "GBA boundary file "
            "contains no features."
        )

    print(
        f"Boundary features: "
        f"{len(gdf)}"
    )

    print(
        f"Boundary CRS: "
        f"{gdf.crs}"
    )

    # Reproject boundary to
    # Landsat CRS.
    if gdf.crs != crs:

        gdf = gdf.to_crs(
            crs
        )

    # --------------------------------------------------------
    # CLIP PRODUCTS
    # --------------------------------------------------------

    clipped_lst, clipped_transform = (
        clip_array_to_boundary(
            lst,
            profile,
            gdf
        )
    )

    clipped_ndvi, _ = (
        clip_array_to_boundary(
            ndvi,
            profile,
            gdf
        )
    )

    clipped_ndbi, _ = (
        clip_array_to_boundary(
            ndbi,
            profile,
            gdf
        )
    )

    clipped_ndwi, _ = (
        clip_array_to_boundary(
            ndwi,
            profile,
            gdf
        )
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    scene_output_dir = (
        OUTPUT_DIR / scene_date
    )

    scene_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\n[7/7] Saving processed rasters..."
    )

    save_raster(
        scene_output_dir
        / "lst_celsius.tif",
        clipped_lst,
        clipped_transform,
        crs,
        "Land Surface Temperature (Celsius)"
    )

    save_raster(
        scene_output_dir
        / "ndvi.tif",
        clipped_ndvi,
        clipped_transform,
        crs,
        "Normalized Difference Vegetation Index"
    )

    save_raster(
        scene_output_dir
        / "ndbi.tif",
        clipped_ndbi,
        clipped_transform,
        crs,
        "Normalized Difference Built-up Index"
    )

    save_raster(
        scene_output_dir
        / "ndwi.tif",
        clipped_ndwi,
        clipped_transform,
        crs,
        "Normalized Difference Water Index"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nOutput directory:"
    )

    print(
        scene_output_dir
    )

    print(
        "\nSaved files:"
    )

    print(
        "  - lst_celsius.tif"
    )

    print(
        "  - ndvi.tif"
    )

    print(
        "  - ndbi.tif"
    )

    print(
        "  - ndwi.tif"
    )

    print(
        "\nFinal clipped raster size:"
    )

    print(
        f"  {clipped_lst.shape[1]} x "
        f"{clipped_lst.shape[0]} pixels"
    )

    print(
        "\nFinal LST statistics:"
    )

    print_statistics(
        "LST (°C)",
        clipped_lst
    )

    print(
        "\nFinal NDVI statistics:"
    )

    print_statistics(
        "NDVI",
        clipped_ndvi
    )

    print(
        "\nFinal NDBI statistics:"
    )

    print_statistics(
        "NDBI",
        clipped_ndbi
    )

    print(
        "\nFinal NDWI statistics:"
    )

    print_statistics(
        "NDWI",
        clipped_ndwi
    )

    print(
        "\nDONE."
    )


# ============================================================
# BATCH PROCESSING
# ============================================================

def preprocess_all_scenes():

    print("=" * 70)
    print("BATCH LANDSAT PREPROCESSING")
    print("=" * 70)

    # Find all downloaded scene folders.
    scene_dirs = sorted(
        [
            path
            for path in RAW_LANDSAT_DIR.iterdir()
            if path.is_dir()
        ]
    )

    if not scene_dirs:

        raise RuntimeError(
            "No Landsat scene directories "
            "were found."
        )

    total = len(scene_dirs)

    print(
        f"\nDownloaded scenes found: {total}"
    )

    successful = []
    skipped = []
    failed = []

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
        # Skip already completed scenes
        # ----------------------------------------------------

        if outputs_exist(scene_date):

            print(
                "\nSKIPPING: "
                "processed outputs already exist."
            )

            skipped.append(
                scene_date
            )

            continue

        # ----------------------------------------------------
        # Process scene
        # ----------------------------------------------------

        try:

            preprocess_scene(
                scene_date
            )

            successful.append(
                scene_date
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
            "Preprocess Landsat 8/9 scenes "
            "for Bengaluru."
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
            "Process all downloaded "
            "Landsat scenes."
        )
    )

    args = parser.parse_args()

    if args.date:

        preprocess_scene(
            args.date
        )

    elif args.all:

        preprocess_all_scenes()