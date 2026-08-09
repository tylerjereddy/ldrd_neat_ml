from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.color
from joblib import Memory
from tqdm.auto import tqdm


__all__: Sequence[str] = [
    "run_opencv"
] 


def _detect_single_image(
    img_path: Path
) -> tuple[int, float, pd.DataFrame]:
    """
    Detect bubbles in a single image using OpenCV's SimpleBlobDetector.

    Parameters
    ----------
    img_path : Path
        Absolute file path to the image to process.

    Returns
    -------
    tuple[int, float, pd.DataFrame]
        num_blobs : int
            Number of blobs detected in the image.
        median_radius : float
            Median radius of detected blobs (NaN if none).
        bubble_data : pd.DataFrame
            DataFrame with one row per detected blob, each containing:
            - 'bubble_number' (int)
            - 'center_x' (float)
            - 'center_y' (float)
            - 'radius' (float)
            - 'area' (float)
            - 'bbox_xmin' (float)
            - 'bbox_ymin' (float)
            - 'bbox_xmax' (float)
            - 'bbox_ymax' (float)
    """
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # type: ignore[call-overload]
    if image is None:
        raise FileNotFoundError(f"Unable to read image file: {img_path}")

    # the parameters for ``SimpleBlobDetector`` were determined manually
    # by hand-tuning via visual inspection NOT by using a standardized
    # hyperparameter optimization method (see: issue #13)
    params = cv2.SimpleBlobDetector_Params() # type: ignore[attr-defined]
    params.minThreshold = 10
    params.maxThreshold = 200
    params.thresholdStep = 10
    params.filterByColor = False
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 50000
    params.filterByCircularity = True
    params.minCircularity = 0.75
    params.filterByConvexity = True
    params.minConvexity = 0.80
    params.filterByInertia = True
    params.minInertiaRatio = 0.75

    detector = cv2.SimpleBlobDetector_create(params) # type: ignore[attr-defined]
    keypoints = detector.detect(image)

    bubble_data = pd.DataFrame(index=range(len(keypoints)),
        columns=[
            "bubble_number",
            "center_x",
            "center_y",
            "radius",
            "area",
            "bbox_xmin",
            "bbox_ymin",
            "bbox_xmax",
            "bbox_ymax",
        ]
    )
    for idx, kp in enumerate(keypoints):
        cx, cy = kp.pt
        r = kp.size / 2.0
        bubble_data_row = {
            "bubble_number": idx + 1,
            "center_x": cx,
            "center_y": cy,
            "radius": r,
            "area": np.pi * r**2,
            "bbox_xmin": cx - r,
            "bbox_ymin": cy - r,
            "bbox_xmax": cx + r,
            "bbox_ymax": cy + r,
        }
        bubble_data.loc[idx] = pd.Series(bubble_data_row)

    num_blobs = len(keypoints)
    median_radius = float(np.nanmedian(bubble_data["radius"]))
    return num_blobs, median_radius, bubble_data

def _save_debug_overlay(
    img_path: Path,
    bubble_data: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Create and save a side-by-side figure containing the original
    image next to the image overlaid with opencv segmentation mask

    Parameters
    ----------
    img_path : Path
        File path to the original image.
    bubble_data : pd.DataFrame
        DataFrame of bubble metadata as returned by _detect_single_image.
    out_dir : Path
        Directory where the debug PNG will be saved.
    """
    image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # type: ignore[call-overload]

    if image_gray is None:
        raise FileNotFoundError(f"Could not read image for debug overlay: {img_path}")
    
    image_rgb = skimage.color.gray2rgb(image_gray)

    overlay = image_rgb.copy()

    for index, bubble in bubble_data.iterrows():
        x_min, y_min, x_max, y_max = map(
            int, bubble[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]]
        )
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].imshow(image_rgb)
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(overlay)
    ax[1].set_title("Detected blobs")
    ax[1].axis("off")

    png_name = f"{img_path.stem}_debug.png"
    fig.savefig(out_dir / png_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

def run_opencv(
    df: pd.DataFrame,
    out_dir: Path,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Detect bubbles in every image referenced by df using the
    OpenCV ``SimpleBlobDetector``.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing absolute image filepaths.
    out_dir : Path
        Path to save the outputs.
    debug : bool
        If True save side by side diagnostic images.

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of df enriched with: 
            num_blobs_opencv number of blobs detected
            median_radii_opencv median droplet radius
    """
    if 'image_filepath' not in df.columns:
        raise ValueError("DataFrame must contain 'image_filepath' column.")
    
    if not out_dir.is_absolute():
        raise ValueError(
            f"Absolute file path required, got {out_dir}"
        )
    
    out_dir.mkdir(parents=True, exist_ok=True)

    memory = Memory(location=out_dir / ".joblib_cache", verbose=0)
    cached_detect = memory.cache(_detect_single_image)

    df_out = df.copy()
    df_out[["num_blobs_opencv", "median_radii_opencv"]] = np.nan
   
    for idx, row in tqdm(
        df_out.iterrows(),
        total=df_out.shape[0],
        desc="OpenCV SimpleBlobDetector",
    ):
        img_path = row.image_filepath

        num_blobs, median_r, bubble_data = cached_detect(img_path)

        df_bubbles = pd.DataFrame(bubble_data)
        df_bubbles.to_parquet(
            str(out_dir / f"{img_path.stem}_bubble_data.parquet.gzip"),
            compression="gzip")

        if debug:
            _save_debug_overlay(img_path, bubble_data, out_dir)

        df_out.loc[idx, "num_blobs_opencv"] = num_blobs  # type: ignore[index]
        df_out.loc[idx, "median_radii_opencv"] = median_r  # type:ignore[index]

    return pd.DataFrame(df_out)
