"""Extract particle masks from scatter plot images."""

import numpy as np
from PIL import Image
from loguru import logger


def find_frame_bounds(img_array: np.ndarray, border_threshold: int = 128) -> tuple[int, int, int, int]:
    """
    Find the inner bounds of the rectangular frame.
    Detects thin black frame lines by looking for dark pixels along edges.

    Returns:
        Tuple of (top, bottom, left, right) pixel coordinates.
    """
    h, w = img_array.shape

    def find_dark_line_row(start, end, step):
        """Find rows that contain a horizontal dark line (frame edge)."""
        last_frame_row = start
        for i in range(start, end, step):
            row = img_array[i, :]
            dark_mask = row < border_threshold
            max_run = 0
            current_run = 0
            for val in dark_mask:
                if val:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            if max_run > w * 0.5:
                last_frame_row = i
        return last_frame_row

    def find_dark_line_col(start, end, step):
        """Find columns that contain a vertical dark line (frame edge)."""
        last_frame_col = start
        for j in range(start, end, step):
            col = img_array[:, j]
            dark_mask = col < border_threshold
            max_run = 0
            current_run = 0
            for val in dark_mask:
                if val:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            if max_run > h * 0.5:
                last_frame_col = j
        return last_frame_col

    top = find_dark_line_row(0, h // 4, 1)
    bottom = find_dark_line_row(h - 1, h * 3 // 4, -1)
    left = find_dark_line_col(0, w // 4, 1)
    right = find_dark_line_col(w - 1, w * 3 // 4, -1)

    margin = 3
    top = top + margin
    bottom = bottom - margin
    left = left + margin
    right = right - margin

    return top, bottom, left, right


def extract_particle_mask(
    image_path: str,
    darkness_threshold: float = 2.0,
    remove_frame: bool = True,
    square: bool = True,
    resolution: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract a binary particle mask from a scatter plot image.

    Parameters:
        image_path: Path to the image file
        darkness_threshold: Minimum darkness percentage (0-100) for a pixel to be considered
            a particle. Default 10% means pixels at least 10% dark are particles.
            Lower values = stricter (fewer particles), higher = more lenient.
        remove_frame: If True, automatically detect and remove the rectangular frame
        square: If True, crop to square (center crop using the smaller dimension)
        resolution: If set, resize the mask to this resolution (e.g., 64 for 64x64)

    Returns:
        Tuple of (mask, cropped_image) where mask has 1 for particles, 0 for background.
        Returns None if the file cannot be loaded.
    """
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        logger.error(f"File not found: {image_path}")
        return None

    img_array = np.array(img)

    if remove_frame:
        top, bottom, left, right = find_frame_bounds(img_array)
        logger.info(f"Detected frame bounds: top={top}, bottom={bottom}, left={left}, right={right}")
        logger.info(f"Original size: {img_array.shape}")
        img_array = img_array[top:bottom, left:right]
        logger.info(f"Cropped size: {img_array.shape}")

    if square and img_array.shape[0] != img_array.shape[1]:
        h, w = img_array.shape
        size = min(h, w)
        # Center crop
        top_offset = (h - size) // 2
        left_offset = (w - size) // 2
        img_array = img_array[top_offset:top_offset + size, left_offset:left_offset + size]
        logger.info(f"Square crop: {size}x{size}")

    # Convert darkness_threshold percentage to grayscale value
    # 0% darkness = white (255), 100% darkness = black (0)
    # threshold% darkness means pixel value < 255 * (1 - threshold/100)
    pixel_threshold = int(255 * (1.0 - darkness_threshold / 100.0))
    logger.info(f"Darkness threshold: {darkness_threshold}% -> pixel value < {pixel_threshold}")

    # Apply threshold BEFORE resizing to preserve particle information
    mask = (img_array < pixel_threshold).astype(np.uint8)

    if resolution is not None:
        # Resize the binary mask using NEAREST to preserve particle counts
        # Also resize the grayscale image for visualization
        mask_pil = Image.fromarray(mask * 255)
        mask_pil = mask_pil.resize((resolution, resolution), Image.Resampling.NEAREST)
        mask = (np.array(mask_pil) > 127).astype(np.uint8)

        img_pil = Image.fromarray(img_array)
        img_pil = img_pil.resize((resolution, resolution), Image.Resampling.LANCZOS)
        img_array = np.array(img_pil)
        logger.info(f"Resized to: {resolution}x{resolution}")

    return mask, img_array
