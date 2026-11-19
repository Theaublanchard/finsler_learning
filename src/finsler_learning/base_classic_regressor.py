import logging
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology

LOGGER = logging.getLogger(__name__)


def get_brain_no_ellipsis_mask(
    img: np.array, background_value: int = 0, margin: float = 0.1
) -> np.array:
    """
    Return a binary mask indicating the skull area.

    Parameters:
    img (np.array): Input image tensor of shape (H, W).
    background_value (int): Pixel value representing the background.

    Returns:
    np.array: Binary mask of shape (H, W) indicating the inside skull area excluding the ellipsis.
    """
    return img >= (background_value + margin)


def inside_skull(skull_mask: np.array) -> np.array:
    """
    Return a binary mask indicating the area inside the skull.
    The skull is assumed to be an ellipsis shape.
    Thus we actually fit an ellipsis to the skull mask to close it if needed.

    Parameters:
    skull_mask (np.array): Binary mask indicating the skull area.

    Returns:
    np.array: Binary mask of shape (H, W) indicating area inside the skull.
    """
    return binary_fill_holes(skull_mask)


def ellipsis_from_mask(
    img: np.array, inside_skull_mask: np.array, skull_mask: np.array, margin: float = 0.1
) -> np.array:
    """
    Get the ellipsis mask from the image inside the skull.
    The ellipsis is defined as the area inside the skull
    where the pixel values are less than or equal to the margin.

    Parameters:
    img (np.array): Input image tensor of shape (H, W).
    inside_skull_mask (np.array): Binary mask indicating area inside the skull.
    skull_mask (np.array): Binary mask indicating the skull area.
    margin (float): Margin value to define the ellipsis area.

    Returns:
    np.array: Binary mask of shape (H, W) indicating the ellipsis area.
    """
    ellipsis_mask = (img <= margin) & inside_skull_mask & (~skull_mask)
    return ellipsis_mask


def get_leftmost_ellipse_mask(mask: np.array, disk_size=12) -> np.array:
    """
    Get the leftmost ellipse mask from the image.
    Assuming two ellipses are present in the image
    Get the connected components and select the leftmost one.

    Parameters:
    mask (np.array): Binary mask of shape (H, W).
    disk_size (int): Size of the disk for erosion and dilation.
    Returns:
    np.array: Binary mask of shape (H, W) indicating the leftmost ellipse.
    """
    if not mask.any():
        raise ValueError("Input mask is empty")
    selem = morphology.disk(disk_size)
    # sometimes the ellipses are connected
    mask = morphology.erosion(mask, selem)

    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask, cache=False)
    if len(regions) == 0:
        raise ValueError("No regions found in the mask to extract the leftmost ellipse.")
    leftmost_region = min(
        regions, key=lambda r: r.bbox[1]
    )  # bbox = (min_row, min_col, max_row, max_col)
    leftmost_mask = labeled_mask == leftmost_region.label

    # Dilate back to original size
    leftmost_mask = morphology.dilation(leftmost_mask, selem)
    return leftmost_mask


def get_ellipsis_area(mask: np.array) -> float:
    """
    Fit an ellipse to the given binary mask.
    Return the area of the fitted ellipse
    Parameters:
    mask (np.array): Binary mask of shape (H, W).
    img_size (int): Size of the image (assumed square).
    """

    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask, cache=False)
    if len(regions) == 0:
        raise ValueError("No regions found in the mask to fit an ellipse.")
    elif len(regions) > 1:
        # Pick the largest region
        regions = sorted(regions, key=lambda r: r.area, reverse=True)
    region = regions[0]
    a = region.axis_major_length / 2
    b = region.axis_minor_length / 2
    area = np.pi * a * b
    return area


def regress_single_image(
    img: np.ndarray,
    img_size: int = 256,
    norm_by_skull: bool = False,
    disk_size: int = 12,
    return_valid: bool = False,
) -> float:
    """
    Process a single image to extract the leftmost ellipse area.

    Parameters:
    img (np.ndarray): Input image of shape (H, W).
    img_size (int): Size of the image (assumed square).
    norm_by_skull (bool): Whether to normalize the area by the skull area.
    disk_size (int): Size of the disk for erosion and dilation.
    return_valid (bool): Whether to return a validity flag along with the area.

    Returns:
    float: Normalized area of the leftmost ellipse.
    """
    try:
        skull_mask = get_brain_no_ellipsis_mask(img, background_value=0, margin=0.1)
        inside_skull_mask = inside_skull(skull_mask)
        ellipsis_mask = ellipsis_from_mask(img, inside_skull_mask, skull_mask)
        left_ellipsis = get_leftmost_ellipse_mask(ellipsis_mask, disk_size=disk_size)
        fitted_ellipse_area = get_ellipsis_area(left_ellipsis)
        if norm_by_skull:
            skull_area = get_ellipsis_area(inside_skull_mask)
            fitted_ellipse_area /= skull_area
        else:
            fitted_ellipse_area = fitted_ellipse_area / (img_size * img_size)
        if return_valid:
            return fitted_ellipse_area, True
        return fitted_ellipse_area
    except ValueError:
        # LOGGER.error(f"Error processing image: {e}")
        if return_valid:
            return 0.0, False
        return 0.0


def regressor(
    imgs: torch.Tensor,
    norm_by_skull: bool = False,
    disk_size: int = 10,  # nice value for img_size=256
    return_valid: bool = False,
) -> torch.Tensor:
    """
    Regressor function to compute the normalized area of the leftmost ellipse.

    Parameters:
    imgs (torch.Tensor): Input image tensor of shape (B,C, H, W).
    norm_by_skull (bool): Whether to normalize the area by the skull area.
    disk_size (int): Size of the disk for erosion and dilation.
    return_valid (bool): Whether to return a validity flag along with the area.

    Returns:
    area (torch.Tensor): Normalized area of the leftmost ellipse. (B,)
    valid (torch.Tensor, optional): Validity flags for each area. Returned if return_valid is True.
    """
    area = torch.zeros(imgs.shape[0])
    if return_valid:
        valid = torch.zeros(imgs.shape[0], dtype=torch.bool)
    imgs = imgs[:, 0].to("cpu").numpy()
    img_size = imgs.shape[1]
    for i in range(imgs.shape[0]):
        img = imgs[i]
        data = regress_single_image(
            img,
            img_size,
            norm_by_skull=norm_by_skull,
            disk_size=disk_size,
            return_valid=return_valid,
        )
        if return_valid:
            area[i], valid[i] = data
        else:
            area[i] = data
    if return_valid:
        return area, valid
    return area


def regressor_parallel(
    imgs: torch.Tensor,
    norm_by_skull: bool = False,
    disk_size: int = 10,  # nice value for img_size=256
    n_workers: int = 8,
    return_valid: bool = False,
) -> torch.Tensor:
    """
    Parallel version of regressor for faster batch processing.

    Parameters:
    imgs (torch.Tensor): Input image tensor of shape (B, C, H, W).
    img_size (int): Size of the image (assumed square).
    norm_by_skull (bool): Whether to normalize the area by the skull area.
    disk_size (int): Size of the disk for erosion and dilation.
    n_workers (int): Number of parallel workers.
    return_valid (bool): Whether to return a validity flag along with the area.

    Returns:
    torch.Tensor: Normalized area of the leftmost ellipse of shape (B,).
    torch.Tensor, optional: Validity flags for each area. Returned if return_valid is True.
    """
    imgs_cpu = imgs[:, 0].cpu().numpy()
    img_size = imgs.shape[2]
    process_func = partial(
        regress_single_image,
        img_size=img_size,
        norm_by_skull=norm_by_skull,
        disk_size=disk_size,
        return_valid=return_valid,
    )
    with Pool(n_workers) as pool:
        data = pool.map(process_func, imgs_cpu)
    if return_valid:
        areas, valids = zip(*data)
        return torch.tensor(areas, dtype=torch.float32), torch.tensor(valids, dtype=torch.bool)
    else:
        areas = data
        return torch.tensor(areas, dtype=torch.float32)


# Construct a class for compatibility with the rest of the codebase
class ClassicEllipseAreaRegressor(torch.nn.Module):
    def __init__(self, norm_by_skull: bool = False, disk_size: int = 10, n_workers: int = 8):
        super().__init__()
        self.norm_by_skull = norm_by_skull
        self.disk_size = disk_size
        self.n_workers = n_workers

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return regressor_parallel(
            imgs,
            norm_by_skull=self.norm_by_skull,
            disk_size=self.disk_size,
            n_workers=self.n_workers,
        ).to(imgs.device)


class HybridRegressor(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        norm_by_skull: bool = False,
        disk_size: int = 10,
        n_workers: int = 8,
    ):
        super().__init__()
        self.norm_by_skull = norm_by_skull
        self.disk_size = disk_size
        self.n_workers = n_workers
        self.base_model = base_model

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        areas, valid_areas = regressor_parallel(
            imgs,
            norm_by_skull=self.norm_by_skull,
            disk_size=self.disk_size,
            n_workers=self.n_workers,
            return_valid=True,
        )
        if valid_areas.all():
            return areas.to(imgs.device)
        else:
            wrong_imgs = imgs[~valid_areas]
            if wrong_imgs.shape[0] > 0:
                model_areas = self.base_model(wrong_imgs).squeeze()
                areas[~valid_areas] = model_areas.to("cpu").to(areas.dtype)
            return areas.to(imgs.device)
