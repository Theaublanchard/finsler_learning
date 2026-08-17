import torch
from pathlib import Path
from torch import Tensor
import matplotlib.pyplot as plt
from torchvision.io import read_image, write_png
from torchvision.transforms.v2 import Resize, InterpolationMode
import tqdm
import pandas as pd

TARGET_SIZE = (256, 256)


def apply_gamma_correction(img: Tensor, gamma: float) -> Tensor:
    """Apply gamma correction to an image tensor.

    Args:
        img: Tensor of shape (C, H, W) uint8 with values in [0, 255].
        gamma: Gamma value for correction.

    Returns:
        Gamma-corrected image tensor of shape (C, H, W) uint8 with values in [0, 255].
    """
    img_float = img.float() / 255.0  # Normalize to [0, 1]
    img_corrected = torch.pow(img_float, gamma)  # Apply gamma correction
    img_corrected = (img_corrected * 255).clamp(0, 255)
    return img_corrected.to(torch.uint8)


def main():
    ds_path = Path("/home/tblanchard/phd/hdd/raw_data/komatsuna")
    img_path = ds_path / "multi_plant"
    label_path = ds_path / "multi_label"
    # GAMMA = 0.9  # Gamma correction because the images are too bright

    export_path = Path("/hdd/processed_data/komatsuna")

    pbar = list(img_path.glob("*.png"))
    pbar = tqdm.tqdm(pbar)
    rows = []
    for img_p in pbar:
        img_filename = img_p.stem

        _, idk_id, plant_id, day_id, time_id = img_filename.split("_")

        label_filename = img_filename.replace("rgb", "label")
        label_p = label_path / f"{label_filename}.png"

        img = read_image(img_p)  # (3, H, W)
        # img = apply_gamma_correction(img, GAMMA)
        label = read_image(label_p)  # (3, H, W)
        label_bw = (label != 0).any(dim=0)  # (H,W)
        label_bw = label_bw.to(torch.uint8) * 255

        img = Resize(TARGET_SIZE)(img)  # (3, H', W')
        label_bw = Resize(TARGET_SIZE, interpolation=InterpolationMode.NEAREST)(
            label_bw.unsqueeze(0)
        ).squeeze(
            0
        )  # (H',W')
        # Remap to 3 channels to export as PNG
        label_bw = label_bw.unsqueeze(0).repeat(3, 1, 1)  # (3, H', W')

        # new_img_p = export_path / f"{img_filename.replace("rgb","")}.png"
        # new_label_p = export_path / f"{label_filename.replace("label","")}.png"
        new_img_p = export_path / "img" / f"{img_filename}.png"
        new_label_p = export_path / "label" / f"{label_filename}.png"

        new_img_p.parent.mkdir(exist_ok=True, parents=True)
        new_label_p.parent.mkdir(exist_ok=True, parents=True)

        write_png(img, new_img_p)
        write_png(label_bw, new_label_p)

        row = pd.DataFrame(
            {
                "idk_id": [idk_id],
                "plant_id": [plant_id],
                "day_id": [day_id],
                "time_id": [time_id],
                "img_path": [str(new_img_p)],
                "label_path": [str(new_label_p)],
                "area": float(label_bw.sum() / 255) / (TARGET_SIZE[0] * TARGET_SIZE[1]),
            }
        )
        rows.append(row)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(export_path / "metadata.csv", index=False)


if __name__ == "__main__":
    main()
