import torch
import numpy as np
import pandas as pd
import random
import logging
from pathlib import Path
from tqdm import tqdm

from torchvision.io import read_image
from torchvision.transforms.v2 import ToTensor
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse
from nibabel import load as load_nifti
from json import load as json_load

from monai.transforms import LoadImage, EnsureChannelFirst, Spacing
from monai.transforms import (
    Resize,
    ToTensor,
    Compose,
    NormalizeIntensity,
    Identity,
)

from .models import BaseVAE

logger = logging.getLogger()


class SheppLoganDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_size: int,
        num_samples: int,
        smoothing: bool = False,
        transform=None,
        key: str = "ventricle1",
        normalize_area: bool = True,
        export_path: Path = None,
        train: bool = True,
        label_noise: float = 0.0,
    ):
        """
        Shepp-Logan phantom dataset.

        Parameters:
        -----------
        key : str
            Key to use as label. Must be in
            ['skull', 'brain', 'central', 'top','bottom1','bottom2','bottom3','ventricle1','ventricle2']
        img_size : int
            Size of the images (img_size, img_size)
        num_samples : int
            Number of samples to generate
        smoothing : bool
            Whether to apply Gaussian smoothing to the images
        transform : callable
            Transformation to apply to the data
        key : str
            Key to use as label. Must be in
            ['skull', 'brain', 'central', 'top','bottom1','bottom2','bottom3','ventricle1','ventricle2']
        normalize_area : bool
            Whether to normalize the area of the selected key by the area of the brain
        export_path : Path
            Path to save the generated dataset. If None, do not save the dataset
        train : bool
            Whether to generate the train or test set.
        label_noise : float
            Amount of noise to add to the labels as normal distribution. Default is 0.0 (no noise).
            The amount correspond to percetnages of the standard deviation of the labels. 
            For example, if label_noise=0.1, then the noise added to the labels will be sampled from a normal distribution with mean 0 and standard deviation 0.1 * std(labels).
        """
        self.img_size = img_size
        self.num_samples = num_samples
        self.transform = transform
        self.smoothing = smoothing
        self.key = key
        self.normalize_area = normalize_area
        self.export_path = export_path
        self.train = train
        self.label_noise = label_noise

        data_exists = self.check_data_exists(export_path)
        if data_exists:
            if self.train:
                export_path_train = export_path / "train_samples.pth"
                self._load_data(export_path_train)
            else:
                export_path_test = export_path / "test_samples.pth"
                self._load_data(export_path_test)
            logger.info(
                f"Dataset (N={len(self.y)}) already exists at {export_path}. Loading it."
            )
        else:
            logger.info(
                f"{'Train' if self.train else 'Test'} Dataset does not exist at {export_path}. Generating it."
            )
            self.create_dataset(export_path, label_noise=self.label_noise)

    @staticmethod
    def generate_scales(size: str) -> tuple[float, float]:
        if size == "large":
            return random.uniform(0.9, 1.3), random.uniform(0.9, 1.3)
        elif size == "small":
            return random.uniform(0.7, 0.9), random.uniform(0.7, 0.9)
        else:
            raise NotImplementedError(f"Size {size} was not implemented for variable sizes.")

    @staticmethod
    def generate_shepplogan_phantom(
        img_size: int, label: int = 0, smoothing: bool = True
    ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Generate 2D Shepp-Logan phantom with random regions size. Phantoms also
        simulate different kind of AD by generating smaller ROIs.

        Args:
            img_size: Size of the generated image (img_size x img_size).
            label: Take 0 or 1 or 2. Label of the generated image.
                If 0, the ROIs simulate a CN subject.
                If 1, the ROIs simulate type 1 of AD.
                if 2, the ROIs simulate type 2 of AD.
            smoothing: Default True. Apply Gaussian smoothing to the image.
        Returns:
            img: 2D Sheep Logan phantom with specified label.
            areas: Dictionary with the area of each region.
        """
        img = np.zeros((img_size, img_size))
        areas = {}
        center = (img_size + 1.0) / 2.0
        a = (center - 2) * random.uniform(0.8, 1.0)
        b = (center * 2 / 3 - 2) * random.uniform(0.8, 1.0)

        color = random.uniform(0.4, 0.6)

        if label == 0:
            roi1, roi2 = "large", "large"
        elif label == 1:
            roi1, roi2 = "large", "small"
        elif label == 2:
            roi1, roi2 = "small", "large"
        else:
            raise NotImplementedError(f"Subtype {label} was not implemented.")

        # Skull
        rr, cc = ellipse(center, center, a, b, (img_size, img_size))
        img[rr, cc] = 1
        areas["skull"] = np.pi * a * b

        # Brain
        offset = random.uniform(1, img_size / 32)
        rr, cc = ellipse(
            center + offset / 2, center, a - offset, b - offset, (img_size, img_size)
        )
        img[rr, cc] = 0.2
        areas["brain"] = np.pi * (a - offset) * (b - offset)

        # Central
        offset1 = random.uniform(1, img_size / 32)
        offset2 = random.uniform(1, img_size / 32)
        scale1, scale2 = SheppLoganDataset.generate_scales("large")
        phi = random.uniform(-np.pi, np.pi)
        a_central = b / 6 * scale1
        b_central = b / 6 * scale2
        rr, cc = ellipse(
            center + offset1,
            center + offset2,
            a_central,
            b_central,
            (img_size, img_size),
            rotation=phi,
        )
        img[rr, cc] = color
        areas["central"] = np.pi * a_central * b_central

        # ROI 1 top
        offset1 = random.uniform(1, img_size / 32)
        offset2 = random.uniform(1, img_size / 32)
        scale1, scale2 = SheppLoganDataset.generate_scales(roi1)
        phi = random.uniform(-np.pi, np.pi)
        a_top = b / 3 * scale1
        b_top = b / 3.5 * scale2
        rr, cc = ellipse(
            center * 0.6 + offset1,
            center + offset2,
            a_top,
            b_top,
            (img_size, img_size),
            rotation=phi,
        )
        img[rr, cc] = color
        areas["top"] = np.pi * a_top * b_top

        # ROI 2 multiple bottom
        offset1 = random.uniform(1, img_size / 32)
        offset2 = random.uniform(1, img_size / 32)
        scale1, scale2 = SheppLoganDataset.generate_scales(roi2)
        phi = random.uniform(-np.pi, np.pi)
        rr, cc = ellipse(
            center * 1.5 + offset1,
            center + offset2,
            b / 10 * scale1,
            b / 10 * scale2,
            (img_size, img_size),
            rotation=phi,
        )
        img[rr, cc] = color
        areas["bottom1"] = np.pi * (b / 10 * scale1) * (b / 10 * scale2)

        offset1 = random.uniform(1, img_size / 32)
        offset2 = random.uniform(1, img_size / 32)
        scale1, scale2 = SheppLoganDataset.generate_scales(roi2)
        phi = random.uniform(-np.pi, np.pi)
        rr, cc = ellipse(
            center * 1.5 + offset1,
            center * 1.1 + offset2,
            b / 10 * scale1,
            b / 10 * scale2,
            (img_size, img_size),
            rotation=phi,
        )
        img[rr, cc] = color
        areas["bottom2"] = np.pi * (b / 10 * scale1) * (b / 10 * scale2)

        offset1 = random.uniform(1, img_size / 32)
        offset2 = random.uniform(1, img_size / 32)
        scale1, scale2 = SheppLoganDataset.generate_scales(roi2)
        phi = random.uniform(-np.pi, np.pi)
        rr, cc = ellipse(
            center * 1.5 + offset1,
            center * 0.9 + offset2,
            b / 10 * scale1,
            b / 10 * scale2,
            (img_size, img_size),
            rotation=phi,
        )
        img[rr, cc] = color
        areas["bottom3"] = np.pi * (b / 10 * scale1) * (b / 10 * scale2)

        # Ventricle 1
        a_roi = a * random.uniform(0.8, 1.2)
        phi = np.random.uniform(-np.pi / 16, np.pi / 16)
        rr, cc = ellipse(
            center,
            center * 0.75,
            a_roi / 3,
            a_roi / 6,
            (img_size, img_size),
            rotation=np.pi / 8 + phi,
        )
        img[rr, cc] = 0.0
        areas["ventricle1"] = np.pi * (a_roi / 3) * (a_roi / 6)

        # Ventricle 2
        a_roi = a * random.uniform(0.8, 1.2)
        phi = np.random.uniform(-np.pi / 16, np.pi / 16)
        rr, cc = ellipse(
            center,
            center * 1.25,
            a_roi / 3,
            a_roi / 6,
            (img_size, img_size),
            rotation=-np.pi / 8 + phi,
        )
        img[rr, cc] = 0.0
        areas["ventricle2"] = np.pi * (a_roi / 3) * (a_roi / 6)

        # Random smoothing
        if smoothing:
            sigma = random.uniform(0, 1)
            img = gaussian_filter(img, sigma * img_size / 100.0)  # smoothing of data

        img.clip(0, 1)

        return img, areas

    def get_img_labels(self):
        """
        Generate a random Shepp-Logan phantom image and its label.
        """
        img, areas = self.generate_shepplogan_phantom(
            img_size=self.img_size, label=0, smoothing=self.smoothing
        )
        img = torch.tensor(img, dtype=torch.double).unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)
        if self.normalize_area:
            areas = {k: v / areas["skull"] for k, v in areas.items()}
        else:
            areas = {k: v / (self.img_size**2) for k, v in areas.items()}
        y = torch.tensor(areas[self.key], dtype=torch.double)
        return img, y

    def check_data_exists(self, export_path: Path):
        if export_path is None:
            return False
        export_path_train = export_path / "train_samples.pth"
        export_path_test = export_path / "test_samples.pth"
        if self.train and export_path_train.is_file():
            return True
        if not self.train and export_path_test.is_file():
            return True
        return False

    def create_dataset(self, export_path: Path, label_noise: float = 0.0):
        imgs, labels = self._generate_fold()

        if label_noise > 0.0:
            std = labels.std()
            noise = torch.randn_like(labels) * label_noise * std
            labels += noise

        self.x = imgs
        self.y = labels

        if export_path is None:
            return

        if self.train:
            export_path_train = export_path / "train_samples.pth"
            torch.save((imgs, labels), export_path_train)
            logger.info(f"Train dataset saved at {export_path_train}.")
        else:
            export_path_test = export_path / "test_samples.pth"
            torch.save((imgs, labels), export_path_test)
            logger.info(f"Test dataset saved at {export_path_test}.")
            
    def _generate_fold(self):
        imgs, labels = [], []
        for i in tqdm(range(self.num_samples)):
            img, label = self.get_img_labels()
            imgs.append(img)
            labels.append(label)
        imgs = torch.stack(imgs)
        labels = torch.stack(labels)
        return imgs, labels

    def _load_data(self, export_path: Path):
        imgs, labels = torch.load(
            export_path,
            weights_only=False,
            map_location=torch.device("cpu"),
        )
        self.x = imgs
        self.y = labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        Generate a random Shepp-Logan phantom image and its label.
        """
        if self.export_path is None:
            x, y = self.get_img_labels()
            if self.label_noise > 0.0:
                noise = torch.randn_like(y) * self.label_noise
                y += noise
            return x, y
        else:
            x, y = self.x[index], self.y[index]
            return x, y


class VAESheppLoganDataset(SheppLoganDataset):
    def __init__(
        self,
        img_size: int,
        num_samples: int,
        vae_model: BaseVAE,
        smoothing: bool = False,
        transform=None,
        key: str = "ventricle1",
        normalize_area: bool = True,
        export_path: Path = None,
        train: bool = True,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Shepp-Logan phantom dataset that randomly uses the reconstructed image from a VAE.

        Parameters:
        -----------
        key : str
            Key to use as label. Must be in
            ['skull', 'brain', 'central', 'top','bottom1','bottom2','bottom3','ventricle1','ventricle2']
        img_size : int
            Size of the images (img_size, img_size)
        num_samples : int
            Number of samples to generate
        smoothing : bool
            Whether to apply Gaussian smoothing to the images
        transform : callable
            Transformation to apply to the data
        key : str
            Key to use as label. Must be in
            ['skull', 'brain', 'central', 'top','bottom1','bottom2','bottom3','ventricle1','ventricle2']
        normalize_area : bool
            Whether to normalize the area of the selected key by the area of the brain
        export_path : Path
            Path to save the generated dataset. If None, do not save the dataset
        train : bool
            Whether to generate/get the train or test set.
        vae_model : torch.nn.Module
            VAE model to use for reconstruction.
        device : torch.device
            Device to use for the VAE model and data.
        """
        super().__init__(
            img_size,
            num_samples,
            smoothing,
            transform,
            key,
            normalize_area,
            export_path,
            train,
        )
        self.vae_model = vae_model
        self.device = device
        self.vae_model.eval()
        self.vae_model.to(self.device)

        self.vae_prob = 0.5  # Probability of using the VAE reconstruction

        def __getitem__(self, index):
            data = super().__getitem__(index)
            x, y = data

            if torch.rand(1).item() < self.vae_prob:
                x = x.to(self.device).unsqueeze(0)  # (1,1,H,W)
                recon_x, _, _ = self.vae_model(x)
                x = recon_x.squeeze(0).cpu()  # (1,H,W)
            return x, y


class ACDCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: Path = Path("/home/tblanchard/phd/hdd/processed_data/ACDC"),
        train: bool = True,
        transform=None,
        target_spacing=(1.5, 1.5, 10),
        target_shape=(80, 80, 10),
        use_base_transform: bool = True,
        return_dummy_label: bool = False,
        return_nii_infos: bool = False,
        mean: float = None,
        std: float = None,
    ):
        super().__init__()
        self.train = train
        self.transform = transform
        self.target_spacing = target_spacing
        self.target_shape = target_shape
        self.return_dummy_label = return_dummy_label
        self.return_nii_infos = return_nii_infos
        if self.train:
            self.dataset_path = dataset_path / "imagesTr"
            self.labels_path = dataset_path / "labelsTr"
            self.volumes_infos = self.labels_path / "labels_train.json"
        else:
            self.dataset_path = dataset_path / "imagesTs"
            self.labels_path = dataset_path / "labelsTs"
            self.volumes_infos = self.labels_path / "labels_test.json"
        self.image_paths = sorted(list(self.dataset_path.glob("*.nii.gz")))

        with open(self.volumes_infos, "r") as f:
            self.volumes_infos = json_load(f)

        if use_base_transform:
            if mean is not None and std is not None:
                norm = NormalizeIntensity(subtrahend=mean, divisor=std)
            elif (mean is not None) != (std is not None):
                raise ValueError("Both mean and std must be provided for normalization.")
            else:
                norm = Identity()
            self.base_transform = Compose(
                [
                    LoadImage(image_only=True),
                    EnsureChannelFirst(),
                    # Orientation(axcodes="RAS"),
                    Spacing(pixdim=self.target_spacing, mode="bilinear", align_corners=True),
                    Resize(
                        spatial_size=self.target_shape, mode="bilinear", align_corners=True
                    ),
                    norm,
                    ToTensor(dtype=torch.double),
                ]
            )
        else:
            self.base_transform = LoadImage(image_only=True)
        if self.transform is not None:
            self.base_transform = Compose([self.base_transform, self.transform])

    def __len__(self):
        return len(self.image_paths)

    def fetch_images(self, img_path: Path):
        img = self.base_transform(img_path)
        file_name = img_path.name.replace(".nii.gz", "")
        LV_vol_ml = self.volumes_infos[file_name]["LV_volume_ml"]
        voxel_volume = self.volumes_infos[file_name]["voxel_volume_ml"]
        return img, LV_vol_ml, voxel_volume

    def fetch_image_and_mask(self, img_path: Path):
        mask_path = self.labels_path / img_path.name
        img = self.base_transform(img_path)
        mask = self.base_transform(mask_path)
        return img, mask

    def __getitem__(self, index):
        img_path: Path = self.image_paths[index]
        img, LV_vol_ml, voxel_volume = self.fetch_images(img_path)
        if self.return_nii_infos:
            nii = load_nifti(str(img_path))
            nii_infos = {
                "affine": nii.affine,
                "header": nii.header,
            }
            return img, LV_vol_ml, nii_infos
        return img, LV_vol_ml


class EllipsisDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_samples: int,
        img_size: int,
        transform=None,
        return_area: bool = False,
        train: bool = True,
        export_path: Path = None,
        circle_only: bool = False,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform
        self.return_area = return_area
        self.train = train
        self.export_path = export_path
        self.circle_only = circle_only

        self._process_data(export_path)

    def __len__(self):
        return self.num_samples

    def generate_ellipsis(self):
        a = random.uniform(self.img_size / 8, self.img_size / 2)
        if self.circle_only:
            b = a
        else:
            b = random.uniform(self.img_size / 8, self.img_size / 2)
        phi = random.uniform(0, 180)
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        rr, cc = ellipse(
            self.img_size / 2,
            self.img_size / 2,
            a,
            b,
            rotation=np.deg2rad(phi),
            shape=img.shape,
        )
        img[rr, cc] = 1.0
        img = torch.tensor(img).unsqueeze(0)  # (1, H, W)
        area = np.pi * a * b
        return img.to(torch.float32), area

    def _process_data(self, export_path: Path):
        logger = logging.getLogger()
        if export_path is None:
            return
        export_path_train = export_path / "train_samples.pth"
        export_path_test = export_path / "test_samples.pth"
        if self.train and export_path_train.is_file():
            self._load_data(export_path_train)
            logger.info(
                f"Train dataset (N={len(self.areas)}) already exists at {export_path_train}. Loading it."
            )
            return
        if not self.train and export_path_test.is_file():
            self._load_data(export_path_test)
            logger.info(
                f"Test dataset (N={len(self.areas)}) already exists at {export_path_test}. Loading it."
            )
            return
        logger.info("Generating dataset...")
        images, areas = [], []
        for _ in tqdm(range(self.num_samples)):
            img, area = self.generate_ellipsis()
            images.append(img)
            areas.append(area)
        images = torch.stack(images)
        areas = torch.tensor(areas, dtype=torch.float32)
        if self.train:
            torch.save((images, areas), export_path_train)
            logger.info(f"Train dataset saved at {export_path_train}.")
        else:
            torch.save((images, areas), export_path_test)
            logger.info(f"Test dataset saved at {export_path_test}.")

        self.images = images
        self.areas = areas

    def _load_data(self, export_path: Path):
        images, areas = torch.load(
            export_path,
            weights_only=False,
            map_location=torch.device("cpu"),
        )
        self.images = images
        self.areas = areas

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.export_path is None:
            img, area = self.generate_ellipsis()
        else:
            img, area = self.images[index], self.areas[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.return_area:
            return img.to(torch.float64), area
        return img.to(torch.float64)


class HC18Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: Path = Path("/home/tblanchard/phd/hdd/processed_data/HC18"),
        folds: str = "train",
        transform=None,
        export_path: Path = None,
        val_frac: float = 0.2,
        return_annots: bool = False,
    ):
        super().__init__()
        assert folds in [
            "train",
            "val",
            "test",
        ], f"Folds must be one of ['train', 'val', 'test'], got {folds}"

        if return_annots and transform is not None:
            raise ValueError("Cannot apply transform when return_annots is True.")

        self.dataset_path = dataset_path
        self.folds = folds
        self.transform = transform
        self.return_annots = return_annots

        if self.folds == "train" or self.folds == "val":
            self.df_info = self.load_and_split_dataset(1 - val_frac, self.folds, export_path)
        else:
            self.df_info = pd.read_csv(dataset_path / "test_set_processed.csv")

    def load_data_(self, folds, save_folder: Path):
        logger = logging.getLogger()
        df_infos = pd.read_csv(self.dataset_path / "train_set_processed.csv")
        if folds == "train":
            idx = save_folder / "train_indices.npy"
        elif folds == "val":
            idx = save_folder / "val_indices.npy"
        else:
            raise ValueError(f"Folds must be one of ['train', 'val'], got {folds}")
        if idx.is_file():
            logger.info(f"Loading indices from {idx} ")
            idx = np.load(idx, allow_pickle=True)
            df_info = df_infos.iloc[idx]
        else:
            df_info = None
        return df_info

    def load_and_split_dataset(self, train_frac: float, folds: str, save_folder: Path = None):
        logger = logging.getLogger()

        df_info = self.load_data_(folds, save_folder)
        if df_info is not None:
            return df_info

        logger.info(f"No saved indices found at {save_folder}. Splitting dataset.")
        df_info = pd.read_csv(self.dataset_path / "train_set_processed.csv")

        n_samples = len(df_info)
        n_train = int(train_frac * n_samples)
        indices = np.random.permutation(n_samples)
        train_indices, val_indices = indices[:n_train], indices[n_train:]

        if save_folder is not None:
            # Save the indices for reproducibility
            train_path = save_folder / "train_indices.npy"
            val_path = save_folder / "val_indices.npy"
            np.save(train_path, train_indices)
            np.save(val_path, val_indices)
            logger.info(
                f"Saved {len(train_indices)} train indices to {train_path} and {len(val_indices)} validation indices to {val_path}"
            )

        if folds == "train":
            return df_info.iloc[train_indices]
        elif folds == "val":
            return df_info.iloc[val_indices]
        else:
            raise ValueError(f"Folds must be one of ['train', 'val'], got {folds}")

    def __len__(self):
        return len(self.df_info)

    def __getitem__(self, index):
        row = self.df_info.iloc[index]
        img = read_image(str(row["path"])).float() / 255.0  # (C, H, W)

        if self.folds == "train" or self.folds == "val":
            label = torch.tensor(row["circumference (px)"], dtype=torch.float64)
        else:
            label = torch.tensor(0, dtype=torch.float64)

        if self.return_annots:
            annot = read_image(str(row["path_annot"]))  # (C, H, W)
            if self.transform is not None:
                img = self.transform(img)
            return img.double(), annot

        if self.transform is not None:
            img = self.transform(img)

        return img.double(), label


class KOMATSUNADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: Path = Path("/home/tblanchard/phd/hdd/processed_data/KOMATSUNA"),
        train: bool = True,
        transform=None,
        export_path: Path = None,
        val_frac: float = 0.2,
        return_annots: bool = False,
    ):
        super().__init__()

        if return_annots and transform is not None:
            raise ValueError("Cannot apply transform when return_annots is True.")

        self.dataset_path = dataset_path
        self.train = train
        self.transform = transform
        self.return_annots = return_annots

        self.df_info = self.load_and_split_dataset(1 - val_frac, export_path)

    def load_data_(self, save_folder: Path):
        logger = logging.getLogger()
        df_infos = pd.read_csv(self.dataset_path / "metadata.csv")

        if save_folder is None:
            return None

        if self.train:
            idx = save_folder / "train_indices.npy"
        else:
            idx = save_folder / "test_indices.npy"
        if idx.is_file():
            logger.info(f"Loading indices from {idx} ")
            idx = np.load(idx, allow_pickle=True)
            df_info = df_infos.iloc[idx]
        else:
            df_info = None
        return df_info

    def load_and_split_dataset(self, train_frac: float, save_folder: Path = None):
        logger = logging.getLogger()

        df_info = self.load_data_(save_folder)
        if df_info is not None:
            return df_info

        logger.info(f"No saved indices found at {save_folder}. Splitting dataset.")
        df_info = pd.read_csv(self.dataset_path / "metadata.csv")

        n_samples = len(df_info)
        n_train = int(train_frac * n_samples)
        indices = np.random.permutation(n_samples)
        train_indices, test_indices = indices[:n_train], indices[n_train:]

        if save_folder is not None:
            # Save the indices for reproducibility
            train_path = save_folder / "train_indices.npy"
            test_path = save_folder / "test_indices.npy"
            np.save(train_path, train_indices)
            np.save(test_path, test_indices)
            logger.info(
                f"Saved {len(train_indices)} train indices to {train_path} and {len(test_indices)} test indices to {test_path}"
            )

        if self.train:
            return df_info.iloc[train_indices]
        else:
            return df_info.iloc[test_indices]

    def __len__(self):
        return len(self.df_info)

    def read_row(self, row: pd.Series):
        img = read_image(str(row["img_path"])).float() / 255.0  # (C, H, W)

        label = torch.tensor(row["area"], dtype=torch.float64)

        if self.return_annots:
            annot = read_image(str(row["label_path"]))  # (C, H, W)
            annot = (annot != 0).any(dim=0).float()  # (H, W), binary mask
            if self.transform is not None:
                img = self.transform(img)
            return img, annot

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __getitem__(self, index):
        return self.read_row(self.df_info.iloc[index])
