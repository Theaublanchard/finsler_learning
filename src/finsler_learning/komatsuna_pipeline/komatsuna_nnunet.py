from pathlib import Path
import re
import hashlib
import shutil
import subprocess
from datetime import datetime
from json import dump, load
import torch
import torchvision
from tqdm import tqdm

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results
from nnunetv2.postprocessing.remove_connected_components import (
    apply_postprocessing,
    load_pickle,
)

from .komatsuna_utils import *

NNUNET_DS_PATH = Path("/home/tblanchard/phd/hdd/nnUnet_datasets/nnUNet_raw")


class NNUnetModel(torch.nn.Module):
    """Torch module wrapper around a trained nnUNet v2 inference pipeline.

    The wrapper resolves the exported KOMATSUNA dataset, locates the trained nnUNet
    model folder, loads the predictor, runs inference on a folder of exported
    images, and applies nnUNet postprocessing when available.
    """

    def __init__(
        self,
        export_path: Path,
        n_epochs: int = 100,
        configuration: str = "2d",
        folds: tuple[int, ...] = (0,),
        checkpoint_name: str = "checkpoint_final.pth",
        device: torch.device = DEVICE,
        # apply_postprocessing: bool = True,
        return_seg: bool = False,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.export_path = Path(export_path)
        self.n_epochs = n_epochs
        self.configuration = configuration
        self.folds = folds
        self.checkpoint_name = checkpoint_name
        self.device = device
        self.apply_postprocessing = apply_postprocessing
        self.return_seg = return_seg
        self.threshold = threshold

        self.dataset_id, self.ds_name, self.ds_path = _load_nnunet_export_link(
            self.export_path
        )
        self.trained_model_folder = _find_nnunet_trained_model_folder(
            self.ds_name, self.n_epochs
        )

        self.predictor: nnUNetPredictor | None = None
        # self.pp_fns = None
        # self.pp_fn_kwargs = None
        # self.postprocessing_path: Path | None = None

        self.setup_unet()

    def setup_unet(self):
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        self.predictor.initialize_from_trained_model_folder(
            str(self.trained_model_folder),
            use_folds=self.folds,
            checkpoint_name=self.checkpoint_name,
        )

        self.predictor.network = self.predictor.network.float()

        self.model = self.predictor.network.to(self.device)

        # if self.apply_postprocessing:
        #     self.pp_fns, self.pp_fn_kwargs, self.postprocessing_path = (
        #         _load_nnunet_postprocessing(self.trained_model_folder)
        #     )

        # self.props = self.load_props()

    # def load_props(self):
    #     # /home/tblanchard/phd/hdd/nnUnet_datasets/nnUNet_results/Dataset426979_KOMATSUNA__2026-04-24_15-28-07/nnUNetTrainer_5epochs__nnUNetPlans__2d/fold_0/validation/case0000_0000.pkl
    #     validation_folder = _find_nnunet_validation_folder(self.trained_model_folder)
    #     sample_case = next(validation_folder.glob("*.pkl"))
    #     props = load_pickle(sample_case)
    #     return props

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # x_in = [x_ for x_ in x.unsqueeze(-1).cpu().numpy()]
        # x_in = x.cpu().numpy().astype(np.float32)  # (B, C, H, W)
        # x_in = x_in[:, None, ...]  # (B, 1, C, H, W)

        # seg = self.predictor.predict_from_list_of_npy_arrays(
        #     image_or_list_of_images=[x_i for x_i in x_in],
        #     segs_from_prev_stage_or_list_of_segs_from_prev_stage=None,
        #     properties_or_list_of_properties=[self.props for _ in range(x_in.shape[0])],
        #     truncated_ofname=None,
        # )
        # if (
        #     self.apply_postprocessing
        #     and self.pp_fns is not None
        #     and self.pp_fn_kwargs is not None
        # ):
        #     seg = apply_postprocessing(
        #         seg,
        #         pp_fns=self.pp_fns,
        #         pp_fn_kwargs=self.pp_fn_kwargs,
        #     )
        # seg = torch.from_numpy(np.array(seg))  # (B, 1, H, W)

        # Don't bother with the predictor and postprocessing
        # Because all the images have the same size and the same properties, and we just want the segmentation masks
        seg = self.model(x)  # (B, 2, H, W)
        seg = torch.argmax(seg, dim=1, keepdim=True)  # (B, 1, H, W)

        B, _, H, W = seg.shape
        area = (seg > self.threshold).sum(dim=[1, 2, 3])  # (B,)
        area = area.float() / (H * W)  # Normalize by image size

        if self.return_seg:
            return area, seg
        else:
            return area


def get_model_regressor(export_path, device, return_seg=False, n_epochs=100):
    model = NNUnetModel(
        export_path=export_path, device=device, return_seg=return_seg, n_epochs=n_epochs
    )
    return model


def get_vae_komatsuna_dataloaders(
    dataset_config: DatasetConfig,
    latent_dim: int,
    export_path: Path,
    device: torch.device,
    vae_prob: float = 1.0,
):
    # Get a dataset of KOMATSUNA reconstructed by the VAE to train the regressor on
    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(export_path, model)
    model = model.to(device)
    model.eval()

    train_dataset = KOMATSUNADataset(
        train=True,
        export_path=export_path,
    )
    test_dataset = KOMATSUNADataset(
        train=False,
        export_path=export_path,
    )

    @torch.no_grad()
    def collate_with_model(batch: list, model: VAE, device: torch.device):
        x_list = [b[0] for b in batch]
        y_list = [b[1] for b in batch]
        x_list = torch.stack(x_list, dim=0)
        y_list = torch.stack(y_list, dim=0)

        if torch.rand(1).item() < vae_prob:
            x_list = x_list.to(device)
            x_list, _, _ = model(x_list)
        return x_list.cpu(), y_list

    komatsuna_train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=dataset_config.n_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_with_model(batch, model, device),
    )
    komatsuna_test_loader = DataLoader(
        test_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=dataset_config.n_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_with_model(batch, model, device),
    )
    return komatsuna_train_loader, komatsuna_test_loader


def _make_run_tag(export_path: Path) -> str:
    """Create a short filesystem-friendly tag linked to the source export directory."""
    export_path = export_path.resolve()
    run_name = export_path.name
    run_name = re.sub(r"[^A-Za-z0-9._-]+", "-", run_name).strip("-")
    return run_name or export_path.name or "run"


def _get_nnunet_dataset_spec(export_path: Path) -> tuple[int, str, Path]:
    """Build a deterministic nnUNet dataset id and folder name from export_path."""
    resolved_export_path = export_path.resolve()
    digest = hashlib.sha1(str(resolved_export_path).encode("utf-8")).hexdigest()
    dataset_id = int(digest[:10], 16) % 900000 + 100000
    run_tag = _make_run_tag(export_path)
    ds_name = f"Dataset{dataset_id:06d}_KOMATSUNA__{run_tag}"
    return dataset_id, ds_name, NNUNET_DS_PATH / ds_name


def _prepare_nnunet_dataset_dir(ds_path: Path):
    """Remove any previous export so reruns overwrite the same dataset cleanly."""
    if ds_path.exists():
        shutil.rmtree(ds_path)
    ds_path.mkdir(parents=True, exist_ok=True)


def _write_nnunet_provenance(
    ds_path: Path,
    *,
    dataset_id: int,
    ds_name: str,
    export_path: Path,
    latent_dim: int,
    device: torch.device,
):
    provenance = {
        "dataset_id": dataset_id,
        "dataset_name": ds_name,
        "source_export_path": str(export_path.resolve()),
        "source_export_name": export_path.name,
        "latent_dim": latent_dim,
        "device": str(device),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(ds_path / "nnunet_export_provenance.json", "w") as f:
        dump(provenance, f, indent=4)


def _load_nnunet_export_link(export_path: Path) -> tuple[int, str, Path]:
    """Load the nnUNet dataset link written next to the originating export path."""
    link_file = export_path / "nnunet_export_dataset.json"
    if not link_file.exists():
        raise FileNotFoundError(
            f"Missing nnUNet export link file: {link_file}. Run export_reco_nnunet(export_path=...) first."
        )
    with open(link_file, "r") as f:
        link_payload = load(f)
    return (
        int(link_payload["dataset_id"]),
        str(link_payload["dataset_name"]),
        Path(link_payload["dataset_path"]),
    )


def _find_nnunet_trained_model_folder(ds_name: str, n_epochs: int) -> Path:
    dataset_root = Path(nnUNet_results) / ds_name
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Missing nnUNet results folder for dataset {ds_name}: {dataset_root}"
        )

    if n_epochs == 1:
        trainer_prefix = "nnUNetTrainer_1epoch"
    else:
        trainer_prefix = f"nnUNetTrainer_{n_epochs}epochs"
    matches = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_dir() and path.name.startswith(trainer_prefix) and "2d" in path.name
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find a trained model folder for {ds_name} with trainer prefix {trainer_prefix} in {dataset_root}"
        )
    if len(matches) > 1:
        LOGGER.warning(
            f"Multiple nnUNet model folders match {trainer_prefix}; using {matches[0]}"
        )
    return matches[0]


def _load_nnunet_postprocessing(trained_model_folder: Path):
    candidates = sorted(trained_model_folder.rglob("postprocessing.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find postprocessing.pkl under {trained_model_folder}"
        )
    pp_fns, pp_fn_kwargs = load_pickle(str(candidates[0]))
    return pp_fns, pp_fn_kwargs, candidates[0]


def _find_nnunet_validation_folder(trained_model_folder: Path) -> Path:
    preferred_candidates = [
        trained_model_folder / "fold_0" / "validation",
        trained_model_folder / "fold_all" / "validation",
    ]
    for candidate in preferred_candidates:
        if candidate.exists():
            return candidate

    candidates = sorted(
        path for path in trained_model_folder.rglob("validation") if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a validation folder under {trained_model_folder}"
        )
    if len(candidates) > 1:
        LOGGER.warning(
            f"Multiple nnUNet validation folders found under {trained_model_folder}; using {candidates[0]}"
        )
    return candidates[0]


@torch.inference_mode()
def export_nnunet_images(loader, ds_path, model, device, train=True):
    desc = (
        "Saving train images and annotations"
        if train
        else "Saving test images and annotations"
    )
    pbar = tqdm(enumerate(loader), desc=desc, total=len(loader))
    for idx, (img, annot) in pbar:
        img_hat = model(img.to(device))[0]  # (B, 3, 256, 256)
        img_hat = (img_hat.clamp(0, 1).cpu() * 255.0).round().to(torch.uint8)
        for i in range(img.shape[0]):
            annot_i = annot[i].cpu()  # (H, W), binary mask
            # Remap to 3 channels to export as PNG
            # annot_i = annot_i.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
            img_hat_i = img_hat[i].cpu()  # (3, H, W)
            case_id = f"case{idx:04d}_{i:04d}"
            img_filename = f"{case_id}_0000.png"
            annot_filename = f"{case_id}.png"
            if train:
                img_path = ds_path / "imagesTr" / img_filename
                annot_path = ds_path / "labelsTr" / annot_filename
            else:
                img_path = ds_path / "imagesTs" / img_filename
                annot_path = ds_path / "labelsTs" / annot_filename
            img_path.parent.mkdir(parents=True, exist_ok=True)
            annot_path.parent.mkdir(parents=True, exist_ok=True)

            torchvision.io.write_png(img_hat_i, img_path)
            torchvision.io.write_png(annot_i.unsqueeze(0).byte(), annot_path)


def export_reco_nnunet(latent_dim: int, export_path: Path, device: torch.device) -> int:
    """
    Given an AE, takes the true labels and the reconstructed images
    Then export them in a nnunet_like format to train a nnUnet on the
    reconstructed images and true labels
    """
    dataset_id, ds_name, ds_path = _get_nnunet_dataset_spec(export_path)
    _prepare_nnunet_dataset_dir(ds_path)

    LOGGER.info(
        f"Exporting reconstructed images and true labels to {ds_path} for nnUnet training..."
    )
    LOGGER.info(f"Source export path for this nnUnet dataset: {export_path.resolve()}")

    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(export_path, model)
    model.to(device)
    ds_config = DatasetConfig(return_annots=True)
    train_loader, test_loader = get_komatsuna_dataloaders(
        ds_config,
        export_path,
        use_augmentation=False,
    )

    export_nnunet_images(train_loader, ds_path, model, device=device, train=True)
    export_nnunet_images(test_loader, ds_path, model, device=device, train=False)

    ds_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "plant": 1},
        "numTraining": len(train_loader.dataset),
        "numTest": len(test_loader.dataset),
        "file_ending": ".png",
        "source_export_path": str(export_path.resolve()),
        "source_export_name": export_path.name,
        "nnunet_dataset_id": dataset_id,
        "nnunet_dataset_name": ds_name,
    }
    with open(ds_path / "dataset.json", "w") as f:
        dump(ds_json, f, indent=4)

    link_file = export_path / "nnunet_export_dataset.json"
    link_payload = {
        "dataset_id": dataset_id,
        "dataset_name": ds_name,
        "dataset_path": str(ds_path.resolve()),
        "source_export_path": str(export_path.resolve()),
        "source_export_name": export_path.name,
    }
    with open(link_file, "w") as f:
        dump(link_payload, f, indent=4)

    _write_nnunet_provenance(
        ds_path,
        dataset_id=dataset_id,
        ds_name=ds_name,
        export_path=export_path,
        latent_dim=latent_dim,
        device=device,
    )

    return dataset_id


def _run_nnunet_cmd(cmd: list[str], *, ds_name: str, step_name: str) -> bool:
    LOGGER.info(f"[{ds_name}] {step_name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        LOGGER.error(f"[{ds_name}] {step_name} failed with exit code {result.returncode}")
        return False
    return True


def train_nnunet(
    export_path: Path,
    n_epochs: int = 100,
    configuration: str = "2d",
    # folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    folds: tuple[int, ...] = (0,),
    disable_ensembling: bool = True,
) -> bool:
    """Run the full nnUNet v2 training workflow for the exported dataset.

    Workflow:
    1) plan and preprocess with integrity check
    2) train requested folds with --npz
    3) find best configuration (this also determines postprocessing)
    """
    dataset_id, ds_name, _ = _load_nnunet_export_link(export_path)
    if n_epochs == 1:
        trainer_name = "nnUNetTrainer_1epoch"
    else:
        trainer_name = f"nnUNetTrainer_{n_epochs}epochs"

    preprocess_cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        str(dataset_id),
        "--verify_dataset_integrity",
    ]
    if not _run_nnunet_cmd(preprocess_cmd, ds_name=ds_name, step_name="plan_and_preprocess"):
        return False

    for fold in folds:
        train_cmd = [
            "nnUNetv2_train",
            str(dataset_id),
            configuration,
            str(fold),
            "--npz",
            "-tr",
            trainer_name,
        ]
        if not _run_nnunet_cmd(train_cmd, ds_name=ds_name, step_name=f"train_fold_{fold}"):
            return False

    # nnUNet docs: this command identifies best configuration and determines postprocessing.
    find_best_cmd = [
        "nnUNetv2_find_best_configuration",
        str(dataset_id),
        "-c",
        configuration,
        "-p",
        "nnUNetPlans",
        "-tr",
        trainer_name,
        "-f",
        *[str(f) for f in folds],
    ]
    if disable_ensembling:
        find_best_cmd.append("--disable_ensembling")

    if not _run_nnunet_cmd(
        find_best_cmd,
        ds_name=ds_name,
        step_name="find_best_configuration_and_postprocessing",
    ):
        return False

    return True


def nnunet_pipeline(
    export_path: Path, latent_dim: int, device: torch.device, n_epochs: int = 100
) -> bool:
    try:
        dataset_id = export_reco_nnunet(
            latent_dim=latent_dim, export_path=export_path, device=device
        )
        LOGGER.info(
            f"Exported nnUNet dataset with ID {dataset_id} for export at {export_path}"
        )
        success = train_nnunet(export_path=export_path, n_epochs=n_epochs)
        if not success:
            LOGGER.error(
                f"nnUNet training failed for dataset ID {dataset_id} at {export_path}"
            )
            return False
        LOGGER.info(
            f"nnUNet training completed successfully for dataset ID {dataset_id} at {export_path}"
        )
        return True
    except Exception as e:
        LOGGER.exception(
            f"An error occurred during the nnUNet pipeline for export at {export_path}: {e}"
        )
        return False
