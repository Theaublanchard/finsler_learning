from pathlib import Path
import shutil

from finsler_learning.shepplogan_pipeline.shepplogan_utils import (
    VALID_COMETRIC_TYPES,
)


def copy_files_to_biaised(config_path: Path, cometric_type: str) -> None:
    """
    Copy the files from the original cometric type to the biased cometric type.

    Parameters
    ----------
    config_path : Path
        The path to the configuration directory.
    cometric_type : str
        The cometric type to copy from.
    """
    if not cometric_type.endswith("_biased"):
        raise ValueError(f"cometric_type must end with '_biased', got {cometric_type}.")

    original_cometric_type = cometric_type.replace("_biased", "")

    if not (config_path / "shepplogan_pipeline_config.json").exists():
        raise FileNotFoundError(
            f"Original config file does not exist: {config_path / 'shepplogan_pipeline_config.json'}"
        )

    original_path = config_path / f"randers_{original_cometric_type}"
    biased_path = config_path / f"randers_{cometric_type}"
    if not original_path.exists():
        raise FileNotFoundError(f"Original path does not exist: {original_path}")
    biased_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(original_path, biased_path, dirs_exist_ok=True)
