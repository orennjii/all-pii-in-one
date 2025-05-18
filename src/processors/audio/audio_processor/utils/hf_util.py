import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from src.commons import find_project_root


def load_custom_model_from_hf(
    repo_id,
    model_filename="pytorch_model.bin",
    config_filename=None,
    sub_dir: str = None,
):
    current_dir = Path(__file__).resolve()
    checkpoint_dir = find_project_root() / "checkpoints"

    if sub_dir is not None:
        checkpoint_dir = checkpoint_dir / sub_dir

    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path = hf_hub_download(
        repo_id=repo_id, filename=model_filename, cache_dir=str(checkpoint_dir)
    )
    if config_filename is None:
        return model_path

    config_path = hf_hub_download(
        repo_id=repo_id, filename=config_filename, cache_dir=str(checkpoint_dir)
    )

    return model_path, config_path