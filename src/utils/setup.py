import random
import zipfile
import requests
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def create_data_folders(project_dir=None, named_output_dirs=None, print_paths=True):
    """
    Create standard data and output directories for the project.

    Creates the following folder structure in the project_dir directory:

    data/
        raw/uspto_mit/
        processed/uspto_mit/
    outputs/
        output directories

    Args:
        project_dir (Path): Project src/ directory. Defaults to Path.cwd().
            - if not specified, it should be called by a notebook or file run in the `src/` folder.
        named_output_dirs (dict): Mapping of output names to directory names.
            Defaults to an empty dictionary.
            - Ex: `{"Recursive NN output directory":"recursive_nn", "Transformer baseline output directory":"transformer_char_baseline"}`
        print_paths (bool): whether to print the paths of the created folders 

    Returns:
        Tuple containing:
            - RAW_DIR (Path): Raw USPTO data directory.
            - PROCESSED_DIR (Path): Processed USPTO data directory.
            - OUTPUT_DIRS_DICT (dict[str, Path]): Mapping of output names
              to created output directories.
    """
    if project_dir is None:
        project_dir = Path.cwd()
    if named_output_dirs is None:
        named_output_dirs = {}
    data_dir = project_dir / "data"
    raw_dir = data_dir / "raw" / "uspto_mit"
    processed_dir = data_dir / "processed" / "uspto_mit"
    output_dirs_dict = {
        name: project_dir/"outputs"/directory for name, directory in named_output_dirs.items()}

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for output_dir in output_dirs_dict.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    if print_paths:
        print("Project directory:", project_dir)
        print("Raw data directory:", raw_dir)
        print("Processed data directory:", processed_dir)
        print("Output Directories:")
        for name, output_dir in output_dirs_dict.items():
            print(f"\t{name} Directory: {output_dir}")
    return raw_dir, processed_dir, output_dirs_dict


def set_seed(seed: int = 274, prefer_reproducible_over_performance=True):
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Configures random number generators and cuDNN settings to control
    reproducibility and performance during training.

    Args:
        seed: Random seed used for Python, NumPy, and PyTorch.
        deterministic: Whether to force deterministic cuDNN operations.
            Improves reproducibility but may reduce performance.
        benchmark: Whether cuDNN should benchmark kernels to select the
            fastest implementation. Can improve performance but may reduce
            reproducibility.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    deterministic = None
    benchmark = None
    if prefer_reproducible_over_performance:
        deterministic = True
        benchmark = False
    else:
        deterministic = False
        benchmark = True
    # These settings improve reproducibility. They can make training slightly slower.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

def pick_device():
    """
    Select the best available PyTorch device.

    Prefers CUDA GPUs, then Apple Metal (MPS), then CPU otherwise.

    Returns:
        torch.device: The selected compute device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def download_uspto_mit(raw_dir):
    USPTO_MIT_URL = "https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip"
    zip_path = raw_dir / "data.zip"
    url = USPTO_MIT_URL
    dest = zip_path
    
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already exists: {dest}")
        return

    print(f"Downloading from {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"Saved to {dest}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    print("Extracted files:")
    for path in sorted(raw_dir.rglob("*"))[:50]:
        print(" -", path.relative_to(raw_dir))


def get_file_info(raw_dir):
    """
    Collect summary information for all text files in a directory tree.

    Recursively searches for .txt files and records each file's relative
    path, size, and first-line preview. Prints an error if a file can't be read.

    Args:
        raw_dir: Root directory to search for text files.

    Returns:
        pd.DataFrame: One row per text file containing path, size_mb, first_line_preview
    """
    txt_files = sorted(raw_dir.rglob("*.txt"))
    file_info = []
    for p in txt_files:
        try:
            size_mb = p.stat().st_size / (1024 ** 2)
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
            file_info.append({
                "path": str(p.relative_to(raw_dir)),
                "size_mb": round(size_mb, 2),
                "first_line_preview": first_line[:120],
            })
        except Exception as e:
            file_info.append({"path": str(p.relative_to(raw_dir)), "error": str(e)})
    return pd.DataFrame(file_info)

def get_split_paths(raw_dir):
    def find_split_file(raw_directory: Path, split: str) -> Path:
        """Find train/dev/test split file recursively."""
        aliases = {
            "train": ["train"],
            "valid": ["valid", "validation", "dev"],
            "test": ["test"],
        }[split]

        candidates = []
        for p in raw_directory.rglob("*.txt"):
            name = p.name.lower()
            stem = p.stem.lower()
            if any(a == stem or a in name for a in aliases):
                candidates.append(p)

        if not candidates:
            raise FileNotFoundError(
                f"Could not find split={split}. Available txt files: {[str(p.relative_to(raw_directory)) for p in raw_directory.rglob('*.txt')]}"
            )

        # Prefer exact train.txt/dev.txt/test.txt names.
        for alias in aliases:
            for p in candidates:
                if p.name.lower() == f"{alias}.txt":
                    return p

        # Otherwise choose the largest matching file.
        return max(candidates, key=lambda p: p.stat().st_size)


    split_paths = {
        "train": find_split_file(raw_dir, "train"),
        "valid": find_split_file(raw_dir, "valid"),
        "test": find_split_file(raw_dir, "test"),
    }

    for split, path in split_paths.items():
        print(split, "->", path.relative_to(raw_dir))
    return split_paths