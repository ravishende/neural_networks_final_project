import sys
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
from .molec_smiles_helpers import remove_atom_mapping


def create_orderly_folders(project_dir=None, print_paths=True):
    """Create standard raw and processed directories for the ORDerly dataset."""
    project_dir = Path(project_dir or Path.cwd())
    raw_dir = project_dir / "data" / "raw" / "orderly_ord"
    processed_dir = project_dir / "data" / "processed" / "orderly_ord"

    for folder in (raw_dir, processed_dir):
        folder.mkdir(parents=True, exist_ok=True)

    if print_paths:
        print(f"Project directory: {project_dir}\n"
              f"ORDerly Raw data directory: {raw_dir}\n"
              f"ORDerly Processed data directory: {processed_dir}")

    return raw_dir, processed_dir


def download_and_process_orderly(raw_dir, download_full_train=False):
    """Downloads ORDerly forward test parquet via Figshare API, 
    sanitizes atom mappings (this dataset doesn't have them anyways), 
    tracks pipeline loss metrics, and structures data splits for sequence tokenizers.
    The resulting SMILES strings are saved to raw_dir/orderly_ord/test.txt and valid.txt
    If download_full_train=True, then the training set will also be processed and saved to train.txt.
    """
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    # 1. Dynamic check for parquet engines
    try:
        import pyarrow
    except ImportError:
        try:
            import fastparquet
        except ImportError:
            print("Parquet engine missing. Installing fastparquet...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fastparquet"])

    # 2. Resolve download URLs via Figshare API or fallback
    ARTICLE_ID = "23298467"
    api_url = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
    print(f"Querying Figshare API for Article ID {ARTICLE_ID} file manifests...")

    try:
        res = requests.get(api_url, timeout=30)
        res.raise_for_status()
        files_list = res.json().get("files", [])
    except Exception as e:
        print(f"API issue ({e}). Using fallback download signatures...")
        files_list = [
            {"name": "orderly_forward_test.parquet", "download_url": "https://figshare.com/ndownloader/files/44350415"},
            {"name": "orderly_forward_train.parquet", "download_url": "https://figshare.com/ndownloader/files/44350418"}
        ]

    targets = ["orderly_forward_test.parquet"] + (["orderly_forward_train.parquet"] if download_full_train else [])
    downloaded_paths = {}

    # 3. Stream asset downloads
    for target_name in targets:
        meta = next((f for f in files_list if f["name"] == target_name), None)
        if not meta: continue

        dest_parquet = raw_dir / target_name
        if dest_parquet.exists() and dest_parquet.stat().st_size > 0:
            print(f"Asset already cached: {dest_parquet.name}")
            downloaded_paths[target_name] = dest_parquet
            continue

        print(f"Streaming binary download: {target_name}")
        res = requests.get(meta["download_url"], stream=True, timeout=120)
        res.raise_for_status()

        with open(dest_parquet, "wb") as f, tqdm(
            total=int(res.headers.get("content-length", 0)), unit="B", unit_scale=True, desc=target_name
        ) as pbar:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        downloaded_paths[target_name] = dest_parquet

    # Internal processing logic
    def process_parquet_to_txt(source_parquet, output_txt_name):
        print(f"\nProcessing token fields: {source_parquet.name} -> {output_txt_name}")
        df = pd.read_parquet(source_parquet)
        if "rxn_str" not in df.columns:
            raise KeyError(f"Could not locate 'rxn_str' column. Found: {list(df.columns)}")

        stats = {"total_rows": 0, "saved_successfully": 0, "invalid_chemistry_dropped": 0, "malformed_syntax_dropped": 0}
        
        with open(raw_dir / output_txt_name, "w", encoding="utf-8") as f_out:
            for rxn in df["rxn_str"].dropna():
                stats["total_rows"] += 1
                parts = str(rxn).strip().split(">")
                
                if len(parts) != 3:
                    stats["malformed_syntax_dropped"] += 1
                    continue

                reactants, reagents, products = parts
                inputs = f"{reactants}.{reagents}" if reagents.strip() else reactants
                
                clean_in = remove_atom_mapping(inputs)
                clean_out = remove_atom_mapping(products)

                if clean_in is None or clean_out is None:
                    stats["invalid_chemistry_dropped"] += 1
                    continue

                f_out.write(f"{clean_in}>>{clean_out} 0\n")
                stats["saved_successfully"] += 1

        print(f" -> Summary Metrics for {output_txt_name}:")
        for key, val in stats.items():
            print(f"    - {key.replace('_', ' ').capitalize():<28}: {val}")
        source_parquet.unlink()

    # 4. Process split configurations natively
    test_file = downloaded_paths.get("orderly_forward_test.parquet")
    if test_file and test_file.exists():
        process_parquet_to_txt(test_file, "test.txt")
        shutil.copyfile(raw_dir / "test.txt", raw_dir / "valid.txt")
        print("Duplicated validation records to initialize valid.txt")

    train_file = downloaded_paths.get("orderly_forward_train.parquet")
    train_txt_path = raw_dir / "train.txt"

    if download_full_train and train_file and train_file.exists():
        process_parquet_to_txt(train_file, "train.txt")
    else:
        print(f"Generating local placeholder verification split: {train_txt_path.name}")
        train_txt_path.write_text("CC.O>>CCO 0\n", encoding="utf-8")

    print("\nORDerly data pipeline structures successfully initialized!")