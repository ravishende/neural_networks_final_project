import os
import time
import random
import zipfile
import numpy as np
import requests
import codecs
import math
import pickle
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Subset
from pathlib import Path
from rdkit import Chem, RDLogger
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from SmilesPE.tokenizer import SPE_Tokenizer
from dotenv import load_dotenv
from collections import Counter
import ray
from ray import tune
import wandb
from ray.air.integrations.wandb import WandbLoggerCallback

def main():
    # constants
    SEED = 274
    MAX_SRC_LEN = 256
    MAX_TGT_LEN = 256
    N_RANDOM_SMILES_AUGMENTATIONS = 4  # set to 0 for no randomized smiles augmentation

    # defaults - change after hyperparameter tuning if specified in the SEARCH_SPACE dict
    D_MODEL = 256
    N_HEAD = 8
    NUM_LAYERS = 4
    FF_DIM = 1024
    BATCH_SIZE = 64

    EPOCHS = 30
    USE_RAY_TUNE = True
    NUM_HPARAM_TUNING_TRIALS = 10

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"



    # Fallback config (used if USE_RAY_TUNE = False, or as defaults before tuning)
    config = {
        "lr": 1e-4,
        "batch_size": BATCH_SIZE,
        "dropout": 0.1,
        "d_model": D_MODEL,
        "ff_dim": FF_DIM,
    }

    SEARCH_SPACE = {
        "lr": tune.loguniform(1e-5, 5e-4),
        "batch_size": tune.choice([32,64,128,]),
        "dropout": tune.uniform(0.0,0.3,),
        "d_model": tune.choice([128,256,384,]),
        "ff_dim": tune.choice([512,1024,2048]),
    }

    RDLogger.DisableLog('rdApp.*')
    set_seed(SEED, prefer_reproducible_over_performance=False)
    RAW_DIR, PROCESSED_DIR, output_dirs_dict = create_data_folders(
        named_output_dirs={"transformer":"transformer_smilespe"})
    OUTPUT_DIR = output_dirs_dict["transformer"]
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    TOKEN_DIR = PROCESSED_DIR / f"{N_RANDOM_SMILES_AUGMENTATIONS}_augmentations"
    TOKEN_CACHE = {
        "train": TOKEN_DIR / "train_tokens.pt",
        "valid": TOKEN_DIR / "valid_tokens.pt",
        "test": TOKEN_DIR / "test_tokens.pt",
    }

    TOKEN_TO_ID_PATH = TOKEN_DIR / "token_to_id.pkl"
    ID_TO_TOKEN_PATH = TOKEN_DIR / "id_to_token.pkl"

    DEVICE = pick_device()
    print(DEVICE)

    load_dotenv()

    print_title("Downloading data")

    download_uspto_mit(RAW_DIR)
    split_paths = get_split_paths(RAW_DIR)
    writer = SummaryWriter(log_dir=OUTPUT_DIR / "tensorboard")

    print_title("Token Processing")

    SPE_PATH = Path("data/SPE_ChEMBL.txt")

    spe_vocab = codecs.open(SPE_PATH)
    spe = SPE_Tokenizer(spe_vocab)
    def tokenize_smiles(smiles):
        return spe.tokenize(smiles).split()

    def build_token_cache():
        token_counter = {}
        encoded = {}
        n_filtered_by_split = {split: 0 for split in split_paths}
        for split, path in split_paths.items():
            rows = []
            with open(path) as f:
                for line in tqdm(f):
                    parsed = parse_reaction_line(line)
                    if parsed is None:
                        continue
                    
                    src, tgt = parsed
                    src_tokens = tokenize_smiles(src)
                    tgt_tokens = tokenize_smiles(tgt)
                    if len(src_tokens) > MAX_SRC_LEN:
                        n_filtered_by_split[split] += 1
                        continue
                    if len(tgt_tokens) > MAX_TGT_LEN:
                        n_filtered_by_split[split] += 1
                        continue
                    rows.append((src_tokens, tgt_tokens))

                    if split == "train":
                        # count number of tokens for later analysis and to build vocab
                        for token in src_tokens + tgt_tokens:
                            token_counter[token] = token_counter.get(token, 0) + 1
                        # add randomized smiles copies (and count resulting tokens)
                        for _ in range(N_RANDOM_SMILES_AUGMENTATIONS):
                            src_aug = randomize_smiles_components(src)
                            src_aug_tokens = tokenize_smiles(src_aug)
                            rows.append((src_aug_tokens, tgt_tokens))
                            
                            for token in src_aug_tokens + tgt_tokens:
                                token_counter[token] = token_counter.get(token, 0) + 1

                    

                        
            encoded[split] = rows
            print(f"{split}: filtered {n_filtered_by_split[split]} reactions")
        pd.Series(n_filtered_by_split).to_csv(OUTPUT_DIR / "filtering_stats.csv", index=False)
        
        token_to_id = {
            PAD_TOKEN: 0,
            BOS_TOKEN: 1,
            EOS_TOKEN: 2,
            UNK_TOKEN: 3,
        }
        for token in sorted(token_counter):
            token_to_id[token] = len(token_to_id)
        id_to_token = {v: k for k, v in token_to_id.items()}

        def encode(tokens):
            return (
                [token_to_id[BOS_TOKEN]]
                + [token_to_id.get(t, token_to_id[UNK_TOKEN]) for t in tokens]
                + [token_to_id[EOS_TOKEN]]
            )
        for split in encoded:
            dataset = [
                (encode(src), encode(tgt))
                for src, tgt in encoded[split]
            ]
            torch.save(dataset, TOKEN_CACHE[split])

        with open(TOKEN_TO_ID_PATH, "wb") as f:
            pickle.dump(token_to_id, f)

        with open(ID_TO_TOKEN_PATH, "wb") as f:
            pickle.dump(id_to_token, f)

    if not TOKEN_TO_ID_PATH.exists():
        build_token_cache()
    else:
        print("Pulling existing token cache from disk")

    with open(TOKEN_TO_ID_PATH, "rb") as f:
        token_to_id = pickle.load(f)

    with open(ID_TO_TOKEN_PATH, "rb") as f:
        id_to_token = pickle.load(f)

    train_data = torch.load(TOKEN_CACHE["train"], weights_only=True)
    valid_data = torch.load(TOKEN_CACHE["valid"], weights_only=True)
    test_data = torch.load(TOKEN_CACHE["test"], weights_only=True)

    train_ref = ray.put(train_data)
    valid_ref = ray.put(valid_data)

    VOCAB_SIZE = len(token_to_id)
    PAD_ID = token_to_id[PAD_TOKEN]

    print(VOCAB_SIZE)


    def ids_to_smiles(ids, id_to_token):
        tokens = []
        for idx in ids:
            token = id_to_token[int(idx)]
            if token == EOS_TOKEN:
                break
            if token in {PAD_TOKEN, BOS_TOKEN}:
                continue
            tokens.append(token)
        return "".join(tokens)
    
    class ReactionDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    def collate_fn(batch):

        srcs, tgts = zip(*batch)

        src = nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in srcs],
            batch_first=True,
            padding_value=PAD_ID,
        )

        tgt = nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in tgts],
            batch_first=True,
            padding_value=PAD_ID,
        )

        return src, tgt

    def generate_square_subsequent_mask(size, device):
        return torch.triu(
            torch.full((size, size), float("-inf"), device=device),
            diagonal=1,
        )

    class PositionalEncoding(nn.Module):
        def __init__(
            self,
            d_model: int,
            dropout: float = 0.1,
            max_len: int = 2048,
        ):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2)
                * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)

    class ReactionTransformer(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            pad_id: int,
            d_model: int = 256,
            num_layers: int = 4,
            num_heads: int = 8,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.pad_id = pad_id
            self.d_model = d_model
            # Shared source/target embedding
            self.embedding = nn.Embedding(
                vocab_size,
                d_model,
                padding_idx=pad_id,
            )
            self.positional_encoding = PositionalEncoding(
                d_model=d_model,
                dropout=dropout,
            )
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=num_heads,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.output_layer = nn.Linear(
                d_model,
                vocab_size,
            )

        def make_causal_mask(self, size: int, device):
            return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

        def forward(self, src, tgt_in):
            memory, src_pad_mask = self.encode(src)
            return self.decode_step(tgt_in, memory, src_pad_mask)
        
        def encode(self, src):
            src_pad_mask = src.eq(self.pad_id)
            src_emb = self.positional_encoding(
                self.embedding(src) * math.sqrt(self.d_model)
            )
            memory = self.transformer.encoder(
                src_emb,
                src_key_padding_mask=src_pad_mask
            )
            return memory, src_pad_mask

        def decode_step(self, tgt, memory, memory_mask):
            tgt_pad_mask = tgt.eq(self.pad_id)
            tgt_mask = self.make_causal_mask(tgt.size(1), tgt.device)
            tgt_emb = self.positional_encoding(
                self.embedding(tgt) * math.sqrt(self.d_model)
            )
            out = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=memory_mask,
            )
            return self.output_layer(out)

    def run_epoch(model, loader, optimizer, criterion, train: bool):
        model.train(train)
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        desc = "train" if train else "valid"
        for src, tgt in tqdm(loader, desc=desc):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            with torch.set_grad_enabled(train):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            mask = tgt_out.ne(PAD_ID)
            preds = logits.argmax(dim=-1)
            n_tokens = mask.sum().item()
            n_correct = ((preds == tgt_out) & mask).sum().item()

            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            total_correct += n_correct

        avg_loss = total_loss / max(total_tokens, 1)
        token_acc = total_correct / max(total_tokens, 1)
        return avg_loss, token_acc

    @torch.no_grad()
    def greedy_decode(
        model,
        src,
        bos_id,
        eos_id,
        max_len=256,
    ):
        model.eval()
        
        src = src.to(DEVICE)
        memory, memory_mask = model.encode(src)

        generated = torch.full(
            (src.size(0), 1),
            bos_id,
            dtype=torch.long,
            device=DEVICE,
        )
        for _ in range(max_len):
            logits = model.decode_step(generated, memory, memory_mask)
            next_token = logits[:, -1].argmax(-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1,)

            if (next_token == eos_id).all():
                break

        return generated

    def train_tune(config, train_ref, valid_ref):
        train = train_ref
        valid = valid_ref

        train_subset = Subset(
            ReactionDataset(train),
            range(min(50000, len(train))),
        )

        valid_subset = Subset(
            ReactionDataset(valid),
            range(min(5000, len(valid))),
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=min(8, os.cpu_count() or 1),
            persistent_workers=True,
        )

        valid_loader = DataLoader(
            valid_subset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=min(8, os.cpu_count() or 1),
            persistent_workers=True,
        )

        model = ReactionTransformer(
            vocab_size=VOCAB_SIZE,
            pad_id=PAD_ID,
            d_model=config["d_model"],
            num_layers=NUM_LAYERS,
            num_heads=N_HEAD,
            dim_feedforward=config["ff_dim"],
            dropout=config["dropout"],
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=1e-4,
        )

        criterion = nn.CrossEntropyLoss(
            ignore_index=PAD_ID,
        )

        for epoch in range(5):

            train_loss, train_token_acc = run_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                train=True,
            )

            valid_loss, valid_token_acc = run_epoch(
                model,
                valid_loader,
                optimizer,
                criterion,
                train=False,
            )

            tune.report(
                {
                    "valid_loss": valid_loss,
                    "valid_token_acc": valid_token_acc,
                    "train_loss": train_loss,
                    "train_token_acc": train_token_acc,
                }
            )

    print_title("Hyperparameter Tuning")

    trainable = tune.with_resources(
        tune.with_parameters(
            train_tune, train_ref=train_ref, valid_ref=valid_ref
        ),
        resources={
            "cpu": 2,
            "gpu": 1,
        }
    )

    analysis = tune.run(
        trainable,
        config=SEARCH_SPACE,
        metric="valid_loss",
        mode="min",
        num_samples=NUM_HPARAM_TUNING_TRIALS,
        callbacks=[
            WandbLoggerCallback(project="reaction-transformer-raytune",)
        ],
    )

    config = analysis.get_best_config(
        metric="valid_loss",
        mode="min",
    )
    print("best config:", config, sep="\n")

    best_trial = analysis.get_best_trial(
        metric="valid_loss",
        mode="min",
    )
    print("Best config:")
    print(best_trial.config)
    print()
    print("Best validation loss:")
    print(best_trial.last_result["valid_loss"])
    print()
    print("Validation token accuracy:")
    print(best_trial.last_result["valid_token_acc"])

    print_title("Training + Evaluation")

    @torch.no_grad()
    def evaluate_model(
        model,
        loader,
        id_to_token,
        bos_id,
        eos_id,
    ):
        model.eval()
        predictions = []
        targets = []
        token_preds = []
        token_targets = []
        for src, tgt in loader:
            src = src.to(DEVICE)
            generated = greedy_decode(model,src,bos_id,eos_id)
            for pred_ids, tgt_ids in zip(generated.cpu(),tgt):
                pred_smiles = ids_to_smiles(pred_ids,id_to_token)
                tgt_smiles = ids_to_smiles(tgt_ids,id_to_token)
                predictions.append(pred_smiles)
                targets.append(tgt_smiles)
                token_preds.append(pred_ids.tolist())
                token_targets.append(tgt_ids.tolist())

        return {
            "string_exact_match":string_exact_match_accuracy(predictions,targets),
            "chemical_match":chemical_equivalence_accuracy(predictions,targets,),
            "valid_smiles":valid_smiles_rate(predictions,),
            "token_accuracy":token_accuracy(token_preds,token_targets,pad_token_id=PAD_ID,),
            "levenshtein":average_levenshtein_distance(predictions,targets),
            "predictions":predictions,
            "targets":targets,
        }

    def train_model(
        model,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        criterion,
        epochs,
        id_to_token,
        bos_id,
        eos_id,
        eval_every=3,
    ):
        best_metric = -1.0
        history = []
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss, train_token_acc = run_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                train=True,
            )
            valid_loss, valid_token_acc = run_epoch(
                model,
                valid_loader,
                optimizer,
                criterion,
                train=False,
            )
            metrics = None
            chemical_match = None
            valid_smiles = None
            if (epoch % eval_every == 0 or epoch == epochs - 1):
                start = time.time()
                metrics = evaluate_model(
                    model,
                    valid_loader,
                    id_to_token,
                    bos_id,
                    eos_id,
                )
                print(f"eval time: {time.time() - start:.2f}s")
                chemical_match = metrics["chemical_match"]
                valid_smiles = metrics["valid_smiles"]
                scheduler.step(
                    chemical_match
                )
                if chemical_match > best_metric:
                    best_metric = chemical_match
                    torch.save({
                        "epoch": epoch,
                        "metric": chemical_match,
                        "model_state_dict":
                            model.state_dict(),
                        "optimizer_state_dict":
                            optimizer.state_dict(),
                        "scheduler_state_dict":
                            scheduler.state_dict()},
                        CHECKPOINT_DIR / "best_model.pt")

                    print(f"New best model: {chemical_match:.4f}")
                print(f"chemical_match={chemical_match:.4f} valid_smiles={valid_smiles:.4f}")

            history_row = {
                "epoch":epoch + 1,
                "train_loss":train_loss,
                "valid_loss":valid_loss,
                "train_token_acc":train_token_acc,
                "valid_token_acc":valid_token_acc,
                "best_chemical_match":best_metric,
            }
            if metrics is not None:
                history_row.update({
                    "chemical_match":metrics["chemical_match"],
                    "string_exact_match":metrics["string_exact_match"],
                    "valid_smiles":metrics["valid_smiles"],
                    "levenshtein":metrics["levenshtein"],
                })
            history.append(history_row)
            torch.save({
                "epoch": epoch,
                "metric": chemical_match,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "scheduler_state_dict":
                    scheduler.state_dict(),
            },
            CHECKPOINT_DIR / "last_model.pt")
            print(
                f"train_loss={train_loss:.4f} "
                f"valid_loss={valid_loss:.4f} "
                f"train_acc={train_token_acc:.4f} "
                f"valid_acc={valid_token_acc:.4f}"
            )
            writer.add_scalar("loss/train",train_loss,epoch)
            writer.add_scalar("loss/valid",valid_loss,epoch)
            if chemical_match is not None:
                writer.add_scalar("metrics/chemical_match", chemical_match, epoch)
            if len(history) > 1:
                history_df = pd.DataFrame(history)
                history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
        
        history_df = pd.DataFrame(history)
        return history_df, best_metric

    train_loader = DataLoader(
        ReactionDataset(train_data),
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=min(8, os.cpu_count() or 1),
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        ReactionDataset(valid_data),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=min(8, os.cpu_count() or 1),
        persistent_workers=True,
    )
    test_loader = DataLoader(
        ReactionDataset(test_data),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=min(8, os.cpu_count() or 1),
        persistent_workers=True,
    )
    model = ReactionTransformer(
        vocab_size=VOCAB_SIZE,
        pad_id=PAD_ID,
        d_model=config["d_model"],
        num_layers=NUM_LAYERS,
        num_heads=N_HEAD,
        dim_feedforward=config["ff_dim"],
        dropout=config["dropout"],
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
    )
    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=EPOCHS,
        id_to_token=id_to_token,
        bos_id=token_to_id[BOS_TOKEN],
        eos_id=token_to_id[EOS_TOKEN],
    )

    print_title("Loading Best Model and Evaluating on Test Set")
    checkpoint = torch.load(CHECKPOINT_DIR/"best_model.pt", map_location=DEVICE, weights_only=False)
    model = ReactionTransformer(
        vocab_size=VOCAB_SIZE,
        pad_id=PAD_ID,
        d_model=config["d_model"],
        num_layers=NUM_LAYERS,
        num_heads=N_HEAD,
        dim_feedforward=config["ff_dim"],
        dropout=config["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Best validation chemical match: {checkpoint['metric']}")
    print(f"Loaded from epoch {checkpoint['epoch']}")

    test_metrics = evaluate_model(
        model,
        test_loader,
        id_to_token,
        token_to_id[BOS_TOKEN],
        token_to_id[EOS_TOKEN],
    )

    print_title("Test Metrics")
    for metric, value in test_metrics.items():
        if isinstance(value, list):
            continue
        print(f"{metric}: {value:.4f}")

    errors = error_analysis_rows(
        test_metrics["predictions"],
        test_metrics["targets"],
    )
    error_df = pd.DataFrame(errors)
    error_df.to_csv(OUTPUT_DIR/"error_analysis.csv", index=False)
    print(error_df.head())

    error_counts = Counter(category for row in errors for category in row["categories"])
    error_counts_series = pd.Series(error_counts).sort_values(ascending=False)
    error_counts_series.to_csv(OUTPUT_DIR/"error_counts.csv", header=["count"])
    
    print_title("Error Category Counts")
    print(error_counts_series)


# ------------------------------------------------
#               HELPER FUNCTIONS
# ------------------------------------------------

def print_title(msg):
    """Prints a title message with a border"""
    min_width = min(100, len(msg) + 20)
    print("\n\n" + "="*min_width)
    padding = (min_width - len(msg)) // 2
    padded_msg = " " * padding + msg + " " * padding
    print(padded_msg)
    print("="*min_width + "\n")


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
    # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return torch.device("mps")
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


def string_exact_match_accuracy(predictions, targets):
    if not targets:
        return 0.0
    return sum(pred == target for pred, target in zip(predictions, targets)) / len(targets)


def canonicalize_smiles_components(smiles):
    parts = [part for part in smiles.split(".") if part]
    if not parts:
        return None
    canonical_parts = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        canonical_parts.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return ".".join(sorted(canonical_parts))


def chemically_equivalent(prediction, target):
    prediction_canonical = canonicalize_smiles_components(prediction)
    target_canonical = canonicalize_smiles_components(target)
    return prediction_canonical is not None and prediction_canonical == target_canonical


def chemical_equivalence_accuracy(predictions, targets):
    if not targets:
        return 0.0
    return sum(chemically_equivalent(pred, target) for pred, target in zip(predictions, targets)) / len(targets)


def exact_match_accuracy(predictions, targets):
    """Primary correctness metric: exact chemical equivalence after canonicalization."""
    return chemical_equivalence_accuracy(predictions, targets)


def token_accuracy(pred_token_ids, target_token_ids, pad_token_id=0):
    correct = 0
    total = 0
    for pred, target in zip(pred_token_ids, target_token_ids):
        for pred_id, target_id in zip(pred, target):
            if target_id == pad_token_id:
                continue
            total += 1
            correct += int(pred_id == target_id)
    return correct / total if total else 0.0


def levenshtein_distance(left, right):
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + int(left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def average_levenshtein_distance(predictions, targets):
    if not targets:
        return 0.0
    return sum(levenshtein_distance(pred, target) for pred, target in zip(predictions, targets)) / len(targets)


def valid_smiles_rate(smiles_list):
    if not smiles_list:
        return 0.0
    valid = 0
    for smiles in smiles_list:
        parts = [part for part in smiles.split(".") if part]
        valid += int(bool(parts) and all(Chem.MolFromSmiles(part) is not None for part in parts))
    return valid / len(smiles_list)


def categorize_prediction(prediction, target):
    categories = []
    if len(target) > 120:
        categories.append("Long sequences")
    is_equivalent = chemically_equivalent(prediction, target)
    if any(ch.isdigit() for ch in prediction + target) and not is_equivalent:
        categories.append("Ring closure errors")
    if "@" in prediction + target and not is_equivalent:
        categories.append("Stereochemistry errors")
    if any(Chem.MolFromSmiles(part) is None for part in prediction.split(".") if part):
        categories.append("Invalid SMILES")
    if not is_equivalent and levenshtein_distance(prediction, target) <= max(5, len(target) * 0.10):
        categories.append("Near misses")
    return categories or ["Other"]


def error_analysis_rows(predictions, targets, limit_correct=50, limit_incorrect=50):
    rows = []
    correct_count = 0
    incorrect_count = 0
    for prediction, target in zip(predictions, targets):
        is_correct = chemically_equivalent(prediction, target)
        if is_correct and correct_count >= limit_correct:
            continue
        if not is_correct and incorrect_count >= limit_incorrect:
            continue
        rows.append(
            {
                "prediction": prediction,
                "target": target,
                "correct": is_correct,
                "string_exact_match": prediction == target,
                "levenshtein": levenshtein_distance(prediction, target),
                "prediction_canonical": canonicalize_smiles_components(prediction),
                "target_canonical": canonicalize_smiles_components(target),
                "categories": categorize_prediction(prediction, target),
            }
        )
        correct_count += int(is_correct)
        incorrect_count += int(not is_correct)
        if correct_count >= limit_correct and incorrect_count >= limit_incorrect:
            break
    return rows


def smiles_to_mol(smiles):
    """
    Convert a SMILES string to an RDKit Mol object.

    Args:
        smiles: SMILES string representation of a molecule.

    Returns:
        RDKit Mol object, or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None
    return mol


def randomize_smiles(smiles):
    """
    Randomize a SMILES string.

    Args:
        smiles: SMILES string representation of a molecule.

    Returns:
        randomized SMILES string, or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return smiles

    return Chem.MolToSmiles(
        mol,
        canonical=False,
        doRandom=True,
        isomericSmiles=True,
    )


def randomize_smiles_components(smiles):
    """
    Randomize the molecules in a multi-molecule SMILES string (separated by '.')

    Args:
        smiles: SMILES string representation of several molecules.

    Returns:
        randomized SMILES string where each molecule was individually randomized.
    """
    randomized = []
    for component in smiles.split("."):
        random_smiles = randomize_smiles(component)
        if random_smiles is None:
            randomized.append(component)
        else:
            randomized.append(random_smiles)
    return ".".join(randomized)


def print_mol_basic_info(smiles):
    """
    Print basic RDKit molecule information and return the molecule.

    Args:
        smiles: SMILES string representation of a molecule.

    Returns:
        RDKit Mol object, or None if the SMILES is invalid.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    print("Original SMILES:", smiles)
    print("Canonical SMILES:", canonical_smiles)
    print("Number of atoms:", mol.GetNumAtoms())
    print("Number of bonds:", mol.GetNumBonds())
    return mol


def remove_atom_mapping(smiles):
    """
    Remove atom-mapping numbers from a SMILES string.

    Args:
        smiles: Atom-mapped SMILES string.

    Returns:
        Canonical SMILES without atom mappings, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def parse_reaction_line(line):
    reaction = line.strip().split()[0]
    src, tgt = reaction.split(">>")

    src = remove_atom_mapping(src)
    tgt = remove_atom_mapping(tgt)

    if src is None or tgt is None:
        return None

    return src, tgt




if __name__ == "__main__":
    main()