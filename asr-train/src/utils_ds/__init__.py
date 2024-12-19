"""Utilities for building the Common Voice dataset"""

import logging
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from app.config import (
    pth_valid_dev_raw,
    pth_valid_test_raw,
    pth_valid_train_raw,
    sampling_rate,
)
from datasets import Dataset, DatasetDict, concatenate_datasets
from pydub import AudioSegment

# Regex for anything that's not an uppercase letter or space
rgx_non_upper_or_space_or_apos = re.compile(r"[^A-Z ']+")


def disp_audio(filename: str) -> AudioSegment:
    """(EDA function) Get the audio for the `filename` for displaying in a Jupyter notebook"""
    from app.config import (
        valid_dev_raw_dir,
        valid_test_raw_dir,
        valid_train_raw_dir,
    )

    if filename.startswith("cv-valid-dev/"):
        cur_dir = valid_dev_raw_dir
    elif filename.startswith("cv-valid-train/"):
        cur_dir = valid_train_raw_dir
    elif filename.startswith("cv-valid-test/"):
        cur_dir = valid_test_raw_dir
    else:
        raise ValueError(f"Invalid filename: {filename}")
    return AudioSegment.from_mp3(cur_dir.parent / filename)


def get_df_wer(
    df_metadata: pd.DataFrame,
    ds: Dataset,
    pth_df_wer: Path = None,
    force: bool = False,
    col_pred: str = "generated_text",
    col_wer: str = "wer",
) -> pd.DataFrame:
    """Get the WER dataframe for the dataset `ds`.

    Args:
        df_metadata (pd.DataFrame): Metadata dataframe with at least the columns ["filename", "generated_text", "label"]
        ds (Dataset): Dataset with at least the columns ["filename", "text"]
        pth_df_we (Path, optional): Path to save the WER dataframe, if provided.
        force (bool): If True, force recompute the WER dataframe
        col_pred (str): Column name to store the predictions
        col_wer (str): Column name to store the WER

    Returns: WER dataframe
    """
    import evaluate

    wer_metric = evaluate.load("wer")

    if not force and pth_df_wer.exists():
        return pd.read_csv(pth_df_wer)

    # ensure df_metadata only has rows in the dataset
    df_filename = ds.select_columns(["filename"]).to_pandas()
    # Add `wer` column`
    df_wer = df_filename.merge(df_metadata, on="filename")
    df_wer[col_wer] = df_wer[[col_pred, "label"]].apply(
        lambda srs: wer_metric.compute(predictions=[srs[col_pred]], references=[srs["label"]]),
        axis=1,
    )
    df_wer.sort_values([col_wer, "filename"], ascending=[False, True], inplace=True)
    # Save
    if pth_df_wer is not None:
        df_wer.to_csv(pth_df_wer, index=False)
    return df_wer


def get_df_wer_dev(pth_df_wer_dev: Path, df_dev: pd.DataFrame, ds_dev: Dataset, force: bool = False) -> pd.DataFrame:
    """Get the WER dataframe for the dev set, or compute it if not available.

    Args:
        pth_df_wer_dev (Path): Path to save the WER dataframe.
        df_dev (pd.DataFrame): DataFrame of the dev set.
        ds_dev (Dataset): Dataset of the dev set.
        force (bool, optional): Whether to force recomputation.

    Returns:
        DataFrame of the WER for the dev set.
    """
    if not force and pth_df_wer_dev.exists():
        return pd.read_csv(pth_df_wer_dev)

    df_wer_dev_original = get_df_wer(df_dev, ds_dev, force=True)
    col_wer = "wer_finetuned"
    col_pred = "generated_text_finetuned"
    df_wer_dev_finetuned = get_df_wer(df_dev, ds_dev, force=True, col_pred=col_pred, col_wer=col_wer)

    df_wer_dev = df_wer_dev_finetuned.merge(df_wer_dev_original[["filename", "wer"]], on="filename")
    # Write
    df_wer_dev.to_csv(pth_df_wer_dev, index=False)

    return df_wer_dev


def backup_file(pth: Path):
    pth_bkp = pth.with_suffix(f"{pth.suffix}.bak")
    if not pth_bkp.exists():
        from shutil import copyfile

        copyfile(pth, pth_bkp)
    return pth_bkp


def save_best_model(best_checkpoint_dir: Path, pth_model: Path):
    """Save the best checkpoint to the `pth_model` directory"""
    from shutil import copytree

    if pth_model.exists():
        raise FileExistsError(f"{pth_model} already exists. Please delete it first.")
    copytree(best_checkpoint_dir, pth_model)


def preprocess_text(txt: str) -> str:
    """Preprocesss the text for the ASR dataset
    (1) Makes all characters uppercase
    (2) Removes all characters that are not uppercase letters or spaces
    """
    return rgx_non_upper_or_space_or_apos.sub("", txt.upper())


def get_df_valid_train(remove_majority_downvotes: bool = True) -> pd.DataFrame:
    """Get the parsed dataframe for the valid_train dataset
    Args:
        remove_majority_downvotes: If True, keep only rows where the majority vote is upvote
    """
    df = pd.read_csv(pth_valid_train_raw)
    df["stats"] = df["age"].fillna("nil") + "," + df["gender"].fillna("nil") + "," + df["accent"].fillna("nil")
    assert df["filename"].str.endswith(".mp3").all(), "Some files are not mp3 files"
    if remove_majority_downvotes:
        df = df[df["up_votes"] > df["down_votes"]].copy()

    return df


def get_df_valid_test(remove_majority_downvotes: bool = True) -> pd.DataFrame:
    """Get the parsed dataframe for the valid_test dataset
    Args:
        remove_majority_downvotes: If True, keep only rows where the majority vote is upvote
    """
    df = pd.read_csv(pth_valid_test_raw)
    df["stats"] = df["age"].fillna("nil") + "," + df["gender"].fillna("nil") + "," + df["accent"].fillna("nil")
    assert df["filename"].str.endswith(".mp3").all(), "Some files are not mp3 files"
    if remove_majority_downvotes:
        df = df[df["up_votes"] > df["down_votes"]].copy()

    return df


def get_df_valid_dev(remove_majority_downvotes: bool = True) -> pd.DataFrame:
    """Get the parsed dataframe for the valid_dev dataset
    Args:
        remove_majority_downvotes: If True, keep only rows where the majority vote is upvote
    """
    df = pd.read_csv(pth_valid_dev_raw)
    df["stats"] = df["age"].fillna("nil") + "," + df["gender"].fillna("nil") + "," + df["accent"].fillna("nil")
    assert df["filename"].str.endswith(".mp3").all(), "Some files are not mp3 files"
    if remove_majority_downvotes:
        df = df[df["up_votes"] > df["down_votes"]].copy()

    return df


def array_to_audio(array: List[float], sampling_rate: int = sampling_rate) -> AudioSegment:
    """Convert a list of floats `array` to an AudioSegment
    (Used for double-checking the preprocessing)
    """
    from audiosegment import from_numpy_array

    return from_numpy_array((np.array(array) * 32767).astype(np.int16), framerate=sampling_rate)


def get_ds_fingerprint_chunk(ds_fingerprint_pfx: str, i_chunk: int) -> str:
    """Get the fingerprint for the dataset chunk
    Args:
        ds_fingerprint_pfx: Prefix for the dataset fingerprint
        i_chunk: Index of the chunk
    """
    return f"{ds_fingerprint_pfx}_chunk{i_chunk}"


def get_ds_cur_chunk_dir(ds_fingerprint_pfx: str, i_chunk: int, ds_wip_dir: Path) -> Path:
    """Get the directory for the current chunk
    Args:
        ds_fingerprint_pfx: Prefix for the dataset fingerprint
        i_chunk: Index of the chunk
        ds_wip_dir: Directory for the WIP datasets
    """
    return ds_wip_dir / get_ds_fingerprint_chunk(ds_fingerprint_pfx, i_chunk)


def load_ds_from_disk(ds_cur_dir: Path) -> Dataset:
    """Load the dataset from disk
    Args:
        ds_cur_dir: Directory for the dataset
    """
    pths = sorted((ds_cur_dir / "wip").glob("*/"), key=lambda x: int(x.name.split("_")[-1][len("chunk") :]))
    return concatenate_datasets([Dataset.load_from_disk(pth) for pth in pths])
