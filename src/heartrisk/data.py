"""Download, verify, load and clean the UCI Heart Disease data."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from heartrisk.config import (
    DATA_DIR,
    DROPPED_FEATURES,
    FEATURES,
    RAW_COLUMNS,
    SITES,
    TARGET,
)

log = logging.getLogger(__name__)

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
# Byte-identical copy of the UCI files, pinned to one commit so it cannot change.
MIRROR_BASE = (
    "https://huggingface.co/datasets/mstz/heart/resolve/112b4c511c700bf8b17f6417e6a3fa8a9e1007c7"
)

SHA256 = {
    "cleveland": "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    "hungarian": "d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796",
    "switzerland": "834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314",
    "va": "e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f",
}


def _filename(site: str) -> str:
    return f"processed.{site}.data"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "heartrisk-data-fetch"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URLs only)
        data: bytes = resp.read()
    return data


def download(data_dir: Path = DATA_DIR, force: bool = False) -> list[Path]:
    """Download the four processed files, trying UCI first, then the pinned mirror.

    Every file is checked against a known SHA-256, so both sources give identical bytes.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for site in SITES:
        path = data_dir / _filename(site)
        if path.exists() and not force and _sha256(path.read_bytes()) == SHA256[site]:
            paths.append(path)
            continue
        last_error: Exception | None = None
        for base in (UCI_BASE, MIRROR_BASE):
            url = f"{base}/{_filename(site)}"
            try:
                data = _fetch(url)
            except OSError as exc:
                log.warning("Could not fetch %s: %s", url, exc)
                last_error = exc
                continue
            if _sha256(data) != SHA256[site]:
                last_error = ValueError(f"Checksum mismatch for {url}")
                log.warning("%s", last_error)
                continue
            path.write_bytes(data)
            log.info("Saved %s from %s", path.name, base)
            break
        else:
            raise RuntimeError(f"Could not download {site} data") from last_error
        paths.append(path)
    return paths


def load_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read the four files into one frame with a ``site`` column. '?' becomes NaN."""
    frames = []
    for site in SITES:
        frame = pd.read_csv(
            data_dir / _filename(site), header=None, names=RAW_COLUMNS, na_values="?"
        )
        frame["site"] = site
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the data decisions described in the README.

    - Binary target: ``num`` 1-4 means disease present.
    - Cholesterol or blood pressure of 0 is not physically possible; treat as missing.
      (All 123 Switzerland rows have cholesterol 0, so leaving it would let the model
      learn "cholesterol 0 = Switzerland = 93% disease".)
    - Drop ``slope``, ``ca`` and ``thal``: 51-99% missing outside Cleveland.
    - Drop exact duplicate rows.
    - Keep ``site`` only for grouping and reporting. It is never a model feature.
    """
    df = raw.copy()
    df[TARGET] = (df["num"] > 0).astype(int)
    for col in ("chol", "trestbps"):
        df.loc[df[col] == 0, col] = np.nan
    df = df.drop(columns=["num", *DROPPED_FEATURES])
    df = df.drop_duplicates().reset_index(drop=True)
    return df[[*FEATURES, TARGET, "site"]]


def load_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Download if needed, then return the cleaned dataset."""
    download(data_dir)
    return clean(load_raw(data_dir))


def data_report(df: pd.DataFrame) -> dict[str, object]:
    """Small summary of the cleaned data, saved alongside the metrics."""
    return {
        "rows": len(df),
        "positive_rate": round(float(df[TARGET].mean()), 3),
        "rows_by_site": df["site"].value_counts().to_dict(),
        "positive_rate_by_site": df.groupby("site")[TARGET].mean().round(3).to_dict(),
        "missing_rate": df[FEATURES].isna().mean().round(3).to_dict(),
    }
