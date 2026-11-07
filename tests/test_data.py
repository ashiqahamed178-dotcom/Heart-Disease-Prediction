from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heartrisk import data
from heartrisk.config import DROPPED_FEATURES, FEATURES, RAW_COLUMNS, TARGET

ROW = [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0]


def _raw(rows: list[list[float]], site: str = "cleveland") -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df["site"] = site
    return df


def test_clean_binarises_target_and_drops_columns() -> None:
    sick = [*ROW[:-1], 3]
    out = data.clean(_raw([ROW, sick]))
    assert out[TARGET].tolist() == [0, 1]
    assert list(out.columns) == [*FEATURES, TARGET, "site"]
    for col in (*DROPPED_FEATURES, "num"):
        assert col not in out.columns


def test_zero_cholesterol_and_pressure_become_missing() -> None:
    bad = ROW.copy()
    bad[3], bad[4] = 0, 0
    out = data.clean(_raw([bad]))
    assert np.isnan(out.loc[0, "chol"])
    assert np.isnan(out.loc[0, "trestbps"])


def test_duplicates_removed() -> None:
    assert len(data.clean(_raw([ROW, ROW]))) == 1


def test_download_falls_back_to_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"1,2,3\n"
    monkeypatch.setattr(data, "SHA256", {s: data._sha256(payload) for s in data.SITES})
    calls: list[str] = []

    def fake_fetch(url: str, timeout: float = 20.0) -> bytes:
        calls.append(url)
        if url.startswith(data.UCI_BASE):
            raise OSError("down")
        return payload

    monkeypatch.setattr(data, "_fetch", fake_fetch)
    paths = data.download(tmp_path)
    assert len(paths) == 4
    assert all(p.read_bytes() == payload for p in paths)
    assert sum(u.startswith(data.MIRROR_BASE) for u in calls) == 4


def test_download_rejects_bad_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "_fetch", lambda url, timeout=20.0: b"tampered")
    with pytest.raises(RuntimeError):
        data.download(tmp_path)
