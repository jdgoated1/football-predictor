"""Refresh active-season data + international results, then retrain.

Designed for a Windows scheduled task. Wipes only the in-progress caches so
historical season CSVs (which never change) are reused.
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

LOG = ROOT / "update.log"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def clear_active_caches() -> None:
    """Remove the current + next season league CSVs, plus the internationals file.
    Historical season files are left alone (they're frozen)."""
    today = datetime.today()
    end_year = today.year if today.month >= 8 else today.year - 1
    active_codes = []
    for y in (end_year - 1, end_year, end_year + 1):
        a, b = str(y)[-2:], str(y + 1)[-2:]
        active_codes.append(f"{a}{b}")
    removed = 0
    for f in RAW.glob("*.csv"):
        for code in active_codes:
            if f.stem.endswith(f"_{code}") or f.name == "international_results.csv":
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
                break
    # Force re-processing too
    for p in PROCESSED.glob("*.parquet"):
        try:
            p.unlink()
        except OSError:
            pass
    log(f"Cleared {removed} stale csv files")


def main() -> int:
    t0 = time.time()
    log("=== Update started ===")
    try:
        clear_active_caches()
        from src.data_fetcher import fetch_all
        from train import train_scope
        leagues, intl = fetch_all()
        log(f"Fetched: leagues={len(leagues):,}  internationals={len(intl):,}")
        if len(leagues):
            train_scope("leagues", leagues)
        if len(intl):
            train_scope("internationals", intl)
        log(f"=== Update complete in {time.time() - t0:.1f}s ===")
        return 0
    except Exception:
        log("FAILED:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
