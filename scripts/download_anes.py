"""
Try to download ANES 2020 Time Series data, or print manual instructions.

ANES often returns 403 for direct requests; in that case you must download
once from electionstudies.org (free registration) and place the file in
project/data/anes/ as described in project/data/README.md.

Usage:
    python demo/scripts/download_anes.py
    # or from repo root:
    python project/demo/scripts/download_anes.py
"""

import os
import sys
import zipfile
from pathlib import Path

# Paths: support running from demo/ or from project root
SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DEMO_DIR.parent
DATA_ANES = PROJECT_ROOT / "data" / "anes"

ANES_2020_URL = "https://electionstudies.org/wp-content/uploads/2022/02/anes_timeseries_2020_csv_20220210.zip"
ANES_2020_ZIP = "anes_timeseries_2020_csv_20220210.zip"
ANES_2020_CSV = "anes_timeseries_2020_csv_20220210.csv"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def try_download():
    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests")
        return False

    DATA_ANES.mkdir(parents=True, exist_ok=True)
    out_zip = DATA_ANES / ANES_2020_ZIP

    print(f"Trying to download ANES 2020 from:\n  {ANES_2020_URL}")
    print("...")

    try:
        r = requests.get(ANES_2020_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"Download failed: {e}")
        if getattr(r, "status_code", None) == 403:
            print("(Server returned 403 Forbidden; ANES may require login for direct access.)")
        return False

    out_zip.write_bytes(r.content)
    print(f"Saved to: {out_zip}")

    # Unzip to get CSV
    with zipfile.ZipFile(out_zip, "r") as z:
        names = z.namelist()
        csv_name = next((n for n in names if n.endswith(".csv")), None)
        if csv_name:
            z.extract(csv_name, path=DATA_ANES)
            extracted = DATA_ANES / csv_name
            if extracted != DATA_ANES / ANES_2020_CSV:
                (DATA_ANES / ANES_2020_CSV).write_bytes(extracted.read_bytes())
                extracted.unlink()
            print(f"Extracted CSV: {DATA_ANES / ANES_2020_CSV}")
    return True


def manual_instructions():
    print()
    print("=" * 60)
    print("Manual download required")
    print("=" * 60)
    print("1. Register (free) at: https://electionstudies.org/data-center/data-registration/")
    print("2. Open: https://electionstudies.org/data-center/2020-time-series-study/")
    print("3. Download the 2020 Time Series CSV (e.g. anes_timeseries_2020_csv_20220210.zip)")
    print(f"4. Place the ZIP or the extracted CSV in: {DATA_ANES}")
    print()
    print("See project/data/README.md for full instructions.")
    print("=" * 60)


def main():
    if try_download():
        print("ANES 2020 data is ready.")
        return 0
    manual_instructions()
    return 1


if __name__ == "__main__":
    sys.exit(main())
