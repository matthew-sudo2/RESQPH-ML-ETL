"""
Download Sentinel-1 GRDH COG scenes over Metro Manila for specific dates.

Requires COPERNICUS_USER / COPERNICUS_PASS in .env.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.utils import config

load_dotenv()

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products({pid})/$value"

# Metro Manila bbox (WGS84)
MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = 120.90, 14.35, 121.15, 14.80

# Target scenes for Typhoon Ulysses (Nov 2020)
# Format: (label, date_start, date_end)
SCENES = [
    ("pre_ulysses_2020",  "2020-11-06T10:00:00.000Z", "2020-11-06T10:10:00.000Z"),
    ("post_ulysses_2020", "2020-11-18T10:00:00.000Z", "2020-11-18T10:10:00.000Z"),
]

OUT_DIR = config.raw_path("sentinel1")


def get_token() -> str:
    user = os.getenv("COPERNICUS_USER")
    pwd = os.getenv("COPERNICUS_PASS")
    if not user or not pwd:
        raise SystemExit("Set COPERNICUS_USER and COPERNICUS_PASS in .env")

    r = requests.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": user,
            "password": pwd,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def find_scene(token: str, date_start: str, date_end: str) -> dict:
    """Return the best matching GRDH COG product for the date window."""
    bbox = (
        f"POLYGON(({MIN_LON} {MIN_LAT}, {MAX_LON} {MIN_LAT}, "
        f"{MAX_LON} {MAX_LAT}, {MIN_LON} {MAX_LAT}, {MIN_LON} {MIN_LAT}))"
    )
    filt = (
        "Collection/Name eq 'SENTINEL-1' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{bbox}') and "
        f"ContentDate/Start gt {date_start} and "
        f"ContentDate/Start lt {date_end} and "
        "contains(Name, 'IW_GRDH_1SDV') and "
        "contains(Name, '_COG')"
    )
    r = requests.get(
        CATALOGUE_URL,
        params={"$filter": filt, "$top": 5, "$orderby": "ContentDate/Start asc"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    r.raise_for_status()
    items = r.json().get("value", [])
    if not items:
        raise SystemExit(f"No COG scene found for {date_start} → {date_end}")
    return items[0]


def download(product_id: str, name: str, token: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = DOWNLOAD_URL.format(pid=product_id)
    print(f"Downloading {name} → {out_path.name}")

    with requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        written = 0
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100 * written / total
                        print(f"\r  {pct:5.1f}%  ({written / 1e6:6.1f} MB)", end="")
    print()


def main() -> None:
    token = get_token()
    print("Authenticated with Copernicus Data Space.\n")

    for label, d_start, d_end in SCENES:
        product = find_scene(token, d_start, d_end)
        pid = product["Id"]
        name = product["Name"]
        out_path = OUT_DIR / f"{label}.tif"
        if out_path.exists():
            print(f"{out_path.name} already exists, skipping.")
            continue
        download(pid, name, token, out_path)

    print(f"\nDone. Files in {OUT_DIR}")


if __name__ == "__main__":
    main()