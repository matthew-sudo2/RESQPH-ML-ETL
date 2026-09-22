"""Delete raw Sentinel-1 downloads after labels are built. Keeps the small mask."""
from pathlib import Path

from src.utils import config

RAW_DIR = config.raw_path("sentinel1")

KEEP = {"flood_mask_ulysses_2020.tif"}   # mask stays in interim/, not raw/

def main() -> None:
    if not RAW_DIR.exists():
        print(f"{RAW_DIR} doesn't exist. Nothing to do.")
        return

    total = 0
    for f in RAW_DIR.iterdir():
        if f.is_file() and f.name not in KEEP:
            size = f.stat().st_size
            f.unlink()
            total += size
            print(f"Deleted {f.name}  ({size / 1e9:.2f} GB)")

    print(f"\nFreed {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()