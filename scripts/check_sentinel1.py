"""
Quick probe: check Sentinel-1 coverage over Metro Manila for a given time window.
Adjust DATE_START and DATE_END to target a specific flood event.
"""
import requests

# Metro Manila bbox (WGS84)
MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = 120.90, 14.35, 121.15, 14.80

# Target window — Typhoon Ulysses (Nov 2020). Adjust for other events.
DATE_START = "2020-11-01T00:00:00.000Z"
DATE_END   = "2020-11-30T23:59:59.000Z"

# Build WKT polygon
bbox_wkt = (
    f"POLYGON(({MIN_LON} {MIN_LAT}, {MAX_LON} {MIN_LAT}, "
    f"{MAX_LON} {MAX_LAT}, {MIN_LON} {MAX_LAT}, {MIN_LON} {MIN_LAT}))"
)

url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

filter_str = (
    "Collection/Name eq 'SENTINEL-1' and "
    f"OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}') and "
    f"ContentDate/Start gt {DATE_START} and "
    f"ContentDate/Start lt {DATE_END}"
)

params = {
    "$filter": filter_str,
    "$top": 50,
    "$orderby": "ContentDate/Start asc",
}

print("Querying Copernicus Data Space...")
r = requests.get(url, params=params, timeout=60)

if r.status_code != 200:
    print("HTTP", r.status_code)
    print(r.text[:1000])
    raise SystemExit(1)

data = r.json()
items = data.get("value", [])
print(f"Found {len(items)} Sentinel-1 scenes over Metro Manila in the window.")
print()

for item in items:
    name = item.get("Name", "?")
    start = item.get("ContentDate", {}).get("Start", "?")
    size_mb = (item.get("ContentLength", 0) or 0) / 1024 / 1024
    print(f"  {start}  |  {size_mb:6.1f} MB  |  {name}")
