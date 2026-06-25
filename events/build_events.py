"""Build the traceable urban-flood event database.

Run from repo root:  python events/build_events.py

Produces (in events/):
  flood_events.csv         — every record with full provenance (news + documented hotspots)
  hotspots.geojson         — geocoded chronic waterlogging hotspots + frequency tier
  gdelt_timeline.csv       — flood-news volume over time (frequency-of-occurrence signal)
And (for the ML pipeline / dashboard):
  data/delhi/waterlog_label.tif   — hotspots rasterised to Delhi's grid (a training label layer)
  docs/assets/delhi_hotspots.json — hotspots for the live map
"""
import csv
import json
import time
from pathlib import Path

import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point, mapping

from floodml.events import gdelt_articles, gdelt_timeline, geocode

REPO = Path(__file__).resolve().parent.parent
EV = REPO / "events"
EV.mkdir(exist_ok=True)

# Documented chronic Delhi waterlogging points (widely reported every monsoon in PWD /
# Delhi Traffic Police advisories and Indian media). Names are geocoded via OSM so the
# coordinates are real, not hand-typed. tier = how frequently it is reported flooded.
HOTSPOTS = [
    ("Minto Bridge, New Delhi", "chronic"),
    ("Pul Prahladpur, Delhi", "chronic"),
    ("Zakhira underpass, Delhi", "chronic"),
    ("Azad Market, Delhi", "chronic"),
    ("Moolchand underpass, Delhi", "chronic"),
    ("Pragati Maidan, Delhi", "chronic"),
    ("ITO, Delhi", "recurring"),
    ("Sarai Kale Khan, Delhi", "recurring"),
    ("Mathura Road, Delhi", "recurring"),
    ("Mukarba Chowk, Delhi", "recurring"),
    ("Azadpur, Delhi", "recurring"),
    ("Loni Road, Shahdara, Delhi", "recurring"),
    ("Tilak Bridge, New Delhi", "recurring"),
    ("Rajghat, Delhi", "recurring"),
    ("Najafgarh, Delhi", "recurring"),
    ("Okhla, Delhi", "recurring"),
    ("Bhairon Marg, Delhi", "recurring"),
    ("Rohtak Road, Delhi", "recurring"),
    ("Karol Bagh, Delhi", "recurring"),
    ("Lajpat Nagar, Delhi", "recurring"),
    ("GTB Nagar, Delhi", "recurring"),
    ("Civil Lines, Delhi", "recurring"),
    ("Mehrauli, Delhi", "recurring"),
    ("Dhaula Kuan, Delhi", "recurring"),
]
NEWS_QUERY = '(waterlogging OR waterlogged OR flooded OR flooding) "Delhi" sourcecountry:india'


def fmt_date(s):  # GDELT 20260616T043000Z -> 2026-06-16
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}" if s and len(s) >= 8 else s


def main():
    rows = []                 # provenance-rich event records
    hotspot_feats = []        # geojson features for the map
    print("Geocoding documented hotspots (OSM Nominatim, ~1/s) ...")
    for i, (name, tier) in enumerate(HOTSPOTS):
        loc = geocode(name)
        time.sleep(1.1)        # respect Nominatim usage policy
        if not loc:
            print("  (skip, no geocode):", name)
            continue
        lat, lon = loc
        rows.append({
            "event_id": f"hs{i:03d}", "source_type": "documented_hotspot",
            "source": "PWD / Delhi Traffic Police waterlogging advisories (media-documented)",
            "source_url": "https://nominatim.openstreetmap.org/ (geocode)",
            "date": "", "title_or_name": name.split(",")[0],
            "lat": round(lat, 5), "lon": round(lon, 5),
            "severity": tier, "frequency_tier": tier,
            "extraction_method": "curated+geocoded", "confidence": "high" if tier == "chronic" else "medium",
        })
        hotspot_feats.append({"type": "Feature",
            "geometry": mapping(Point(lon, lat)),
            "properties": {"name": name.split(",")[0], "tier": tier}})
        print(f"  {name.split(',')[0]:28} {tier:9} {lat:.4f},{lon:.4f}")

    print("\nFetching GDELT news (free, no key) ...")
    try:
        for a in gdelt_articles(NEWS_QUERY, timespan="24m", maxrecords=75):
            rows.append({
                "event_id": "", "source_type": "news",
                "source": a["domain"], "source_url": a["url"],
                "date": fmt_date(a["date"]), "title_or_name": (a["title"] or "")[:160],
                "lat": "", "lon": "", "severity": "", "frequency_tier": "",
                "extraction_method": "gdelt_doc_api", "confidence": "medium",
            })
        print(f"  +{sum(1 for r in rows if r['source_type']=='news')} news articles")
    except Exception as e:
        print("  GDELT articles unavailable:", str(e)[:120])

    time.sleep(5)
    print("Fetching GDELT timeline (frequency-over-time) ...")
    timeline = []
    try:
        timeline = gdelt_timeline(NEWS_QUERY, timespan="36m")
        with open(EV / "gdelt_timeline.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["date", "news_volume"])
            for p in timeline:
                w.writerow([p.get("date", "")[:10] if isinstance(p.get("date"), str) else p.get("date"), p.get("value")])
        print(f"  {len(timeline)} time points -> events/gdelt_timeline.csv")
    except Exception as e:
        print("  GDELT timeline unavailable:", str(e)[:120])

    # write the event database
    cols = ["event_id", "source_type", "source", "source_url", "date", "title_or_name",
            "lat", "lon", "severity", "frequency_tier", "extraction_method", "confidence"]
    with open(EV / "flood_events.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    with open(EV / "hotspots.geojson", "w") as fh:
        json.dump({"type": "FeatureCollection", "features": hotspot_feats}, fh)

    # hotspots for the live dashboard
    (REPO / "docs" / "assets").mkdir(parents=True, exist_ok=True)
    with open(REPO / "docs" / "assets" / "delhi_hotspots.json", "w") as fh:
        json.dump([f["properties"] | {"lat": f["geometry"]["coordinates"][1],
                                       "lon": f["geometry"]["coordinates"][0]} for f in hotspot_feats], fh)

    # rasterise hotspots -> a training label aligned to Delhi's grid
    mask_path = REPO / "data" / "delhi" / "flood_mask.tif"
    if mask_path.exists() and hotspot_feats:
        with rasterio.open(mask_path) as src:
            prof, tr, shape = src.profile, src.transform, (src.height, src.width)
        shapes = [(Point(f["geometry"]["coordinates"]).buffer(0.0035), 1) for f in hotspot_feats]
        lab = rasterize(shapes, out_shape=shape, transform=tr, fill=0, all_touched=True, dtype="uint8")
        prof.update(count=1, dtype="uint8", nodata=255, compress="lzw")
        with rasterio.open(REPO / "data" / "delhi" / "waterlog_label.tif", "w", **prof) as dst:
            dst.write(lab, 1)
        print(f"\nRasterised {int((lab==1).sum()):,} hotspot pixels -> data/delhi/waterlog_label.tif")

    n_hs = sum(1 for r in rows if r["source_type"] == "documented_hotspot")
    n_news = sum(1 for r in rows if r["source_type"] == "news")
    print(f"\nDB: {len(rows)} records ({n_hs} hotspots + {n_news} news) -> events/flood_events.csv")


if __name__ == "__main__":
    main()
