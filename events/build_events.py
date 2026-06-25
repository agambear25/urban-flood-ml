"""Build the traceable, multi-city, multi-source urban-flood event database.

Run from repo root (after the pipeline has produced data/<city>/flood_mask.tif):
    python events/build_events.py

Sources (all free, no paid X/Twitter):
  - Documented waterlogging hotspots (PWD/Traffic-Police/media), geocoded via OSM  [OFFICIAL]
  - GDELT DOC 2.0 news mentions + flood-news volume timeline                        [NEWS]
  - Global Flood Database v1 (GEE GLOBAL_FLOOD_DB/MODIS_EVENTS/V1), dated events     [SCIENTIFIC]

Outputs:
  events/flood_events.csv            — every record with full provenance
  events/hotspots.geojson            — all geocoded hotspots
  events/gdelt_timeline_<city>.csv   — flood-news frequency over time per city
  data/<city>/waterlog_label.tif     — hotspots rasterised to each city's grid (model label)
  docs/assets/<city>_hotspots.json   — hotspots for the live map
"""
import csv
import json
import time
from pathlib import Path

import ee
import rasterio
import yaml
from rasterio.features import rasterize
from shapely.geometry import Point, mapping

from floodml.config import load_city
from floodml.events import gdelt_articles, gdelt_timeline, geocode

REPO = Path(__file__).resolve().parent.parent
EV = REPO / "events"; EV.mkdir(exist_ok=True)
CITIES = ["delhi", "mumbai", "bengaluru", "chandigarh"]
NAME = {"delhi": "Delhi", "mumbai": "Mumbai", "bengaluru": "Bengaluru", "chandigarh": "Chandigarh"}
COLS = ["event_id", "source_type", "source", "source_url", "city", "date", "title_or_name",
        "lat", "lon", "severity", "frequency_tier", "extraction_method", "confidence"]


def fmt_date(s):
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}" if s and len(s) >= 8 and s[:8].isdigit() else (s or "")


def gfd_events(region):
    """Global Flood Database — dated MODIS flood events intersecting the AOI (2000-2018)."""
    try:
        col = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1").filterBounds(region)
        feats = col.limit(60).getInfo().get("features", [])
        out = []
        for f in feats:
            p = f.get("properties", {})
            out.append({"began": str(p.get("began", "")), "ended": str(p.get("ended", "")),
                        "severity": p.get("dfo_severity", ""), "cause": p.get("dfo_main_cause", "")})
        return out
    except Exception as e:
        print("    GFD skipped:", str(e)[:80])
        return []


def process_city(slug, rows, all_feats):
    cfg = load_city(slug, str(REPO / "configs" / "city"))
    W, S, E, N = cfg.west, cfg.south, cfg.east, cfg.north
    region = ee.Geometry.Rectangle([W, S, E, N])
    hs = yaml.safe_load(open(REPO / "configs" / "hotspots" / f"{slug}.yaml"))["hotspots"]

    feats = []
    print(f"[{slug}] geocoding {len(hs)} hotspots ...")
    for i, h in enumerate(hs):
        loc = geocode(h["name"]); time.sleep(1.1)
        if not loc:
            continue
        lat, lon = loc
        if not (S - 0.05 <= lat <= N + 0.05 and W - 0.05 <= lon <= E + 0.05):
            continue  # bad geocode, outside AOI
        rows.append({"event_id": f"{slug}-hs{i:03d}", "source_type": "documented_hotspot",
                     "source": "PWD / Traffic-Police waterlogging advisories (media-documented)",
                     "source_url": "https://nominatim.openstreetmap.org/ (geocode)", "city": slug,
                     "date": "", "title_or_name": h["name"].split(",")[0],
                     "lat": round(lat, 5), "lon": round(lon, 5), "severity": h["tier"],
                     "frequency_tier": h["tier"], "extraction_method": "curated+geocoded",
                     "confidence": "high" if h["tier"] == "chronic" else "medium"})
        f = {"type": "Feature", "geometry": mapping(Point(lon, lat)),
             "properties": {"name": h["name"].split(",")[0], "tier": h["tier"], "city": slug}}
        feats.append(f); all_feats.append(f)
    print(f"  [{slug}] {len(feats)} hotspots geocoded inside AOI")

    # dashboard hotspots
    (REPO / "docs" / "assets").mkdir(parents=True, exist_ok=True)
    json.dump([{"name": x["properties"]["name"], "tier": x["properties"]["tier"],
                "lat": x["geometry"]["coordinates"][1], "lon": x["geometry"]["coordinates"][0]}
               for x in feats], open(REPO / "docs" / "assets" / f"{slug}_hotspots.json", "w"))

    # NEWS — GDELT
    q = f'(waterlogging OR waterlogged OR flooded OR flooding) "{NAME[slug]}" sourcecountry:india'
    try:
        for a in gdelt_articles(q, timespan="24m", maxrecords=40):
            rows.append({"event_id": "", "source_type": "news", "source": a["domain"],
                         "source_url": a["url"], "city": slug, "date": fmt_date(a["date"]),
                         "title_or_name": (a["title"] or "")[:160], "lat": "", "lon": "",
                         "severity": "", "frequency_tier": "", "extraction_method": "gdelt_doc_api",
                         "confidence": "medium"})
    except Exception as e:
        print(f"  [{slug}] GDELT articles skipped:", str(e)[:70])
    time.sleep(6)
    try:
        tl = gdelt_timeline(q, timespan="36m")
        with open(EV / f"gdelt_timeline_{slug}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["date", "news_volume"])
            for p in tl:
                w.writerow([p.get("date"), p.get("value")])
        print(f"  [{slug}] {len(tl)} timeline points")
    except Exception as e:
        print(f"  [{slug}] timeline skipped:", str(e)[:60])
    time.sleep(6)

    # SCIENTIFIC — Global Flood Database
    for ev in gfd_events(region):
        rows.append({"event_id": "", "source_type": "global_flood_db",
                     "source": "Global Flood Database v1 (MODIS/DFO)",
                     "source_url": "https://global-flood-database.cloudtostreet.ai/", "city": slug,
                     "date": ev["began"][:10], "title_or_name": f"DFO flood ({ev['cause']})",
                     "lat": "", "lon": "", "severity": ev["severity"], "frequency_tier": "",
                     "extraction_method": "gee_global_flood_db", "confidence": "medium"})

    # rasterise hotspots -> training label aligned to this city's grid
    mask = REPO / "data" / slug / "flood_mask.tif"
    if mask.exists() and feats:
        with rasterio.open(mask) as src:
            prof, tr, shape = src.profile, src.transform, (src.height, src.width)
        shapes = [(Point(f["geometry"]["coordinates"]).buffer(0.002), 1) for f in feats]
        lab = rasterize(shapes, out_shape=shape, transform=tr, fill=0, all_touched=True, dtype="uint8")
        prof.update(count=1, dtype="uint8", nodata=255, compress="lzw")
        with rasterio.open(REPO / "data" / slug / "waterlog_label.tif", "w", **prof) as dst:
            dst.write(lab, 1)
        print(f"  [{slug}] {int((lab == 1).sum()):,} hotspot pixels -> waterlog_label.tif")
    return len(feats)


def main():
    ee.Initialize(project="urban-flood-analysis-ncr-in")
    rows, all_feats = [], []
    for slug in CITIES:
        process_city(slug, rows, all_feats)

    with open(EV / "flood_events.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    json.dump({"type": "FeatureCollection", "features": all_feats}, open(EV / "hotspots.geojson", "w"))

    by = {}
    for r in rows:
        by[r["source_type"]] = by.get(r["source_type"], 0) + 1
    print(f"\nDB: {len(rows)} records across {len(CITIES)} cities -> events/flood_events.csv")
    print("  by source:", by)


if __name__ == "__main__":
    main()
