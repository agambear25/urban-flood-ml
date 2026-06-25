# Urban-flood event database

A **traceable, multi-source database of urban (street/underpass) waterlogging** — the labels that satellite radar *can't* provide. Every record carries provenance; hotspots carry a frequency tier; the news timeline tracks frequency-of-occurrence over time.

Built entirely from **free, no-key, legal** sources (no paid X/Twitter, no scraping behind logins).

```bash
python events/build_events.py   # rebuilds everything below
```

## Sources

| Source | What it gives | Access |
|---|---|---|
| **Documented hotspots** (PWD / Delhi Traffic Police advisories, media) | Chronic waterlogging points (Minto Bridge, Pul Prahladpur, Zakhira, ITO…), **geocoded via OpenStreetMap** so coordinates are real | free |
| **GDELT DOC 2.0** | News-mined flood mentions (article + domain + date) and a **flood-news volume timeline** | free, no key |

## Outputs

- `flood_events.csv` — every record with full provenance: `source_type, source, source_url, date, lat, lon, severity, frequency_tier, extraction_method, confidence`
- `hotspots.geojson` — geocoded chronic/recurring waterlogging hotspots
- `gdelt_timeline.csv` — flood-news volume over time → **frequency of occurrence** (clear monsoon spikes, ~10× the dry-season baseline)
- `../data/delhi/waterlog_label.tif` — hotspots rasterised to Delhi's grid: **a training-label layer the `floodml` model can learn from** (the urban-waterlogging equivalent of the SAR flood mask)
- `../docs/assets/delhi_hotspots.json` — hotspots for the live dashboard map

## How frequency-of-occurrence is tracked

1. **Per location** — each hotspot has a `frequency_tier` (`chronic` = reported almost every monsoon, `recurring` = reported in heavy years).
2. **Over time** — the GDELT timeline gives city-wide flood-news volume per day; monsoon months spike sharply, quantifying *when* and *how often* flooding is reported.

## Honest notes

- This is a **v1 seed**. Documented hotspots are media/advisory-derived (no single official storm-drain shapefile exists for Delhi); they're geocoded, not surveyed.
- **X/Twitter is deliberately excluded** — its API is paid/gated post-2023 and scraping violates ToS. GDELT covers the news signal legally.
- News-article ingestion is implemented (`gdelt_articles`); GDELT rate-limits aggressive querying, so re-run if the articles list is empty.
- The label raster marks *known* hotspots only — absence of a hotspot ≠ safe. It complements, not replaces, the SAR riverine labels.
