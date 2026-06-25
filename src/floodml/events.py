"""Traceable urban-flood / waterlogging EVENT database from free, no-key sources.

Escapes single-event SAR labels (which can't see in-street waterlogging) by assembling a
provenance-rich database of where streets/underpasses actually flood, from:
  - GDELT DOC 2.0   — news-mined flood mentions (free, no key) -> temporal frequency + article provenance
  - Nominatim/OSM   — geocode documented chronic waterlogging hotspots (free)

Every record carries provenance (source, url, date, geocode). Hotspots carry a frequency tier.
Outputs feed `floodml` as a label raster, and the dashboard as a hotspots layer.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "urban-flood-ml/0.1 (portfolio research; agambir.bhatia@gmail.com)"}
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
NOMINATIM = "https://nominatim.openstreetmap.org/search"


def _get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=45).read()
        except Exception:
            time.sleep(4 * (i + 1))  # back off on 429 / transient
    raise RuntimeError("request failed: " + url)


def gdelt_articles(query: str, timespan: str = "24m", maxrecords: int = 75) -> list[dict]:
    """News articles matching a flood query, with date + domain + url (provenance)."""
    q = urllib.parse.quote(query)
    url = (f"{GDELT}?query={q}&mode=artlist&maxrecords={maxrecords}"
           f"&format=json&timespan={timespan}&sort=datedesc")
    data = json.loads(_get(url) or b"{}")
    return [{"date": a.get("seendate"), "domain": a.get("domain"),
             "url": a.get("url"), "title": a.get("title")} for a in data.get("articles", [])]


def gdelt_timeline(query: str, timespan: str = "36m") -> list[dict]:
    """Volume of flood news over time — the frequency-of-occurrence signal (monsoon spikes)."""
    q = urllib.parse.quote(query)
    url = f"{GDELT}?query={q}&mode=timelinevol&format=json&timespan={timespan}"
    data = json.loads(_get(url) or b"{}")
    series = data.get("timeline", [])
    return series[0].get("data", []) if series else []


def geocode(name: str):
    """(lat, lon) for a place name via OpenStreetMap Nominatim, or None."""
    q = urllib.parse.quote(name)
    data = json.loads(_get(f"{NOMINATIM}?q={q}&format=json&limit=1") or b"[]")
    return (float(data[0]["lat"]), float(data[0]["lon"])) if data else None
