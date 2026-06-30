# Roadmap

*Companion to the [design](DESIGN.md). Updated 2026-06-30.*

**The rule for every step:** it must produce something real and test it honestly before we
move on. Nothing is "done" because the code runs — it's done when we can show a result and
say truthfully how good it is. UI polish is deliberately last.

---

## ✅ Already built (v1)

- **Multi-city engine** — Delhi, Mumbai, Bengaluru, Chandigarh, each a config file, not new
  code.
- **The ~100-metre grid + ground features** (terrain, low spots, drainage backbone, paving,
  population) for every city.
- **Two models per city** — a river/floodplain model from multi-event satellite radar, and
  a separate street-waterlogging model, validated with within-city spatial cross-validation.
- **A traceable event database** of documented waterlogging hotspots (+ news-frequency over
  time) and a **live, interactive map**.
- **Clean engineering** — installable package, CLI, typed configs, experiment tracking, CI.

## 🔜 Next (v2)

### ✅ v2.1 — Evaluation that survives scrutiny *(shipped 2026-06-30)*
*Goal: numbers a sharp reviewer would believe.*
- ✅ Added the harder, honest test: **train on three cities, predict the unseen fourth**
  (`floodml eval-crosscity`).
- ✅ Report a **fair rare-event score** (PR-AUC vs the prevalence baseline) alongside ROC-AUC.
- ✅ **Calibrate** the probabilities (isotonic) and show reliability before/after.

**Result:** generalizes to unseen *inland* cities (ROC-AUC 0.71–0.77) but **fails on coastal
Mumbai (0.59)**; rare-event PR-AUC is a sober 0.08–0.19; calibration cuts Brier from
0.23–0.64 to 0.03–0.06. Full writeup → [EVALUATION.md](EVALUATION.md). This directly sets the
priorities below (the Mumbai failure is the case for the tidal feature + drainage entities).

### ✅ Explainability — "why is this spot risky?" *(shipped 2026-06-30)*
*Goal: a map you can interrogate, not a black box.*
- For any location the model gives plain-English reasons, from exact per-prediction feature
  contributions (tree SHAP, no extra dependency). `floodml explain <city>` → a per-hotspot
  JSON + a chart.

**Result:** across all four cities the model's reasons are consistent and physically
sensible — flooded underpasses come back as pronounced dips / low points with poor local
drainage. That's the trust groundwork the rest of v2 builds on.

### v2.2 — Drains as a network, and report scoring
*Goal: the city behaves like a connected system, not loose squares.*
- Build the drain map (free shapes; hand-stitch the fragmented Najafgarh; capacity left
  blank and marked "unknown").
- Link each square to the drain it feeds.
- Score reports with multi-signal confidence (rainfall at the time + multiple sources +
  known bad spot; satellite image only as an optional bonus, only where it could see).

**Done when:** you can drill down region → drain → square, and a reported drain overflow
visibly raises the risk of every square feeding it.

### v2.3 — Reading the government PDFs
*Goal: get drain facts out of documents, safely.*
- A local model pulls candidate facts from drainage PDFs; each is tagged with its source
  document and page and **must be approved by a human** before it affects the map.

**Done when:** drain facts in the map are traceable to a source and human-checked.

### v2.4 — Forecasts and alerts
*Goal: a forward-looking view that never overclaims.*
- Pull free rain forecasts (good to about 15 days) and build the three-tier outlook (0–3
  day warning, 3–10 day advisory, 2–4 week regional watch), with the official extended
  outlook read from its PDF (using v2.3).

**Done when:** alerts are tiered by how far ahead they look, each labelled with its honest
confidence, and no long-range outlook is ever shown as a specific forecast.

### v2.5 — Map / dashboard refinements
*Goal: make it clear and usable.* Deferred by choice until the engine is solid.

---

**Order of dependencies:** v2.1 stands alone and comes first (it uses the models that
already exist). v2.2 adds the drain network and report scoring. v2.3 supports v2.2 and is
reused by v2.4. v2.5 comes last.
