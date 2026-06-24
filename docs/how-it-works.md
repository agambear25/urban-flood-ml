# How this works — explained from zero

You don't need any GIS, satellite, or machine-learning background to read this. Every term is explained in plain language the first time it shows up. Read it top to bottom once and the whole project will click.

---

## 0. The one-sentence version

> We use **satellite radar** to find where a city *actually flooded*, then teach a **computer model** to learn the *terrain pattern* of those flooded places, so it can point at *other* places with the same pattern and say "this is flood-prone too."

That's the entire idea. Everything below is just the details of those two steps, plus the engineering that makes it repeatable for many cities.

---

## 1. The problem

When a river bursts or a cloudburst hits, water doesn't go everywhere equally — it collects in **low, flat ground near drainage**. We want a map that shades every spot in a city by *how flood-prone it is*. Two honest sub-problems:

1. **Where did it flood last time?** (an observed fact — we get this from satellites)
2. **Where is it likely to flood in general?** (a prediction — we get this from a model that learned from #1)

---

## 2. Part A — Finding a flood that already happened (the satellite half)

### What a satellite radar actually does

There's a European satellite called **Sentinel-1**. It carries **radar** (called *SAR* — Synthetic Aperture Radar). Think of radar like this:

> Imagine standing in a canyon and **shouting**, then listening to the echo. Rough rocky walls bounce your shout back loudly. A smooth glassy lake bounces your shout *away* — so you hear almost **silence**.

The satellite "shouts" microwaves at the ground and listens to the echo. Two superpowers:
- **It works through clouds and at night.** Floods happen under thick monsoon cloud — exactly when a normal camera satellite is blind. Radar doesn't care about clouds.
- **Calm water sounds like silence.** Smooth floodwater bounces the radar pulse *away* from the satellite, so flooded ground suddenly comes back **dark** (a quiet echo) compared to before.

That darkening is the only thing we hunt for.

### "Before vs after" — change detection

It's literally a **spot-the-difference** game between two pictures:

- **Before** picture: a radar image of the city in a *dry* month (e.g. June 2023, before the Delhi flood).
- **After** picture: a radar image *during* the flood (mid-July 2023).

We subtract one from the other, pixel by pixel:

```
change = after − before
```

If a spot got a lot **darker** (echo went quiet) → water arrived there → **flooded**. If nothing changed → dry land that stayed dry. We draw a line ("if it darkened by at least 3 decibels, call it flood") and we get a yes/no map.

### Wait — what is a "pixel" / "image" here?

A satellite image isn't really a photo; it's a giant **grid of numbers**, like a spreadsheet laid over the city. Each cell ("**pixel**") covers a **10 m × 10 m** square of real ground and holds one number (how loud the echo was). The whole grid for Delhi is about 3,000 × 2,200 cells ≈ 7 million little squares.

A few words you'll see everywhere:
- **Raster** = that grid of numbers (the spreadsheet-over-a-map).
- **GeoTIFF** = a raster saved as a file that *also remembers where each pixel sits on Earth* (its longitude/latitude). A normal photo doesn't know where it was taken; a GeoTIFF does. That's the whole difference.
- **CRS** (Coordinate Reference System) = the "language" the coordinates are written in (we use plain longitude/latitude, called EPSG:4326). Two maps must speak the same CRS to line up.

### The output of Part A

A single GeoTIFF where every pixel is **1 = flooded** or **0 = dry**. That's the **flood mask**. For Delhi it lit up the Yamuna floodplain almost perfectly — that's how we know the method works.

---

## 3. Part B — Predicting where floods are *likely* (the machine-learning half)

The flood mask only tells us about *one* flood. To make a general *risk* map, we let a computer find the **terrain pattern** of flooded places.

### The intuition first

If I asked *you* "where will it flood?", you'd reason: *low ground, flat ground, near a river or drain, paved-over areas where rain can't soak in.* The model learns exactly those rules — but from data, with no opinions of its own.

### The "ingredient layers" (we call them *features*)

For every pixel we compute a stack of clues. Each is its own raster, lined up on the same grid:

| Feature (clue) | Plain meaning | Why it matters |
|---|---|---|
| **elevation** | height of the ground (a "height map", from a *DEM* = Digital Elevation Model) | water runs downhill into low spots |
| **slope** | how steep the ground is | flat ground drains slowly, so water sits |
| **HAND** | "Height Above Nearest Drainage" — how high you stand above the nearest stream | low above a stream = water reaches you easily |
| **distance to river / drain** | how far to the nearest waterway | closer = more exposed |
| **drainage density** | how much drain length is packed nearby | relates to how water collects/clears |
| **upstream area** | how much land drains *into* this spot | more upstream land = more water funnelled here |
| **sink depth** | how much this spot is a local dip/pothole | dips become puddles (think underpasses) |
| **built-up** | is it concrete/buildings? | concrete can't absorb rain → more runoff |

Stack all these together and you get a **feature stack**: one tall pile of grids, one clue per layer, perfectly aligned so pixel (x, y) means the same ground in every layer. ("**Aligned**" = same size, same CRS, same pixel positions — getting layers aligned is half the work in geospatial.)

### What "training a model" actually means

This is the part people find mysterious. It's just **learning a recipe from examples**.

- We take a big sample of pixels.
- For each, we know its **clues** (the features above) **and** the **answer** (did it flood? — from the SAR flood mask). That known answer is called the **label**.
- The model looks at thousands of (clues → answer) examples and figures out *which combinations of clues tend to go with "flooded."*

That's it. Training = "here are 80,000 solved examples, work out the pattern." After training, you can hand it a *new* pixel's clues and it outputs a number from 0 to 1 = "how flood-prone."

### What is XGBoost (the model we use)?

Picture a **huge team of simple yes/no flowcharts** ("decision trees"). One flowchart might ask: *"Is HAND low? → Is it near a drain? → Is the ground flat? → then probably flooded."* Each flowchart is a bit dumb on its own. **XGBoost** builds hundreds of them, where each new one focuses on fixing the mistakes of the previous ones, then they all **vote**. The combined vote is surprisingly accurate. You don't need to know the maths — just "a big team of question-askers that vote."

### How do we know the model is any good? (AUC)

We hide some pixels from the model during training, then test it on them. The score is **AUC**:

> AUC = the chance that, if you pick one *truly flooded* pixel and one *truly dry* pixel at random, the model gives the flooded one a higher score.

- **0.5** = useless (a coin flip).
- **1.0** = perfect.
- **~0.85–0.93** (what we get) = strong.

### The sneaky trap: why we test *spatially* (the honest bit)

Here's a subtlety that separates a real project from a naive one. Neighbouring pixels are almost identical (the spot 10 m away looks the same). If you let the model **train on a pixel and test on its next-door neighbour**, it's basically *memorising the answer* and the score is fake-high.

So we split the city into big **blocks** and make the model **train on some blocks and get tested on entirely different blocks** it has never seen nearby. That's **spatial cross-validation**, and it's the *honest* score. You'll see two numbers in our results:

```
Random  CV AUC: 0.97   ← cheats (tested on neighbours) — looks great, lies
Spatial CV AUC: 0.92   ← honest (tested on unseen blocks) — the real number we report
```

The gap between them *is* the cheating being removed. Reporting the lower, honest one is the whole point.

### The output of Part B

We run the trained model over **every** pixel and get a **susceptibility surface**: a GeoTIFF where each pixel is 0–1 = "flood-proneness." Shade it yellow→red and you have the risk map.

---

## 4. Part C — Doing it for many cities without copy-pasting (the engineering)

Doing the above once, by hand, in a notebook, is a *student project*. Doing it for 4 cities with one command is an *engineering project*. Here's what each piece is for, in plain terms.

### Config files = a recipe card per city

A **config** is a tiny text file (`configs/city/delhi.yaml`) that says everything specific to one city: its bounding box (the rectangle of map we care about), which dates to use for the before/after radar, etc. To add a city you write **one new recipe card** — you don't copy any code. That's the single biggest "this person can engineer" signal.

### The package + CLI = one button

Instead of clicking through a notebook, the code lives in a proper installed program called `floodml`. You run it from the terminal:

```
floodml run delhi
```

…and it does the whole chain — find flood mask → fetch layers → build features → train → predict — for that city. (**CLI** = Command-Line Interface = "a program you run by typing its name." **Package** = code organised into a reusable program instead of loose scripts.)

### The "MLOps" words, finally demystified

**MLOps** = the boring habits that make machine-learning *reproducible and trustworthy*. You asked what's actually needed; here's each tool as one sentence:

- **MLflow** = an automatic **lab notebook**. Every time you train a model it records the settings, the AUC, and which features mattered — so you never lose track of what gave which result. Run `mlflow ui` to see a table of every experiment.
- **CI (GitHub Actions)** = a **robot that re-checks your code every time you save it** to GitHub. Ours runs a style checker (`ruff`) and a few small tests (`pytest`). If something breaks, you get a red ✗; if it's fine, a green ✓ badge. It catches "you broke it" before anyone else sees.
- **Tests** = tiny programs that check your code does what it should (e.g. "the 4 city configs all load and make sense"). They're the safety net.
- **DVC** *(roadmap, not yet used)* = "git for big files" — versions your large maps/models so anyone can rebuild your exact result.

We **deliberately don't use** Kubernetes / Airflow / feature stores. Those are for big teams running hundreds of models in production. For a solo 4-city project they'd be over-engineering — and *knowing not to use them* is itself a senior signal.

---

## 5. The whole journey on one page

```
   ┌─ Sentinel-1 radar: BEFORE (dry)  ─┐
   │                                   ├──►  subtract  ──►  FLOOD MASK   (1=flooded / 0=dry)
   └─ Sentinel-1 radar: AFTER (flood) ─┘                       │  this is the "answer key" (label)
                                                               ▼
   Elevation, slope, HAND, distance-to-drain,            TRAIN XGBoost
   built-up, sinks, …  ──►  FEATURE STACK  ─────────►  (learn clues → answer)
        (the "clues")                                         │
                                                              ▼
                                                   run on every pixel
                                                              │
                                                              ▼
                                                   SUSCEPTIBILITY MAP  (0–1 flood-proneness)

   All of it driven by one config file per city, run with `floodml run <city>`,
   logged to MLflow, checked by CI.
```

---

## 6. Mini-glossary (skim anytime)

- **SAR / Sentinel-1** — satellite radar that sees through clouds; calm water looks dark to it.
- **Change detection** — comparing a before and after image to spot what changed (the flood).
- **Pixel** — one cell of the map grid; here, a 10 m × 10 m patch of ground.
- **Raster** — a grid of numbers laid over the map.
- **GeoTIFF** — a raster file that knows where each pixel is on Earth.
- **CRS** — the coordinate "language" (we use longitude/latitude).
- **DEM** — Digital Elevation Model = a height map of the ground.
- **HAND** — Height Above Nearest Drainage = how high you are above the nearest stream.
- **Feature** — one input clue for the model (elevation, slope, …).
- **Label** — the known answer for an example (flooded / dry, from the SAR mask).
- **Feature stack** — all the clue-rasters aligned into one pile.
- **Aligned** — same grid/size/CRS so layers line up exactly.
- **Training** — learning the clues→answer pattern from solved examples.
- **XGBoost** — a big team of yes/no flowcharts that vote; the model.
- **AUC** — score from 0.5 (useless) to 1.0 (perfect) for how well it ranks flooded above dry.
- **Spatial cross-validation** — testing on map areas the model never saw nearby, so it can't cheat.
- **Susceptibility surface** — the final 0–1 flood-proneness map.
- **Config** — a per-city recipe card (a small YAML file).
- **CLI / package** — a program you run by typing `floodml`.
- **MLflow** — auto lab-notebook for model runs.
- **CI** — robot that re-checks your code on every save.

---

## 7. What this honestly does *not* do (so you can speak about it confidently)

- It finds **riverine / open-water** flooding well. It is **mostly blind to shallow street-waterlogging between buildings** — radar can't see that. So in dense cities the maps are *susceptibility*, not live street-flood forecasts.
- It's **relative** flood-proneness validated on **one flood per city** — not a calibrated prediction of water depth.
- The **drainage features are approximate** because clean official storm-drain maps barely exist for these cities.

Knowing and *saying* these limits is what makes the project credible rather than overhyped. If someone asks "is this a flood warning system?", the honest answer is: *"No — it's an experimental decision-support map. Here's exactly what it can and can't see, and here's the roadmap to make it stronger."*
