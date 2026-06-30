# urban-flood-ml — design for the next version (v2)

*Plain-English design. It explains the idea from first principles before any technical
choices. If a part can't be explained simply here, that's a sign it isn't thought through
yet. What's **already built** is described in the [README](README.md); the **build order**
is in [ROADMAP.md](ROADMAP.md).*

---

## 1. The problem

When it rains hard, specific streets and low pockets in Indian cities flood. We want to
answer three questions, honestly:

1. **Where** is likely to flood?
2. **How sure** are we about each place?
3. **Can we give any useful warning** before it happens?

The hard part isn't the maths. It's that the honest answer to "where floods" is built on
messy, biased information — and a lot of work in this space hides that to look impressive.
Being upfront about it is the whole point of this one.

## 2. What we have, and what we don't

**What we have (all free):**
- Terrain, paving, population, and rainfall data for every city.
- A patchy map of the major drains.
- Scattered flood reports: traffic-police lists, social posts, news articles.

**What we don't have:**
- A clean, trustworthy list of "here are the places that flooded, with locations." No such
  dataset exists for these cities.
- Fine detail. The best free terrain data sees the ground in 30-metre blocks; a street is
  10–20 metres wide. So we literally cannot see individual curbs and gutters.
- Eyes from space during the monsoon. Clouds hide optical satellites, and radar satellites
  can't see water sitting between buildings.

**The reports we do have are biased**, in known ways: official lists favour big roads, and
news/social posts over-cover dramatic, central, well-off areas. So the design is built
*around* this messiness, not in denial of it.

## 3. The core idea: three layers that talk to each other

Think of it as three maps stacked on top of each other.

### Layer 1 — Squares (the ground) — *built in v1*
Cut each city into roughly 100-metre squares. For every square, measure the things that
make water collect and stay: is it a low spot water drains *into*, how much is paved, how
well the surrounding area drains, how many people live there. These barely change year to
year, so we measure them **once per city** and reuse them.

### Layer 2 — Drains (the real water network) — *new in v2*
Water doesn't flow into squares — it flows into drains, like Delhi's huge **Najafgarh
drain**. So we also keep a map of the major drains and **which squares feed into each
one**. The payoff: if a drain is reported overflowing, **every square that drains into it
becomes riskier at once.** That's the behaviour a plain grid can't capture, and it's what
makes this a model of the city rather than a list of independent squares.

Honest limit, built in: we can get the **shape and names** of drains for free, but **not
their capacity** (how much water they can carry) — that's locked in government records that
require a formal request or payment. So those fields stay **blank**, clearly marked
"unknown," rather than guessed. The drain map's job is to *connect* reports and squares
together, not to do precise water-flow engineering.

### Layer 3 — Reports (what people see) — *partly in v1, deepened in v2*
When flooding is reported, we don't just believe it. We give each report a **confidence
score** built from things we can actually check: was it raining hard there at that time,
did more than one source report it, is this already a known bad spot, and how reliable is
the source (an official list beats a lone post).

**Satellite images are a bonus, never the judge.** When a usable image happens to exist it
can add confidence — but in the monsoon that's rare, so most of the time there's no image.
**A missing image never counts against a report.** And we only even *try* the satellite
check where it could work (a wide open drain), not on a narrow street where we know it
can't see — otherwise we'd quietly trust reports from big roads more than reports from poor
neighbourhoods, which is exactly the bias we're trying to avoid.

## 4. Learning where floods happen

Given a square's features, how likely is it to flood? We start with a **model that explains
its own reasoning** — the kind that can tell us "this square scored high because it's low,
paved, and feeds an overloaded drain." We prefer that over a black box, especially early
on, because we need to trust it.

**Testing it honestly** is the part most projects fake, so we're strict:
- **Don't test on the areas we trained on.** Hold some areas back and test there. (Testing
  on the same area you trained on gives a flattering, fake score.)
- **Train on some cities, then predict an unseen one.** If it still works, it learned *how
  flooding works*. If it falls apart, it just memorised one city. Either answer is worth
  knowing — and it's the headline number this project will stand behind.
- **Use a fair score for rare events.** Floods are rare, so a lazy model that says "no flood
  everywhere" looks 95% right while being useless. We use a measure that doesn't fall for
  that trick, and report it plainly.
- **Account for the biased reports** and state the caveat next to every result.

We start simple on purpose. Only if the simple model leaves a clear, measured gap do we try
a fancier one (for example, one that follows how water moves along the road and drain
network) — and we keep it **only if it measurably beats the simple model** on the honest
tests above. Complexity has to earn its place.

## 5. Looking ahead: forecasts and alerts — *new in v2*

We **cannot** predict a specific street flooding weeks in advance. Nobody can — weather
forecasts become unreliable after about a week. So we're honest about how far we look, and
**our confidence shrinks the further out we go:**

| How far ahead | What we can honestly say | Alert |
|---|---|---|
| **0–3 days** | A real warning for specific places (rain forecasts are good this close). | **Warning** |
| **3–10 days** | A city-wide heads-up — "heavy rain likely," less certain where. | Advisory |
| **2–4 weeks** | Only "the region looks wetter or drier than usual." Never a street-level number. | Soft watch |

A weekly check-in feeds only the longest, vaguest view. Real warnings are reserved for the
near term. We never dress up a 2–4 week outlook as a specific forecast.

## 6. Reading the government PDFs — *new in v2*

A lot of useful drain information is trapped inside government PDF reports. We use a **local
model** (running on our own machine, not a paid service) to pull facts out of them. But
these models make confident-sounding mistakes, so: every extracted fact is marked
**"unverified,"** tagged with the exact document and page it came from, and **a human
approves each fact** before it's allowed to affect the map. We keep it small — a few
scripts, not a sprawling system.

## 7. What the system produces

A map shading each area by how flood-prone it is; a watch-list of the worst spots and
drains; an honest confidence level on each; and tiered alerts (near-term warnings,
longer-term watches).

## 8. What it deliberately can't do (stated up front)

These aren't buried footnotes — they're shown on the product itself:
- 30-metre terrain can't see individual curbs and drains.
- Satellites can't see most monsoon street-flooding.
- Our flood reports lean toward big roads and prominent areas.
- The 2–4 week view is only "wetter or drier," not a flood forecast.

## 9. Engineering foundations

The reproducibility spine is carried over from earlier event-processing work of mine,
retargeted from "events and regions" to "flood reports and drains":
- an **append-only log with deterministic replay**, so every result can be reproduced
  exactly;
- a **score → judge → rate** pipeline, reused to weigh noisy flood reports;
- **grid-to-feature linking**, reused to link squares to the drains they feed;
- an **independent evaluation harness**, reused to test the flood model honestly.

The engine is the same across cities; each city plugs in through its own config file.
