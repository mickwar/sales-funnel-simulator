# Sales Funnel Simulator — Technical Plan

Last updated: 2026-09-04

## 1. Purpose & design philosophy

This app simulates a sales funnel (Leads → Tasks/activities → Opportunities → Closed) with tunable
statistical parameters, then runs machine learning against the simulated history to demonstrate three
things at once:

1. ML (regression / random forest) finds real, quantified effects that a dashboard of charts will miss
   or misattribute.
2. Dashboards still have a role — they're good at *surfacing candidate levers* — but aren't sufficient
   on their own.
3. Being intentional about which data you look at beats "throw everything at the wall."

The single most useful design decision in this project: **you control the generative process, so you
know the ground truth.** Every "statistical effect" (rep skill, industry modifier, speed-to-lead, etc.)
is a parameter you set. That means every analysis in the app can be scored against a known answer —
"I configured rep A to close 15% more often; the fitted model estimated +14.2% (p<0.001)." That
recovery story *is* the demo. Design the effects system and the ML layer so this comparison is easy to
show on screen.

## 2. Decisions from initial scoping

- **Stack:** Python-only, Streamlit-first MVP (matches your R/Python/ML background, no new frontend
  stack to learn, fastest path to something interactive).
- **Storage:** Postgres, no BigQuery/Looker layer for now (kept lean). See §5 — a DuckDB analytics
  layer sits on top of Postgres instead, at effectively no infra cost.
- **Audience:** Solo use to start, but the data model is multi-tenant from day one (every row carries a
  `run_id`) so opening it up to a few concurrent users later is a hosting/config change, not a rebuild.

## 3. Data model — entities & event sourcing

### Core entities
- **Account** — a company (industry, employee count, revenue band, ICP-fit score).
- **Lead** — an individual/contact tied to an Account, with a lifecycle stage.
- **Task** — an activity a rep performs (call, email, meeting, text), tied to a Lead or Opportunity.
- **Opportunity** — a deal in progress: stage, probability, deal size, expected close date.
- **Rep** — a salesperson: capacity (tasks/day), assigned quota, tenure/skill parameters.
- **RepQuota** — quota period, target ($ or count), attainment.
- **Run/Scenario** — one named parameter set + its simulated history. This is what makes
  "save scenario A vs B" possible later.

### Event sourcing pattern
Two tiers of tables, not one:

1. `events` — append-only, full history, never updated or deleted (except by the retention job in
   §5). Columns: `event_id, run_id, entity_type, entity_id, event_type, sim_time, payload jsonb,
   actor_rep_id, recorded_at`. This is your audit trail and "time travel" source — e.g., "show the
   funnel as it stood on simulated day 47" is a filter on this table, not a special feature.
2. `<entity>_current` projection tables (`leads_current`, `opportunities_current`, `tasks_current`,
   `reps_current`, ...) — upserted as each event lands. Everyday reads (dashboards, ML feature
   pulls) hit these, never the full event log. This is the difference between a query that's
   instant and one that replays hundreds of thousands of events every time someone opens a chart.

Rebuilding a projection from events should always be possible (rerun the event stream through the
same upsert logic) — that's your safety net if a projection table ever drifts or you change its shape.

## 4. Simulation engine

- **Time-stepped, not full discrete-event.** "Fast forward N days" advances a daily (or hourly, if you
  want finer speed-to-lead effects) tick. Per tick: sample new leads (Poisson arrival process is the
  natural choice for lead gen), assign tasks up to each rep's remaining capacity, sample task outcomes,
  update opportunity stage/probability, log every change as an event. Weekends/holidays/PTO are simply
  not modeled — every simulated day is treated the same.
- **Vectorize generation.** Don't loop row-by-row in Python. Generate a day's (or a whole
  fast-forward's) worth of draws as numpy arrays / a pandas DataFrame using `numpy.random` /
  `scipy.stats`, then bulk-load (§5). Row-by-row generation is the difference between a fast-forward
  that takes 2 seconds and one that takes 2 minutes at the volumes you're targeting.
- **Effects system — do this in logit (log-odds) space, not raw probability space.** If "rep A: +15%
  close rate" and "industry B: −10% close rate" and "fast first touch: +20% close rate" all apply to
  the same opportunity, multiplying/adding them in raw probability space breaks down (you can exceed
  100% or go negative). Instead: represent the base close probability as a logit, apply each effect as
  an additive shift on that logit scale, then convert back to a probability with the sigmoid function.
  This is exactly a logistic regression's functional form — which means when you later *fit* a logistic
  regression to the simulated outcomes, you're fitting a correctly-specified model, and the recovered
  coefficients should closely match the true effects you configured. That's a clean, honest demo instead
  of a rigged one.
- **Distribution parameterization.** Let users pick a family (Normal, Gamma, Beta, Poisson, Lognormal —
  add Negative Binomial too, see below) and specify mean & variance; convert to the distribution's
  native parameters via method-of-moments (e.g., Gamma: shape = mean²/var, scale = var/mean; Beta:
  solve α, β from mean & var, valid only for mean ∈ (0,1)). Validate/clamp to each distribution's valid
  domain and surface a clear error rather than silently producing garbage.
  - **Watch this constraint:** Poisson has one free parameter — mean = variance, always. If you let
    someone set mean and variance independently for a count field (e.g., tasks per day), either ignore
    their variance input for Poisson and say so, or offer **Negative Binomial** as the alternative
    (it has independent mean/variance and is the standard model for over-dispersed counts anyway —
    task/lead counts in real sales data usually are over-dispersed).

## 5. Storage & scale plan

This section directly answers your question: hundreds of thousands of simulated rows per session,
several concurrent users, where should it live and how do you keep it fast?

**Short answer: server-side Postgres, not the browser, not per-user local files.** Streamlit is
server-rendered Python — there's no realistic client-side ("store it in the user's browser") option
without switching to a JS/WASM frontend, and you don't want hundreds of thousands of rows sitting in
Streamlit's per-session memory anyway (that's a memory leak waiting to happen across concurrent
sessions).

**Multi-tenancy without a redesign.** Every event and every projection row carries `run_id` (a UUID).
One shared Postgres instance, one schema. `run_id` (and later `user_id`, if you add accounts) is just
another filter column — always in the WHERE clause, always the leading column in your indexes. You are
already multi-tenant on day one; "several users at once" later is a hosting change, not a data model
change.

**Bulk loading is the single biggest performance lever.** Never insert hundreds of thousands of rows
one at a time — not via an ORM creating one object per row, not via a Python loop calling `execute()`
per row. Build a day's (or a whole fast-forward's) worth of events as a pandas DataFrame in memory,
then load it with Postgres `COPY` (`psycopg` v3's `copy()` context manager, or at minimum
`pandas.DataFrame.to_sql(..., method="multi", chunksize=...)`). COPY turns "insert 300k rows" from a
multi-minute operation into low single-digit seconds.

**Analytics layer: DuckDB, not BigQuery.** Add `duckdb` (pure Python package, nothing to host, free).
Either query Postgres directly from DuckDB via its `postgres_scanner` extension, or export a run's data
to Parquet once you're done writing to it and query that. DuckDB will chew through hundreds of
thousands to low millions of rows for feature-building and aggregation far faster than issuing many
small queries against Postgres row-by-row — this gets you a genuine "operational DB → analytical
engine" split (the real architecture pattern you'd use with Salesforce → warehouse → BI) without
standing up or paying for BigQuery. You can always bolt on BigQuery + Looker Studio later (see
Phase 5, §10) if you decide you want that explicit showcase layer.

**Rough sizing.** A JSONB event row with its indexes is probably ~0.5–1KB. 300k rows ≈ 150–300MB for
one big run. A handful of concurrent large runs can approach 1–2GB — past most Postgres free tiers
(Supabase free ≈ 500MB; Neon free ≈ 0.5GB active / a few GB total depending on plan). Budget roughly
$25/mo for a small paid Postgres tier (Neon or Supabase "Pro") once you're doing real multi-user
testing rather than the free tier.

**Retention.** This is a playground, not a system of record. Add a cheap cleanup job — purge or
archive-to-Parquet any run untouched for N days — so storage doesn't grow without bound. Offer an
"export this run" (CSV/Parquet download) button before a run is purged if people want to keep results.

**Concurrency & compute.** Streamlit Community Cloud's free tier is modest (roughly 1 CPU / ~1GB RAM
shared, and the app can sleep when idle). Fine solo. The first real constraint you'll hit with "several
people fast-forwarding big runs at once" is compute on the app server, not Postgres. The fix is moving
the Streamlit app to a small paid container (Render or Fly.io, ~$7–25/mo) — the Postgres + DuckDB
design underneath doesn't change.

**The real ceiling — and how to avoid getting boxed in.** If you ever want *many* simultaneous users
each running independent large simulations truly concurrently (not just "several"), Streamlit's
one-Python-process-per-session execution model becomes the bottleneck before Postgres does. The
graduation path is: pull the simulation engine into plain Python functions/classes (not entangled with
Streamlit callbacks or `st.session_state`), wrap them in a stateless FastAPI service, and optionally
hand large fast-forwards to a background worker (RQ or Celery + Redis) so a big run doesn't block the
UI. Any frontend — Streamlit today, a real website later — just calls the API. You don't need this on
day one; the only thing to do *now* to keep this option open is write the simulation core as plain,
UI-agnostic Python from the start.

## 6. ML / analysis layer

- **Auto-run after each fast-forward** (or on demand): fit (a) OLS/logistic regression via
  `statsmodels` — not scikit-learn alone — because you need coefficients *with* p-values and confidence
  intervals for the "highlight significant effects" requirement; and (b) `RandomForestClassifier` /
  `RandomForestRegressor` via scikit-learn, using permutation importance (`sklearn.inspection`) or SHAP
  for effect attribution — both more trustworthy than default impurity-based importances.
- **"Highlight significant results":** auto-generate a plain-English summary of the top N statistically
  significant effects, sorted by effect size among those that clear significance — e.g. "Leads
  contacted within 5 minutes convert 34% more often (p<0.001)."
- **Correct for multiple comparisons.** If you're scanning many candidate features for "is this
  significant," apply a correction (Benjamini-Hochberg FDR is a reasonable default) rather than reading
  raw p-values off a dozen tests — otherwise you'll surface spurious effects, which undercuts the whole
  "be intentional, not throw-everything-at-the-wall" thesis.
- **The ground-truth recovery demo** (see §1): show the model's estimated effect next to the true
  configured effect side by side. Pair this with a naive dashboard view of the same relationship
  (e.g., raw win rate by rep, unadjusted) to show where the simple view is misleading — that
  comparison *is* the product's core argument.
- **Rep performance analysis — adjust for confounding.** Don't just rank raw win rate or quota
  attainment; that's confounded by which leads each rep happened to get. Compute performance as a
  regression-adjusted residual (or stratify by ICP-fit/lead source) so "above/below average" reflects
  skill, not the luck of lead assignment. Show "raw ranking" next to "adjusted ranking" — this is a
  real RevOps best practice and doubles as another illustration of your thesis.

## 7. Frontend / UX (Streamlit)

- Streamlit + Plotly for everything: funnel visualizations, distribution PDF/PMF previews, SHAP
  summary/waterfall plots, rep leaderboards.
- **Distribution picker widget** — one reusable component: pick a family, enter mean & variance (or
  native parameters), see a live PDF/PMF preview update as sliders move. Compute the preview
  server-side with `scipy.stats` on every rerun (cheap at this scale); validate/clamp to the
  distribution's valid domain and show a clear message when the input is out of range (e.g., Beta mean
  outside (0,1)).
- **Scenario management.** Let users name and save parameter sets against the `runs` table already in
  the data model — "compare scenario A vs B" (e.g., "what if speed-to-lead effect were doubled?") is a
  natural, high-value feature for the executive-playground goal, and it's nearly free once `run_id` is
  already threaded through everything.
- **Fast-forward UX.** A day/date-range input plus a progress indicator (`st.status`) while vectorized
  generation + bulk load runs; target a few seconds even for large runs so it feels interactive.
- Streamlit reruns the whole script on interaction — keep large DataFrames out of raw
  `st.session_state`; query fresh from Postgres/DuckDB per view and cache aggregates with
  `st.cache_data`.

## 8. Tech stack summary

| Layer | Choice | Why |
|---|---|---|
| App framework | Streamlit | Pure Python, matches your background, fast to build, free hosting to start |
| Charts | Plotly | Interactive, good distribution/PDF plots, SHAP-friendly, works natively in Streamlit |
| Simulation core | numpy, scipy.stats | Vectorized sampling, all the distribution families you need |
| Database | Postgres | Real transactional store, handles millions of rows fine with proper indexing |
| DB driver / loading | psycopg (v3) | Native `COPY` support for fast bulk loads |
| Analytics engine | DuckDB | Free, in-process, fast on hundreds of thousands–millions of rows, can query Postgres directly |
| Regression | statsmodels | Coefficients with p-values/CIs, not just point estimates |
| Random forest / importances | scikit-learn (+ optional `shap`) | Standard, well-supported, permutation importance built in |
| Data wrangling | pandas | Glue between Postgres/DuckDB and scikit-learn/statsmodels |
| Prototyping (optional) | R or a Jupyter notebook | Validate distribution choices / effect sizes before wiring into the app |

## 9. Hosting & cost estimate

| Item | Option | Est. monthly cost |
|---|---|---|
| Streamlit app hosting | Streamlit Community Cloud | $0 to start |
| Streamlit app hosting (scale-up) | Render / Fly.io small container | $7–25 |
| Postgres | Neon or Supabase free tier | $0 to start |
| Postgres (scale-up) | Neon or Supabase Pro | ~$25 |
| DuckDB | N/A — in-process library | $0 |
| Domain (optional) | Any registrar | ~$1/mo (~$12/yr) |
| **Total, solo/MVP** | | **$0** |
| **Total, several concurrent users** | | **~$30–50/mo** |

## 10. Phased roadmap

- **Phase 0 — Prototype the stats, not the app.** Validate the effects system (logit-space
  composition) and distribution parameterization (method-of-moments conversions) in a notebook (R or
  Jupyter) before building any UI around them. Cheap to get this wrong early, expensive to get it wrong
  after the app is built on top of it.
- **Phase 1 — MVP.** Streamlit app, Postgres, core entities, basic fast-forward, basic charts. Solo use.
- **Phase 2 — ML layer.** Regression + random forest, significance highlighting, ground-truth recovery
  view, rep performance analysis (raw vs. adjusted).
- **Phase 3 — Polish.** Scenario save/compare, distribution picker refinement, retention/cleanup job.
- **Phase 4 — Open it up.** Move hosting to a small paid always-on container, load-test with a few
  concurrent large runs, add lightweight access control if needed. Revisit the FastAPI graduation only
  if compute is actually the bottleneck at that point.
- **Phase 5 — Optional/stretch.** Add the BigQuery + Looker Studio layer if you later decide you want
  an explicit "traditional BI dashboard" comparison point for the thesis, or want it as a resume line.
  Not required for the core product to work.

## 11. Key challenges & risks

- **Effects composability.** Raw-probability multiplication/addition breaks down with several stacked
  effects — use logit space (§4).
- **Poisson's variance = mean constraint** — offer Negative Binomial for count fields where users
  expect independent control (§4).
- **Generation performance at scale.** Row-by-row Python loops will not keep "fast forward" fast at
  hundreds of thousands of rows — vectorize and bulk-load (§4, §5).
- **Event-log replay cost.** Don't rebuild current state from the full event log on every read —
  maintain projection tables (§3).
- **ML validity.** Guard against leakage (don't feed post-outcome fields back in as predictors),
  correct for multiple comparisons (§6), and expect random forest to need enough rows per
  segment/industry/rep to give stable importances — check sample sizes before trusting a "significant"
  result on a thin slice.
- **Confounded rep comparisons.** Raw win-rate ranking rewards whoever got the best leads, not the best
  rep — adjust for lead/opportunity difficulty (§6).
- **Reproducibility.** Seed the RNG per run so a saved scenario can be reproduced or audited later.
- **Storage growth.** Add retention/cleanup from the start rather than bolting it on after storage
  becomes a problem (§5).
- **Streamlit's concurrency ceiling.** Fine for "several" users; write the simulation core as
  UI-agnostic Python so a future move to FastAPI doesn't mean a rewrite (§5).

## 12. Open questions to settle before/while building

- Daily ticks are the default assumption for "fast forward" — confirm that's fine, or whether some
  effects (e.g., speed-to-lead within minutes) need finer-grained sub-day resolution for at least the
  first task after a lead comes in.
- Do you want a short R/Jupyter prototyping pass on the effects system and distributions before
  touching Streamlit at all (Phase 0), or go straight to the app?
- Any specific industries/ICP archetypes you want seeded as realistic defaults, versus fully
  user-configurable from a blank slate?

## 13. Code organization — repo structure

**Single repo (monorepo), not split by function/layer.** The simulation engine, storage layer,
ML/analysis layer, and UI are tightly coupled through one shared thing — the data model (entity
schemas, event types, the effects system) — and that model will keep changing through Phases 0–3.
Splitting into separate repos turns every schema change into a multi-repo coordination problem
(version-pin an internal package, bump it, update the consumer, repeat) for no payoff at solo-developer
scale: there's no team-ownership boundary or independent-deploy-cadence reason to pay that cost yet.

The separation that actually matters is package boundaries *inside* one repo — a "modular monolith":

```
sales-funnel-simulation/
  pyproject.toml
  PLAN.md
  src/funnel_sim/
    simulation/    # engine: entities, effects system, distributions, fast-forward — UI-agnostic
    storage/       # Postgres schema, event log writer, projections, bulk COPY loader, DuckDB queries
    analysis/      # regression, random forest, significance highlighting, rep performance
    app/           # Streamlit UI — imports simulation/storage/analysis, never the reverse
  notebooks/       # Phase 0 prototyping, kept separate from app code
  migrations/      # schema migrations
  tests/
  scripts/         # retention/cleanup job, seed data, etc.
```

The rule that makes this work: `simulation/`, `storage/`, and `analysis/` never import from `app/` —
dependency flows one direction only. That single rule is what delivers the FastAPI graduation path
from §5/§10 for free: if Streamlit is ever outgrown, `app/` gets swapped or added to, and the core
packages don't change. It also gives clean unit-test boundaries per package without needing separate
CI pipelines.

**Notebooks in git are noisy** (diffs full of cell output) — add `nbstripout` as a pre-commit hook so
outputs never get committed. Treat notebooks as scratch exploration for Phase 0: once an effect or
distribution choice is validated there, port the logic into `simulation/` as a real module rather than
letting the notebook stay the source of truth.

**When to reconsider multi-repo:** only if the simulation engine becomes something worth reusing or
open-sourcing independently of this app. That's a cheap split later (`git filter-repo`/`subtree split`
on an already-clean package boundary), so it's not a reason to start multi-repo now.

### Future packages under `src/`

Right now `src/` holds a single package, `funnel_sim` — that's correct, and more packages shouldn't
be added preemptively. The trigger that justifies a new top-level package is needing to install or
deploy it *independently, with a different dependency footprint* — not "this feels like a different
concern" (that's what the `simulation/`/`storage/`/`analysis/`/`app/` submodules inside `funnel_sim`
are already for). Realistic future candidates:

- **`funnel_sim_core`** — if the simulation engine (`simulation/`: entities, effects system,
  distributions, already zero-dependency on DB/UI) is ever worth reusing or open-sourcing
  independently (see the multi-repo trigger at the end of this section), pull it into its own
  top-level package first, still inside this repo, with `funnel_sim` depending on it. That validates
  the boundary is actually clean before ever doing a real repo split.
- **`funnel_sim_api`** — if the Phase 4/5 FastAPI graduation happens and the API and Streamlit UI need
  to deploy as separate containers (API doesn't need `streamlit`/`plotly`; UI doesn't need
  `fastapi`/`uvicorn`), that dependency-footprint split is the trigger — not "API vs UI" as a concept.
- **A CLI** — for batch operations (retention/cleanup job, seed data, and a good one to plan for: an
  automated regression test that runs a batch of simulations and checks the fitted model recovers
  known configured effects within tolerance — a CI check on the core thesis). Usually doesn't need its
  own package — a couple of console-script entry points in `pyproject.toml` pointing at functions
  already in `funnel_sim` are enough, unless it accumulates real CLI-specific dependency weight.

Explicitly not planned: a separate shared-schemas package (`storage/` holds entity/event schemas fine
at this scale) or a client SDK package (no external consumers yet).
