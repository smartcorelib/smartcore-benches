<p align="center">
  <a href="https://smartcorelib.org">
    <img src="smartcore.svg" width="450" alt="smartcore">
  </a>
</p>
<p align="center">
  <strong>
    <a href="https://smartcorelib.org">User guide</a> | <a href="https://docs.rs/smartcore/">API</a> | <a href="https://github.com/smartcorelib/smartcore-jupyter">Notebooks</a>
  </strong>
</p>

-----

<p align="center">
  <b>Machine Learning in Rust</b>
</p>

-----

Benchmark harness for [smartcore](https://github.com/smartcorelib/smartcore). Covers the five hot paths flagged in [smartcore#407](https://github.com/smartcorelib/smartcore/issues/407) as well as the legacy algorithm benches, and wires CI to run them on every `development` push.

> **Live trend charts:** [criterion (wall-clock)](https://smartcorelib.github.io/smartcore-benches/dev/) · [iai-callgrind (instruction count, deterministic gate)](https://smartcorelib.github.io/smartcore-benches/iai-dev/) — see [Results](#results) for how to read them.

## Local run

```bash
# Compile-check every bench without running the timed loops:
cargo bench --no-run

# Run a single criterion bench (fast, small grid) locally:
cargo bench --bench distance

# Run the full criterion suite (some grids are large — minutes to hours):
cargo bench

# iai-callgrind benches are Linux-only (Valgrind) and run each input once:
#   cargo bench --bench iai_matmul
```

The `smartcore` dependency resolves from crates.io (latest 0.6.x) by default — `cargo bench` pulls the published release. To bench a local smartcore checkout instead, temporarily swap the dep line in `Cargo.toml`:

```toml
# Default (crates.io):
smartcore = { version = "0.6", features = ["ndarray-bindings"] }

# Local checkout:
# smartcore = { path = "../smartcore", features = ["ndarray-bindings"] }
```

Then `cd ../smartcore && git checkout development && cd -` and run `cargo bench --bench matmul`.

## Benchmarks

### Hot-path benches (criterion)

| Bench | Hot path | Sizes |
|---|---|---|
| `matmul` | `Array2::matmul` (`arrays.rs:1117`) | 64², 256², 1024² |
| `ab` | `Array2::ab` — 4 transpose combos (`arrays.rs:1143`; `high_order.rs:21` is currently unimplemented, see issue) | 256² |
| `svd` | `SVDDecomposable::svd` (`svd.rs:71`) — DenseMatrix + ndarray | 128², 256×64 |
| `cover_tree` | `CoverTree::new`/`find` (`cover_tree.rs:37`) vs `LinearKNNSearch` | 1k/10k/100k × 10/100 |
| `iterator_mut` | `DenseMatrix::iterator_mut` (`matrix.rs:545`) + `MutView::iterator_mut` (`matrix.rs:254`) — the #368 refactor surface | 1024², 4096² |

### Algorithm benches (criterion, legacy)

`distance`, `fastpair`, `linear`, `naive_bayes`, `svc`.

### Deterministic gate (iai-callgrind, Linux-only)

`iai_matmul`, `iai_ab`, `iai_svd`, `iai_cover_tree`, `iai_iterator_mut` mirror the hot-path benches under Valgrind's instruction counter. Counts are machine-independent, so a tight `120%` alert threshold is safe on GitHub-hosted runners. See `.github/workflows/bench.yml` (`iai` job).

## CI

`.github/workflows/bench.yml` runs on `repository_dispatch` from smartcore (`event_type: smartcore-dev-push`), on push to this repo's `main` (so harness edits re-record a baseline against the same published smartcore), on a nightly cron, and on manual `workflow_dispatch`.

### Dispatch contract

smartcore's CI fires `repository_dispatch` on every `development` push with this payload:

```json
{
  "event_type": "smartcore-dev-push",
  "client_payload": {
    "sha": "<commit SHA>",
    "ref": "development",
    "actor": "<pusher>",
    "run_id": "<smartcore run id>",
    "pr": "<PR number or null>"
  }
}
```

smartcore is resolved from crates.io (latest 0.6.x — currently 0.6.3) on every run, so the recorded history tracks the published release. The `repository_dispatch` payload carries the smartcore commit SHA only for the status-check step (not for checkout); the nightly cron provides a baseline when smartcore has no recent pushes.

### Jobs

| Job | Runner | Tool | Role |
|---|---|---|---|
| `criterion` | ubuntu-latest | criterion (wall-clock) | Advisory. 200% alert threshold — noisy on shared runners, posts a comment but does not fail the job. |
| `iai` | ubuntu-latest | iai-callgrind (instruction count) | Deterministic gate. 120% alert threshold, `fail-on-alert: true`. Posts a `success`/`failure` status check back to the smartcore commit. |

### Results

There are five places to read benchmark results, each answering a different question.

#### 1. Live trend charts (gh-pages, web)

[benchmark-action/github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark) renders an interactive chart page per tool on the `gh-pages` branch, published via GitHub Pages:

| Tool | Chart URL | What it plots | Direction | Alert |
|---|---|---|---|---|
| criterion (wall-clock) | <https://smartcorelib.github.io/smartcore-benches/dev/> | `cargo bench` wall-clock time per bench (ns/iter) over time | lower = better | 200% — advisory, posts a comment, does not fail CI |
| iai-callgrind (instruction count) | <https://smartcorelib.github.io/smartcore-benches/iai-dev/> | instructions retired (`Ir`) per bench — deterministic, machine-independent | lower = better | 120% — fails the `iai` job + status check |

Open the URLs above in a browser. Each page shows a searchable line chart (`data.js` is the raw history) with one series per benchmark name (e.g. `matmul/1024`, `iai_matmul::matmul::bench_matmul_256`). Hover for the value, range, and commit that produced each point. The iai page is the one to watch for regressions: instruction counts are deterministic on GitHub-hosted runners, so a >20% jump is a real code change, not noise.

#### 2. Per-version criterion snapshots (`benches-results/`)

The `criterion` job also records one frozen `criterion.json` per published smartcore version, committed to `main`:

```
benches-results/v<smartcore-version>/criterion.json
```

e.g. `benches-results/v0.6.4/criterion.json`. This is the **point-in-time** record of "what the published release measured" — useful for diffing two smartcore releases against each other rather than tracking the rolling trend. Storage is idempotent (a version is recorded at most once); see [`benches-results/README.md`](benches-results/README.md) for the storage contract.

#### 3. Raw JSON artifacts (per CI run)

Each run uploads its raw bench output as a CI artifact (retained per the repo's artifact policy):

- `criterion-json` → `bench-output/criterion.json` (bencher format)
- `iai-json` → `bench-output/iai.ndjson` (raw iai NDJSON) + `bench-output/iai.json` (the `customSmallerIsBetter` array the adapter produced)

Download from a run page, or via `gh`:

```bash
gh run download <run-id> --repo smartcorelib/smartcore-benches -n iai-json -D ./iai-out
gh run download <run-id> --repo smartcorelib/smartcore-benches -n criterion-json -D ./crit-out
```

The `iai.ndjson` is the upstream `summary.v3` schema (one `BenchmarkSummary` per line); `iai.json` is the `[{name,unit,value}]` array fed to benchmark-action. See `scripts/iai_to_benchmark_action.py` for the projection (`Ir` → instructions retired).

#### 4. CI job status

The run page itself shows pass/fail per job:

- `iai` job green ⇒ instruction counts within 120% of baseline (the merge gate).
- `criterion` job green ⇒ ran successfully; it does **not** gate (advisory only, 200% threshold).
- The `iai` job also posts a `smartcore-benches / iai-callgrind` status check back to the smartcore commit that triggered the run (via `repository_dispatch`), so smartcore's branch protection can require it.

#### 5. Alert comments on commits/PRs

Both jobs have `comment-on-alert: true`, so benchmark-action posts a comment on the triggering commit when a >threshold regression is detected — look for the bot comment on the smartcore commit for the iai signal.

> **Re-running a failed job** reuses the workflow file pinned to the triggering SHA, so re-running an old run whose SHA predates a fix will re-fail the same way. To validate a fix, trigger a fresh run from the new tip: `gh workflow run bench.yml --repo smartcorelib/smartcore-benches --ref main`.

### Required secrets

| Secret | Used by | Purpose |
|---|---|---|
| `BENCHES_STATUS_PAT` | `Post status check back to smartcore` steps | Fine-grained PAT with `commit_status:write` on `smartcorelib/smartcore`. Lets this repo post the `smartcore-benches / iai-callgrind` status check that smartcore's branch protection can require. |

## Contributing

This repo has no separate CONTRIBUTING file — follow smartcore's [CONTRIBUTING](https://github.com/smartcorelib/smartcore/blob/development/.github/CONTRIBUTING.md) and [AGENTS.md](https://github.com/smartcorelib/smartcore/blob/development/AGENTS.md) conventions (edition 2024, no `unsafe`, `#[expect]` over `#[allow]`, bind intermediates before iterating in edition-2024 tail positions). Bench code targets the same dep versions as smartcore's Cargo.toml so cargo unifies a single copy of each crate across the graph.

A pre-commit hook (`.githooks/pre-commit`) mirrors the `lint.yml` gate locally — it runs `cargo fmt --check` and `cargo clippy -Dwarnings` before each commit. Install it after a fresh clone:

```bash
git config core.hooksPath .githooks
```

Bypass with `git commit --no-verify` when intentionally committing mid-edit.