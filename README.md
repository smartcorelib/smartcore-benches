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

`.github/workflows/bench.yml` runs on `repository_dispatch` from smartcore (`event_type: smartcore-dev-push`) and manual `workflow_dispatch`. It does **not** self-trigger on pushes to this repo's `main` — bench history reflects the benchmarked library, not harness edits.

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

Trend charts and JSON history are persisted to the `gh-pages` branch by [benchmark-action/github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark):

- criterion: `gh-pages/dev/`
- iai-callgrind: `gh-pages/iai-dev/`

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