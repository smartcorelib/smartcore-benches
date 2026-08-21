# AGENTS.md

Agent-focused guidance for working on the `smartcore-benches` benchmark harness.

Always use ASD-STE100 Simplified Technical English

## Project basics

- **What this repo is**: Criterion + iai-callgrind benchmark harness for [smartcore](https://github.com/smartcorelib/smartcore). It is *not* smartcore itself; it pulls a published smartcore release from crates.io (latest `0.6.x`) and measures the hot paths flagged in [smartcore#407](https://github.com/smartcorelib/smartcore/issues/407).
- **Language / edition**: Rust 2024 (MSRV 1.85), mirroring smartcore's toolchain.
- **Repository**: https://github.com/smartcorelib/smartcore-benches
- **Default branch**: `main`. There is no `development` branch here.
- **License**: Apache-2.0.
- **Sibling repos**: `smartcore` and `smartcore-jupyter` live alongside this one under `smartcore-lib/`; follow smartcore's [AGENTS.md](https://github.com/smartcorelib/smartcore/blob/development/AGENTS.md) and CONTRIBUTING for algorithm-side conventions.

## Build and run benches

```bash
# Compile-check every bench without running timed loops (portable gate):
cargo bench --no-run

# Run a single criterion bench (fast, small grid) locally:
cargo bench --bench distance

# Run the full criterion suite (some grids are large — minutes to hours):
cargo bench

# iai-callgrind benches are Linux-only (Valgrind) and run each input once:
cargo bench --bench iai_matmul
```

`cargo bench --no-run` is a portable gate (the iai benches have a `#[cfg(not(target_os = "linux"))] fn main()` stub), but the actual Valgrind run only happens on Linux.

### Local smartcore checkout

The `smartcore` dep resolves from crates.io by default. To bench a local smartcore checkout, temporarily swap the dep line in `Cargo.toml`:

```toml
# Default (crates.io):
smartcore = { version = "0.6", features = ["ndarray-bindings"] }
# Local checkout:
# smartcore = { path = "../smartcore", features = ["ndarray-bindings"] }
```

Then `cd ../smartcore && git checkout development && cd -` and run `cargo bench --bench matmul`.

## Lint and format (enforced in CI)

```bash
cargo fmt --all -- --check
cargo clippy -Dwarnings
```

A pre-commit hook (`.githooks/pre-commit`) mirrors the `lint.yml` gate locally. Install it after a fresh clone:

```bash
git config core.hooksPath .githooks
```

Bypass with `git commit --no-verify` when intentionally committing mid-edit.

## iai-callgrind — the runner binary (IMPORTANT)

`iai-callgrind` 0.14 split the runtime into a **separate `iai-callgrind-runner` binary** that is *not* auto-installed by `cargo bench`. `cargo install iai-callgrind` installs the **library crate** only; the runner is its own published crate `iai-callgrind-runner` (see `Cargo.lock`). Symptom when the runner is missing (seen in run 31322933166, job 93268654705):

```
iai-callgrind: Error: The following errors occurred:
--> Error in iai_matmul: Failed to run benchmarks: No such file or directory (os error 2).
Is iai-callgrind-runner installed and iai-callgrind-runner in your $PATH?
You can set the environment variable IAI_CALLGRIND_RUNNER to the absolute path of the iai-callgrind-runner executable.
```

`cargo bench --no-run` compiles the bench crate but does **not** install the runner. Bench binaries then exec `iai-callgrind-runner` from `$PATH` at run time and fail immediately — no callgrind run happens, no JSON is produced.

Fix locally / in CI:

```bash
# Install the runner at the version Cargo.lock pins (must match the lib dep).
cargo install iai-callgrind-runner --version 0.14.2 --locked
# Or point the env var at a known binary path:
export IAI_CALLGRIND_RUNNER=/path/to/iai-callgrind-runner
```

The runner version **must match** the `iai-callgrind` library version (see `Cargo.lock`). A mismatch produces confusing "different version than the runner" errors at startup.

Valgrind itself also needs installing on Linux runners: `sudo apt-get update && sudo apt-get install -y valgrind`.

## iai-callgrind `--output-format=json` semantics (IMPORTANT)

`iai-callgrind` 0.14's `--output-format=json` (env `IAI_CALLGRIND_OUTPUT_FORMAT`) prints one **`BenchmarkSummary` object per benchmark line** on stdout (NDJSON — one JSON document per line; *not* a JSON array). All other iai/callgrind logging goes to stderr. The JSON schema is `iai-callgrind-runner/schemas/summary.v3.schema.json` in the upstream repo (`v0.14.2` tag). Combine lines into an array with `jq -s` if needed:

```
cargo bench -- --output-format=json | jq -s
```

The instruction count lives at (note: **total.summary.Ir.metrics**, *not* `events.Ir.metrics` — `events` is per-segment and a bench can have many segments):

```
summary.callgrind_summary.callgrind_run.total.summary.Ir.metrics
```

`callgrind_run` has two relevant metric locations:
- `total.summary.<EventKind>.metrics` — the aggregated headline number across all segments (threads/subprocesses/parts). This is the value the `iai` gate tracks.
- `segments[].events.<EventKind>.metrics` — per-segment breakdown; `scripts/iai_to_benchmark_action.py` falls back to `segments[0]` only if `total` is absent.

`total.summary` and `segments[].events` are `MetricsSummary_for_EventKind` — a map keyed by `EventKind` string (`"Ir"` = instructions retired, the default event; also `"L1hits"`, `"LLhits"`, `"RamHits"`, `"EstimatedCycles"`, etc.). The most stable / machine-independent signal is `"Ir"`, which is what the `iai` gate tracks.

`metrics` is an `EitherOrBoth_for_uint64`:
- `{"Left": n}` — new-only (first run, no prior baseline on disk)
- `{"Right": o}` — old-only (degenerate; the new run produced nothing)
- `{"Both": [n, o]}` — new `n` and old `o`; project `n` = `Both[0]` for the current value, `o` = `Both[1]` for the comparison baseline

`benchmark-action/github-action-benchmark@v1` has **no `iai-callgrind` tool key** — its v1 API accepts only `cargo, go, benchmarkjs, benchmarkluau, pytest, googlecpp, catch2, julia, jmh, benchmarkdotnet, customBiggerIsBetter, customSmallerIsBetter`. Passing `tool: iai-callgrind` fails the job with:

```
Invalid value 'iai-callgrind' for 'tool' input. It must be one of cargo,go,benchmarkjs,benchmarkluau,pytest,googlecpp,catch2,julia,jmh,benchmarkdotnet,customBiggerIsBetter,customSmallerIsBetter
```

So the `iai` CI job maps iai output through `tool: customSmallerIsBetter`, which expects a JSON **array** of entries shaped `{"name", "unit", "value"}` (optional `range`, `extra`; see the action README §customBiggerIsBetter/customSmallerIsBetter). The NDJSON→array conversion is done by `scripts/iai_to_benchmark_action.py` (run it with `--self-test /tmp/sc-iai-test` to verify; it's stdlib-only). It projects each `BenchmarkSummary`'s `module_path` (or `function_name`) → `name`, `Ir` `Left`/`Both[0]` → `value`, and fixes `unit: "Instructions"`. Instruction counts are deterministic and smaller-is-better, so a tight `120%` alert threshold is safe on GitHub-hosted runners.

### Failure cascade when the runner is missing (debugging trail)

When the runner isn't on `$PATH`, the failure cascades through three confusing error layers — know the chain so you don't chase the wrong layer:

1. **Runner missing** (root cause): `iai-callgrind: Error: Failed to run benchmarks: No such file or directory (os error 2).` in the "Run iai-callgrind benchmarks" step. The bench binary exits 1; `tee` writes nothing to the `.json` file.
2. **Empty artifacts**: the "Upload iai JSON artifacts" step still succeeds (648-byte zip of five 0-byte `.json` files), masking the upstream failure.
3. **benchmark-action parse error**: the "Store + chart iai benchmark history" step then fails with `Output file for 'custom-(bigger|smaller)-is-better' must be JSON file containing an array of entries in BenchmarkResult format: Unexpected end of JSON input` — because `output-file-path` points at the 0-byte `iai_matmul.json`. This error mentions the custom tool, not the runner, which is misleading. If you see only this parse error, scroll **up** to the run-benches step for the real root cause.

## CI workflow (`.github/workflows/bench.yml`)

Triggers: push to `main`, `repository_dispatch` from smartcore (`event_type: smartcore-dev-push`, carrying the smartcore commit SHA in `client_payload.sha`), a nightly cron (`0 7 * * *`), and manual `workflow_dispatch`.

| Job | Runner | Tool | Role |
|---|---|---|---|
| `criterion` | ubuntu-latest | criterion (wall-clock) | Advisory. 200% alert threshold — noisy on shared runners, posts a comment but does not fail the job. |
| `iai` | ubuntu-latest | iai-callgrind (instruction count) | Deterministic gate. 120% alert threshold, `fail-on-alert: true`. Posts a `success`/`failure` status check back to the smartcore commit. |

### Results persistence

- Trend charts + JSON history → `gh-pages` branch, pushed by `benchmark-action@v1` (`auto-push: true`):
  - criterion: `gh-pages/dev/`
  - iai-callgrind: `gh-pages/iai-dev/`
- Per-version criterion snapshots → `benches-results/v<smartcore>/criterion.json` (idempotent; committed to `main` by the `criterion` job).

### Required secrets

| Secret | Used by | Purpose |
|---|---|---|
| `BENCHES_STATUS_PAT` | `Post status check back to smartcore` steps | Fine-grained PAT with `commit_status:write` on `smartcorelib/smartcore`. Lets this repo post the `smartcore-benches / iai-callgrind` status check that smartcore's branch protection can require. |

### Re-running failed jobs (IMPORTANT)

Re-running a failed job reuses the **workflow file pinned to the triggering SHA**, so re-running an old run whose SHA predates a fix will re-fail the same way. To validate a fix, trigger a **fresh** `workflow_dispatch` from the new tip:

```bash
gh workflow run bench.yml --repo smartcorelib/smartcore-benches --ref main
```

### gh-pages branch must exist (IMPORTANT)

`benchmark-action@v1` with `auto-push: true` + `gh-pages-branch: gh-pages` does a `git fetch` of that branch and fails hard if it doesn't exist:

```
fatal: couldn't find remote ref gh-pages
Error: The process '/usr/bin/git' failed with exit code 128
```

If `gh-pages` is missing (e.g. on a fresh repo), create an orphan branch with an empty root commit and push it:

```bash
git checkout --orphan gh-pages
git rm -rf --cached .
git clean -fdx
git config user.name  "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git commit --allow-empty -m "gh-pages: initial orphan branch for benchmark-action history [skip ci]"
git push origin gh-pages
git checkout main
```

### Dispatch contract (smartcore → benches)

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

smartcore is resolved from crates.io on every run, so the recorded history tracks the published release; the SHA is only for the status-check step, not for checkout.

## Benchmark inventory

### Hot-path benches (criterion)

| Bench | Hot path | Sizes |
|---|---|---|
| `matmul` | `Array2::matmul` | 64², 256², 1024² |
| `ab` | `Array2::ab` — 4 transpose combos | 256² |
| `svd` | `SVDDecomposable::svd` — DenseMatrix + ndarray | 128², 256×64 |
| `cover_tree` | `CoverTree::new`/`find` vs `LinearKNNSearch` | 1k/10k/100k × 10/100 |
| `iterator_mut` | `DenseMatrix::iterator_mut` + `MutView::iterator_mut` | 1024², 4096² |

### Algorithm benches (criterion, legacy)

`distance`, `fastpair`, `linear`, `naive_bayes`, `svc`.

### Deterministic gate (iai-callgrind, Linux-only)

`iai_matmul`, `iai_ab`, `iai_svd`, `iai_cover_tree`, `iai_iterator_mut` mirror the hot-path benches under Valgrind's instruction counter.

## Python tooling

When a script (e.g. `scripts/iai_to_benchmark_action.py`) needs Python deps, use [`uv`](https://docs.astral.sh/uv/) with a repo-local `.venv` — do **not** pollute the system interpreter or expect a global install:

```bash
uv venv                              # creates ./.venv
uv pip install <pkg>                 # installs into ./.venv
uv run python scripts/iai_to_benchmark_action.py --self-test /tmp/sc-iai-test
```

The existing scripts are stdlib-only (no `uv pip install` needed), but any new script with third-party imports should declare its deps in a `pyproject.toml` or `requirements.txt` and be run via `uv run`.

`.venv/` is already covered by the spirit of the `.gitignore` patterns — keep it local, do not commit it.

## Code conventions

- Follow smartcore's [AGENTS.md](https://github.com/smartcorelib/smartcore/blob/development/AGENTS.md): edition 2024, no `unsafe`, `#[expect]` over `#[allow]`, bind intermediates before iterating in edition-2024 tail positions.
- Bench code targets the same dep versions as smartcore's `Cargo.toml` so cargo unifies a single copy of each crate across the graph.
- No comments in generated bench code unless asked.
