# benches-results/

Per-version criterion benchmark outputs, persisted by the `criterion` job in
[`.github/workflows/bench.yml`](../.github/workflows/bench.yml).

## Layout

```
benches-results/
└── v<smartcore-version>/
    └── criterion.json      # bencher-format output from `cargo bench -- --output-format bencher`
```

e.g. `benches-results/v0.6.4/criterion.json`.

## Storage contract

- A run resolves the smartcore version cargo picked from crates.io (read from
  `Cargo.lock` after the build step) and stores its `criterion.json` under
  `benches-results/v<version>/`.
- **Idempotent**: if `criterion.json` for that version already exists in the
  repo (checked against the latest `main` after a `git fetch`), the storage
  step is a no-op. A given smartcore version is recorded at most once.
- Storage commits carry `[skip ci]` so they don't re-trigger `lint.yml` or
  `bench.yml`; pushes use `GITHUB_TOKEN` (which also doesn't trigger workflows).
- A storage push failure (e.g. a non-ff rejection from a concurrent run) does
  not fail the `criterion` job — the next run will re-attempt, and the
  idempotent guard prevents duplicates once one succeeds.

## Where the live trend charts live

The `benchmark-action/github-action-benchmark` step also persists a rolling
history (every run, not per-version) to the `gh-pages` branch under `dev/`.
This directory is the **per-release snapshot**; `gh-pages/dev/` is the
**across-releases trend**.