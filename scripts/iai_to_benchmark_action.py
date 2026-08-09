#!/usr/bin/env python3
"""Convert iai-callgrind NDJSON into a benchmark-action customSmallerIsBetter array.

``benchmark-action/github-action-benchmark`` has no dedicated ``iai-callgrind`` tool
key, so the ``iai`` CI job in ``.github/workflows/bench.yml`` maps the output
through ``tool: customSmallerIsBetter``. That tool expects a JSON **array** of
entries shaped ``{"name": .., "unit": .., "value": ..}`` (see the action README
§customBiggerIsBetter/customSmallerIsBetter).

``iai-callgrind`` 0.14's ``--output-format=json`` instead prints one
``BenchmarkSummary`` object per benchmark line (NDJSON: one JSON document per
line). Each document aggregates per-segment metrics under
``callgrind_summary.callgrind_run`` in two places:

  - ``total.summary.<EventKind>.metrics`` — the aggregated total across all
    segments (threads/subprocesses/parts). This is what the terminal output
    shows as the headline number for each bench.
  - ``segments[].events.<EventKind>.metrics`` — per-segment breakdown.

We read the **total** (``summary.v3`` schema: ``CallgrindRun.total`` is a
``MetricsSummary`` keyed by ``EventKind`` string), since that is the
machine-independent instruction count the gate should track. ``EventKind="Ir"``
is "instructions retired" (the default callgrind event).

``metrics`` is an `EitherOrBoth_for_uint64`:
  - ``{"Left": n}``  — new-only (first run, no prior baseline on disk)
  - ``{"Right": o}`` — old-only (degenerate; new run produced nothing)
  - ``{"Both": [n, o]}`` — new ``n`` and old ``o``; we project ``n`` =
    ``Both[0]`` for the current value, ``o`` = ``Both[1]`` being the comparison
    baseline the action keeps in its own history.

Inputs (positional, in order):
  ndjson_path  File of newline-separated BenchmarkSummary JSON objects as
               emitted by ``cargo bench -- --output-format=json``. Multiple
               bench crates may be concatenated — each line is independent.
  out_path     Destination JSON array file (e.g. bench-output/iai.json).

Exits non-zero with a ``::error::`` annotation if no entries were produced, so
the workflow step fails loudly rather than feeding the action an empty array
(which the action rejects with a confusing "Unexpected end of JSON input"
parse error).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _metric_value(metrics: dict[str, Any]) -> int:
    both = metrics.get("Both")
    if isinstance(both, list) and both:
        return int(both[0])
    left = metrics.get("Left")
    if left is not None:
        return int(left)
    right = metrics.get("Right")
    if right is not None:
        return int(right)
    raise ValueError(f"unexpected EitherOrBoth shape: {metrics!r}")


def _ir_metrics(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Locate the ``Ir`` MetricsDiff under callgrind_run.

    Prefer ``total.summary.Ir`` (the aggregated headline); fall back to the
    first segment's ``events.Ir`` if a future/alternate schema omits ``total``.
    Returns the inner ``metrics`` dict (the EitherOrBoth) or ``None``.
    """
    run = summary.get("callgrind_summary", {}).get("callgrind_run", {})
    total = run.get("total") or {}
    summary_map = total.get("summary") if isinstance(total, dict) else None
    if isinstance(summary_map, dict) and "Ir" in summary_map:
        return summary_map["Ir"].get("metrics")
    segments = run.get("segments")
    if isinstance(segments, list) and segments:
        events = segments[0].get("events") or {}
        if "Ir" in events:
            return events["Ir"].get("metrics")
    return None


def _entry_from_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    name = summary.get("module_path") or summary.get("function_name")
    if not name:
        return None
    metrics = _ir_metrics(summary)
    if not isinstance(metrics, dict):
        return None
    try:
        value = _metric_value(metrics)
    except (KeyError, TypeError, ValueError):
        return None
    return {"name": name, "unit": "Instructions", "value": value}


def convert(ndjson_path: Path, out_path: Path) -> int:
    entries: list[dict[str, Any]] = []
    total_lines = 0
    for raw in ndjson_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        total_lines += 1
        summary = json.loads(stripped)
        entry = _entry_from_summary(summary)
        if entry is not None:
            entries.append(entry)

    if not entries:
        print(
            f"::error::no iai benchmark entries extracted from {ndjson_path} "
            f"(parsed {total_lines} non-empty line(s))",
            file=sys.stderr,
        )
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(
        f"::notice::wrote {len(entries)} iai entries "
        f"(from {total_lines} summary line(s)) to {out_path}"
    )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--self-test":
        return _self_test(Path(argv[2]))
    if len(argv) != 3:
        print(
            f"usage: {argv[0]} <ndjson_path> <out_path>",
            file=sys.stderr,
        )
        return 2
    return convert(Path(argv[1]), Path(argv[2]))


def _self_test(tmp: Path) -> int:
    """Synthetic NDJSON covering total/segment paths, Left/Both/Right, skips."""
    tmp.mkdir(parents=True, exist_ok=True)
    ndjson = tmp / "in.ndjson"
    # real schema: callgrind_run.total.summary.Ir.metrics (and segments fallback)
    ndjson.write_text(
        """
{"module_path":"iai_matmul::matmul::bench_matmul_64","function_name":"bench_matmul_64","callgrind_summary":{"callgrind_run":{"total":{"summary":{"Ir":{"metrics":{"Left":352135}}}},"segments":[{"events":{"Ir":{"metrics":{"Left":352135}}}}]}}}
{"module_path":"iai_matmul::matmul::bench_matmul_256","function_name":"bench_matmul_256","callgrind_summary":{"callgrind_run":{"total":{"summary":{"Ir":{"metrics":{"Both":[1000000,1010000]}}}},"segments":[{"events":{"Ir":{"metrics":{"Left":1000000}}}}]}}}
{"function_name":"iai_ab::ab::bench_ab_tn","callgrind_summary":{"callgrind_run":{"total":{"summary":{"Ir":{"metrics":{"Left":44000}}}}}}}
{"module_path":"iai_svd::svd::bench_svd_128","callgrind_summary":{"callgrind_run":{"total":{"summary":{"L1hits":{"metrics":{"Left":1234}}}}}}}
{"module_path":"iai_cover_tree::ct::bench_find","callgrind_summary":{"callgrind_run":{"total":{"summary":{"Ir":{"metrics":{"Right":999}}}}}}}
{"module_path":"iai_x::x::bench_only_segment","callgrind_summary":{"callgrind_run":{"segments":[{"events":{"Ir":{"metrics":{"Left":7}}}}]}}}
""",
        encoding="utf-8",
    )
    out = tmp / "out.json"
    rc = convert(ndjson, out)
    if rc != 0:
        print("self-test FAILED: convert returned non-zero", file=sys.stderr)
        return 1
    entries = json.loads(out.read_text(encoding="utf-8"))
    expected = [
        {"name": "iai_matmul::matmul::bench_matmul_64", "unit": "Instructions", "value": 352135},
        {"name": "iai_matmul::matmul::bench_matmul_256", "unit": "Instructions", "value": 1000000},
        {"name": "iai_ab::ab::bench_ab_tn", "unit": "Instructions", "value": 44000},
        {"name": "iai_cover_tree::ct::bench_find", "unit": "Instructions", "value": 999},
        {"name": "iai_x::x::bench_only_segment", "unit": "Instructions", "value": 7},
    ]
    if entries != expected:
        print(f"self-test FAILED:\n  got:      {entries}\n  expected: {expected}", file=sys.stderr)
        return 1
    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))