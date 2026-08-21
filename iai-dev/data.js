window.BENCHMARK_DATA = {
  "lastUpdate": 1787298726760,
  "repoUrl": "https://github.com/smartcorelib/smartcore-benches",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "41898282+github-actions[bot]@users.noreply.github.com",
            "name": "github-actions[bot]",
            "username": "github-actions[bot]"
          },
          "committer": {
            "email": "41898282+github-actions[bot]@users.noreply.github.com",
            "name": "github-actions[bot]",
            "username": "github-actions[bot]"
          },
          "distinct": true,
          "id": "67929d1521b5937433d6d280bb8aeafe94a56a74",
          "message": "fix(iai): read total.summary.Ir.metrics, not the per-segment events\n\nThe previous adapter looked at callgrind_run.events.Ir.metrics, but the\nactual summary.v3 schema puts aggregated values under\ncallgrind_run.total.summary.Ir.metrics (with segments[].events as a\nper-segment breakdown). The adapter found 0 entries against the real\nCI NDJSON (12 bench lines parsed, 0 extracted) and the workflow failed\nwith 'no iai benchmark entries extracted'.\n\nLocate the Ir metric via total.summary first (the headline number the\nterminal shows), falling back to segments[0].events if total is absent.\nValidated against the real CI artifact from run 31323708625 — all 12\nbench entries extract cleanly. Update AGENTS.md schema note to the real\npath.",
          "timestamp": "2026-08-09T17:33:54+01:00",
          "tree_id": "90722cc06f6e3fb804a389a41e6ec9e076f2c5ab",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/67929d1521b5937433d6d280bb8aeafe94a56a74"
        },
        "date": 1786293440937,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "41898282+github-actions[bot]@users.noreply.github.com",
            "name": "github-actions[bot]",
            "username": "github-actions[bot]"
          },
          "committer": {
            "email": "41898282+github-actions[bot]@users.noreply.github.com",
            "name": "github-actions[bot]",
            "username": "github-actions[bot]"
          },
          "distinct": true,
          "id": "25d362a133f9b8ddec6d16ed4c4f3b6d0f38fc47",
          "message": "docs(readme): add emojis to live chart links",
          "timestamp": "2026-08-09T17:41:58+01:00",
          "tree_id": "9c2f820576befd738e5f5c13362b815d27e0ada3",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/25d362a133f9b8ddec6d16ed4c4f3b6d0f38fc47"
        },
        "date": 1786293862291,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "25d362a133f9b8ddec6d16ed4c4f3b6d0f38fc47",
          "message": "docs(readme): add emojis to live chart links",
          "timestamp": "2026-08-09T16:41:58Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/25d362a133f9b8ddec6d16ed4c4f3b6d0f38fc47"
        },
        "date": 1786351655998,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "25d362a133f9b8ddec6d16ed4c4f3b6d0f38fc47",
          "message": "docs(readme): add emojis to live chart links",
          "timestamp": "2026-08-09T16:41:58Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/25d362a133f9b8ddec6d16ed4c4f3b6d0f38fc47"
        },
        "date": 1786380331499,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786436100933,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786523229381,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786609764447,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786695867370,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786779561109,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786865981668,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786953550203,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1787039256568,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1787125725668,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1787212238773,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20329144,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76066223,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1787298725789,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "iai_matmul::matmul::bench_matmul_64",
            "value": 15886971,
            "unit": "Instructions"
          },
          {
            "name": "iai_matmul::matmul::bench_matmul_256",
            "value": 807278865,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_false",
            "value": 790369313,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_false_true",
            "value": 1059198507,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_false",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_ab::ab::bench_ab_true_true",
            "value": 1042421292,
            "unit": "Instructions"
          },
          {
            "name": "iai_svd::svd::bench_svd_square_128",
            "value": 20270901,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_build_10k_x_10",
            "value": 66929673,
            "unit": "Instructions"
          },
          {
            "name": "iai_cover_tree::cover_tree::bench_cover_tree_find_10k_x_10",
            "value": 76126330,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_fast_path",
            "value": 599936426,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_dense_iterator_mut_cross_axis",
            "value": 860315552,
            "unit": "Instructions"
          },
          {
            "name": "iai_iterator_mut::iterator_mut::bench_mutview_iterator_mut",
            "value": 599936825,
            "unit": "Instructions"
          }
        ]
      }
    ]
  }
}