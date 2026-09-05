window.BENCHMARK_DATA = {
  "lastUpdate": 1788606948269,
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
          "id": "7ddb398df7f6299e4fc1155be68642f6de355f90",
          "message": "add harness and agents doc",
          "timestamp": "2026-08-21T13:49:12+01:00",
          "tree_id": "c352bdac10823ac3895315b1a0b47ccd06915a33",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/7ddb398df7f6299e4fc1155be68642f6de355f90"
        },
        "date": 1787316715896,
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
          "id": "7ddb398df7f6299e4fc1155be68642f6de355f90",
          "message": "add harness and agents doc",
          "timestamp": "2026-08-21T12:49:12Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/7ddb398df7f6299e4fc1155be68642f6de355f90"
        },
        "date": 1787316878414,
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
          "id": "4a2622b8337bd4807aaebd9504148a48bd89d06b",
          "message": "fix(criterion): run only criterion benches so the full suite is recorded\n\nA bare `cargo bench -- --output-format bencher` also executes the iai_*\nbinaries, whose Linux main execs iai-callgrind-runner (not installed in\nthe criterion job). cargo aborts at iai_ab, so every bench sorted after\nit (iterator_mut, linear, matmul, naive_bayes, svc, svd) never ran and\nthe stored snapshots were partial. Pass the 10 criterion benches\nexplicitly and set pipefail so a bench failure fails the step.\n\nDrop the partial v0.6.7 snapshot so the next run re-records it.",
          "timestamp": "2026-08-21T14:14:41+01:00",
          "tree_id": "085a249a767d48a27be97fe7df0b2095fa5fedf0",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/4a2622b8337bd4807aaebd9504148a48bd89d06b"
        },
        "date": 1787318230259,
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
          "id": "543598f895910d6dc6bac938486defd3e60cff31",
          "message": "fix(linear): use a valid grid so LinearRegression::fit does not panic\n\nThe legacy grid (n_samples 10-100, n_features up to 1000) predates the\nintercept-augmentation rework: fit now solves an n x (p+1) system, so\nany case with p + 1 > n_samples is underdetermined. The SVD solver then\nindexes past b in SVD::solve (index out of bounds) and QR panics with\n'Matrix is rank deficient'. Both should be Err(Failed); filed upstream\nas smartcorelib/smartcore#435.\n\nReplace the grid with n_samples > n_features + 1 pairs\n(64x16, 256x64, 1024x256) and keep the default SVD solver. The bench\nnever recorded data in CI before (blocked by the iai_ab abort), so no\nhistory continuity is lost.",
          "timestamp": "2026-08-21T15:19:34+01:00",
          "tree_id": "5cb9e4b6b667c611193392fbf582708ef6474b36",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/543598f895910d6dc6bac938486defd3e60cff31"
        },
        "date": 1787322130454,
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
          "id": "41c23ae4e7f6bed4b25042da0a93242e72df99c6",
          "message": "build: criterion 0.8.2, MSRV 1.86, bounded bench runtime; bench smartcore 0.6.10\n\n- criterion 0.5 -> 0.8.2 (0.8 needs Rust 1.86; rust-version raised\n  accordingly - CI runs stable, smartcore itself stays MSRV 1.85).\n  Replace deprecated criterion::black_box with std::hint::black_box.\n- Cap wall-clock cost via CLI (criterion 0.8 removed Criterion.toml):\n  --warm-up-time 0.5 --measurement-time 2 --sample-size 20 keeps the\n  full suite inside ~15 min on shared runners instead of hours.\n- Trim MultiClassSVC::fit grid to 100/500/1000 x 10/100: RBF SVM fit\n  scales ~quadratically; CI measured 7.5 s/fit at 1000x100 and the old\n  10000-sample rows needed minutes per fit (run cancelled after >1 h\n  stuck in svc).\n- Lockfile now resolves smartcore 0.6.10 for the next recording run.",
          "timestamp": "2026-08-21T16:55:03+01:00",
          "tree_id": "89819297b0ca36d44621d48fc88842e78986f2ff",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/41c23ae4e7f6bed4b25042da0a93242e72df99c6"
        },
        "date": 1787327868623,
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
          "id": "a0b2669a95ea9bc02f12c58c06019c5adbb59652",
          "message": "bench(criterion): record v0.6.10 results [skip ci]",
          "timestamp": "2026-08-21T16:20:36Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/a0b2669a95ea9bc02f12c58c06019c5adbb59652"
        },
        "date": 1787384399784,
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
          "id": "a0b2669a95ea9bc02f12c58c06019c5adbb59652",
          "message": "bench(criterion): record v0.6.10 results [skip ci]",
          "timestamp": "2026-08-21T16:20:36Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/a0b2669a95ea9bc02f12c58c06019c5adbb59652"
        },
        "date": 1787470890026,
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
          "id": "a0b2669a95ea9bc02f12c58c06019c5adbb59652",
          "message": "bench(criterion): record v0.6.10 results [skip ci]",
          "timestamp": "2026-08-21T16:20:36Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/a0b2669a95ea9bc02f12c58c06019c5adbb59652"
        },
        "date": 1787558598359,
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
          "id": "c9e5a99a1813d61ff3026797cc708737a2d97ea8",
          "message": "bench(criterion): record v0.6.11 results [skip ci]",
          "timestamp": "2026-08-24T08:26:26Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/c9e5a99a1813d61ff3026797cc708737a2d97ea8"
        },
        "date": 1787591749032,
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
          "id": "d28e8cea28566ca0127bf981e8d3fd75b9130478",
          "message": "bench(criterion): record v0.6.12 results [skip ci]",
          "timestamp": "2026-08-24T17:38:48Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/d28e8cea28566ca0127bf981e8d3fd75b9130478"
        },
        "date": 1787644372619,
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
          "id": "d28e8cea28566ca0127bf981e8d3fd75b9130478",
          "message": "bench(criterion): record v0.6.12 results [skip ci]",
          "timestamp": "2026-08-24T17:38:48Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/d28e8cea28566ca0127bf981e8d3fd75b9130478"
        },
        "date": 1787653824299,
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
          "id": "1b2eb0fde4141dcade56ace88e9726713cf0de22",
          "message": "bench(criterion): record v0.6.13 results [skip ci]",
          "timestamp": "2026-08-25T10:54:07Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/1b2eb0fde4141dcade56ace88e9726713cf0de22"
        },
        "date": 1787730864936,
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
          "id": "1b2eb0fde4141dcade56ace88e9726713cf0de22",
          "message": "bench(criterion): record v0.6.13 results [skip ci]",
          "timestamp": "2026-08-25T10:54:07Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/1b2eb0fde4141dcade56ace88e9726713cf0de22"
        },
        "date": 1787746585479,
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
          "id": "147686f018600f3be5f4f6c46b824e183fd6a89d",
          "message": "bench(criterion): record v0.6.14 results [skip ci]",
          "timestamp": "2026-08-26T12:39:31Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/147686f018600f3be5f4f6c46b824e183fd6a89d"
        },
        "date": 1787854504686,
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
          "id": "147686f018600f3be5f4f6c46b824e183fd6a89d",
          "message": "bench(criterion): record v0.6.14 results [skip ci]",
          "timestamp": "2026-08-26T12:39:31Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/147686f018600f3be5f4f6c46b824e183fd6a89d"
        },
        "date": 1787945098585,
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
          "id": "147686f018600f3be5f4f6c46b824e183fd6a89d",
          "message": "bench(criterion): record v0.6.14 results [skip ci]",
          "timestamp": "2026-08-26T12:39:31Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/147686f018600f3be5f4f6c46b824e183fd6a89d"
        },
        "date": 1788008968626,
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
          "id": "147686f018600f3be5f4f6c46b824e183fd6a89d",
          "message": "bench(criterion): record v0.6.14 results [skip ci]",
          "timestamp": "2026-08-26T12:39:31Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/147686f018600f3be5f4f6c46b824e183fd6a89d"
        },
        "date": 1788094045767,
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
          "id": "147686f018600f3be5f4f6c46b824e183fd6a89d",
          "message": "bench(criterion): record v0.6.14 results [skip ci]",
          "timestamp": "2026-08-26T12:39:31Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/147686f018600f3be5f4f6c46b824e183fd6a89d"
        },
        "date": 1788188580111,
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
          "id": "147686f018600f3be5f4f6c46b824e183fd6a89d",
          "message": "bench(criterion): record v0.6.14 results [skip ci]",
          "timestamp": "2026-08-26T12:39:31Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/147686f018600f3be5f4f6c46b824e183fd6a89d"
        },
        "date": 1788265990717,
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
          "id": "147686f018600f3be5f4f6c46b824e183fd6a89d",
          "message": "bench(criterion): record v0.6.14 results [skip ci]",
          "timestamp": "2026-08-26T12:39:31Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/147686f018600f3be5f4f6c46b824e183fd6a89d"
        },
        "date": 1788350784923,
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
          "id": "147686f018600f3be5f4f6c46b824e183fd6a89d",
          "message": "bench(criterion): record v0.6.14 results [skip ci]",
          "timestamp": "2026-08-26T12:39:31Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/147686f018600f3be5f4f6c46b824e183fd6a89d"
        },
        "date": 1788437093905,
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
          "id": "147686f018600f3be5f4f6c46b824e183fd6a89d",
          "message": "bench(criterion): record v0.6.14 results [skip ci]",
          "timestamp": "2026-08-26T12:39:31Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/147686f018600f3be5f4f6c46b824e183fd6a89d"
        },
        "date": 1788523666663,
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
          "id": "147686f018600f3be5f4f6c46b824e183fd6a89d",
          "message": "bench(criterion): record v0.6.14 results [skip ci]",
          "timestamp": "2026-08-26T12:39:31Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/147686f018600f3be5f4f6c46b824e183fd6a89d"
        },
        "date": 1788606947496,
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