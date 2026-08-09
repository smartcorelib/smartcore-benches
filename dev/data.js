window.BENCHMARK_DATA = {
  "lastUpdate": 1786292718410,
  "repoUrl": "https://github.com/smartcorelib/smartcore-benches",
  "entries": {
    "Benchmark": [
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
          "id": "a381724427c3a056cc18948344052ab35da99c44",
          "message": "ci(iai): build before runner install so Cargo.lock exists\n\nCargo.lock is gitignored (not committed), so the grep for\niai-callgrind-runner version failed. Move the build-no-run step before\nthe runner install: it regenerates Cargo.lock and the runner is only\nneeded at run time, not to compile the bench crate.",
          "timestamp": "2026-08-09T16:24:30Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/a381724427c3a056cc18948344052ab35da99c44"
        },
        "date": 1786292717363,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63349379,
            "range": "± 456574",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 70705147,
            "range": "± 335626",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 72843101,
            "range": "± 648671",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 67322022,
            "range": "± 440975",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 142887,
            "range": "± 3773",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 479768,
            "range": "± 7214",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1366199,
            "range": "± 5465",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4758902,
            "range": "± 12882",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 13683764,
            "range": "± 116379",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 47921097,
            "range": "± 210841",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 79205,
            "range": "± 321",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 428550,
            "range": "± 522",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 786385,
            "range": "± 9515",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4294122,
            "range": "± 28045",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 7853842,
            "range": "± 28133",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 53445402,
            "range": "± 187247",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 67610,
            "range": "± 97",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 405797,
            "range": "± 709",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 673315,
            "range": "± 4666",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4050920,
            "range": "± 7110",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6731437,
            "range": "± 30649",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40531236,
            "range": "± 82267",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 37,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 57999,
            "range": "± 1780",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 46189,
            "range": "± 204",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 329154,
            "range": "± 5254",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 245841,
            "range": "± 951",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2961466,
            "range": "± 70734",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2205482,
            "range": "± 6886",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16844,
            "range": "± 391",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13553,
            "range": "± 60",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 100966,
            "range": "± 1145",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 82170,
            "range": "± 69",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 940273,
            "range": "± 13105",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 752350,
            "range": "± 1438",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}