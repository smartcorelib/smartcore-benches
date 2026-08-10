window.BENCHMARK_DATA = {
  "lastUpdate": 1786352585304,
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
        "date": 1786294815527,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 45719438,
            "range": "± 227180",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 59496016,
            "range": "± 356115",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 66987659,
            "range": "± 233376",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 62046508,
            "range": "± 509744",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 115966,
            "range": "± 2633",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 367837,
            "range": "± 8747",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1133491,
            "range": "± 25656",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 3682407,
            "range": "± 71977",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 13172227,
            "range": "± 378938",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 42160277,
            "range": "± 418554",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 65225,
            "range": "± 235",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 308418,
            "range": "± 704",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 666267,
            "range": "± 3315",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 3272747,
            "range": "± 135423",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 7051294,
            "range": "± 204004",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 39304935,
            "range": "± 457318",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 52784,
            "range": "± 391",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 295348,
            "range": "± 1908",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 524630,
            "range": "± 4019",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 2957074,
            "range": "± 12920",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 5274367,
            "range": "± 26330",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 30475158,
            "range": "± 144696",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 31,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 52883,
            "range": "± 313",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 39462,
            "range": "± 148",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 297885,
            "range": "± 211",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 219161,
            "range": "± 144",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2717680,
            "range": "± 1500",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 1881256,
            "range": "± 1517",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 15125,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 11684,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 91732,
            "range": "± 132",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 70126,
            "range": "± 96",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 850202,
            "range": "± 403",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 643536,
            "range": "± 734",
            "unit": "ns/iter"
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
        "date": 1786352584589,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 70515631,
            "range": "± 540113",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 71910383,
            "range": "± 188929",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 73037482,
            "range": "± 193391",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 72166634,
            "range": "± 805864",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 147356,
            "range": "± 3658",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 525799,
            "range": "± 8243",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1444845,
            "range": "± 14829",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 5215157,
            "range": "± 8741",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14507332,
            "range": "± 53887",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 52533172,
            "range": "± 176699",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 82759,
            "range": "± 177",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 469764,
            "range": "± 558",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 832564,
            "range": "± 2989",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4862773,
            "range": "± 17358",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8243200,
            "range": "± 16288",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 63805085,
            "range": "± 1292567",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 68985,
            "range": "± 100",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 449282,
            "range": "± 571",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 687061,
            "range": "± 1468",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4492980,
            "range": "± 6753",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6862619,
            "range": "± 15831",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 44932733,
            "range": "± 38417",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 40,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 59865,
            "range": "± 604",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 49736,
            "range": "± 1157",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 331993,
            "range": "± 778",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 251459,
            "range": "± 494",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 3117104,
            "range": "± 1501",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2294357,
            "range": "± 2180",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16971,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 14244,
            "range": "± 160",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 103883,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 83916,
            "range": "± 241",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 982815,
            "range": "± 1404",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 785644,
            "range": "± 1838",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}