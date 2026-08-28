window.BENCHMARK_DATA = {
  "lastUpdate": 1787946769923,
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
        "date": 1786381270766,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63429326,
            "range": "± 2154069",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 84505549,
            "range": "± 6720662",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 74539129,
            "range": "± 1415161",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 67612829,
            "range": "± 610832",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 144086,
            "range": "± 3840",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 481617,
            "range": "± 8686",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1395184,
            "range": "± 7131",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4801256,
            "range": "± 78124",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14222703,
            "range": "± 114463",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 48136718,
            "range": "± 201455",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 79401,
            "range": "± 719",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 430518,
            "range": "± 7659",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 794176,
            "range": "± 46121",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4437415,
            "range": "± 234767",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 7982308,
            "range": "± 288565",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 54976821,
            "range": "± 440803",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 68423,
            "range": "± 1295",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 406520,
            "range": "± 669",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 683192,
            "range": "± 1834",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4058113,
            "range": "± 8964",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6840255,
            "range": "± 203228",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40560044,
            "range": "± 73624",
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
            "value": 57089,
            "range": "± 1191",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 45685,
            "range": "± 1176",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 315732,
            "range": "± 4854",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 242995,
            "range": "± 10030",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2918502,
            "range": "± 50478",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2260786,
            "range": "± 122143",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16265,
            "range": "± 132",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 14006,
            "range": "± 86",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 97634,
            "range": "± 533",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 79861,
            "range": "± 244",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 912929,
            "range": "± 7126",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 758120,
            "range": "± 26460",
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
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786437022186,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63237319,
            "range": "± 80568",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 70789636,
            "range": "± 255103",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 73674997,
            "range": "± 4333942",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 67467570,
            "range": "± 956250",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 144857,
            "range": "± 4246",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 480588,
            "range": "± 7756",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1412994,
            "range": "± 13362",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4783000,
            "range": "± 12402",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14137433,
            "range": "± 123409",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 48245824,
            "range": "± 193602",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 79418,
            "range": "± 1592",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 430477,
            "range": "± 1025",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 788873,
            "range": "± 1657",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4285547,
            "range": "± 12700",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 7795125,
            "range": "± 79053",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 53315246,
            "range": "± 1115423",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 68466,
            "range": "± 261",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 406718,
            "range": "± 1623",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 682216,
            "range": "± 1649",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4054333,
            "range": "± 10009",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6802387,
            "range": "± 173469",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40577553,
            "range": "± 74870",
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
            "value": 58081,
            "range": "± 647",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 46849,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 317338,
            "range": "± 8323",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 246956,
            "range": "± 3482",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2958243,
            "range": "± 88777",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2210502,
            "range": "± 11187",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16339,
            "range": "± 405",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 14211,
            "range": "± 256",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 98128,
            "range": "± 382",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 82329,
            "range": "± 1644",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 921014,
            "range": "± 19541",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 755835,
            "range": "± 1937",
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
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786524124829,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 40154471,
            "range": "± 1833075",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 51599531,
            "range": "± 2004215",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 51583425,
            "range": "± 2991867",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 48153826,
            "range": "± 2399407",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 101383,
            "range": "± 7127",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 307673,
            "range": "± 14287",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 957262,
            "range": "± 38944",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 3039501,
            "range": "± 128725",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 9684412,
            "range": "± 452702",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 32592190,
            "range": "± 2088946",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 54518,
            "range": "± 3620",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 263750,
            "range": "± 10146",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 553400,
            "range": "± 36004",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 2732068,
            "range": "± 92297",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 5686624,
            "range": "± 234749",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 30289218,
            "range": "± 2604261",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 46229,
            "range": "± 3200",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 247815,
            "range": "± 8312",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 439538,
            "range": "± 16659",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 2475638,
            "range": "± 96998",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 4415969,
            "range": "± 221661",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 25025289,
            "range": "± 1512395",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 31,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 49177,
            "range": "± 886",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 38203,
            "range": "± 505",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 287320,
            "range": "± 6241",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 208387,
            "range": "± 3881",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2671914,
            "range": "± 50011",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 1823825,
            "range": "± 33338",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 14038,
            "range": "± 288",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 11050,
            "range": "± 171",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 86840,
            "range": "± 1678",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 68443,
            "range": "± 1293",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 809739,
            "range": "± 25958",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 636393,
            "range": "± 6306",
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
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786610683501,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63280745,
            "range": "± 152438",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 70872439,
            "range": "± 210583",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 73819977,
            "range": "± 351536",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 67436570,
            "range": "± 1590481",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 166762,
            "range": "± 5418",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 508910,
            "range": "± 9482",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1580873,
            "range": "± 5951",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4900895,
            "range": "± 17053",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 16637421,
            "range": "± 119858",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 49562533,
            "range": "± 298888",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 84024,
            "range": "± 516",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 432155,
            "range": "± 642",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 870234,
            "range": "± 9356",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4307930,
            "range": "± 62217",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8410868,
            "range": "± 56861",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 56944921,
            "range": "± 420080",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 70558,
            "range": "± 301",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 408714,
            "range": "± 714",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 703956,
            "range": "± 3271",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4084312,
            "range": "± 7463",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 7042628,
            "range": "± 38783",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40856191,
            "range": "± 85794",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 44,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 58286,
            "range": "± 1410",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 48723,
            "range": "± 1194",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 319569,
            "range": "± 2067",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 245080,
            "range": "± 340",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2962152,
            "range": "± 16310",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2219037,
            "range": "± 33551",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16697,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 14076,
            "range": "± 215",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 100278,
            "range": "± 429",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 82536,
            "range": "± 3926",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 928013,
            "range": "± 3948",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 750720,
            "range": "± 37779",
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
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786696822629,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 64603481,
            "range": "± 500226",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 70773315,
            "range": "± 482448",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 78149401,
            "range": "± 373644",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 67393566,
            "range": "± 179706",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 144449,
            "range": "± 4167",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 481294,
            "range": "± 8503",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1414189,
            "range": "± 6029",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4752302,
            "range": "± 24108",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14096941,
            "range": "± 113882",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 47791193,
            "range": "± 160546",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 79780,
            "range": "± 367",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 429808,
            "range": "± 2268",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 800083,
            "range": "± 2420",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4359016,
            "range": "± 467029",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8063778,
            "range": "± 77619",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 55333041,
            "range": "± 2354654",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 67922,
            "range": "± 283",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 406096,
            "range": "± 1224",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 677954,
            "range": "± 15663",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4051742,
            "range": "± 77075",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6781165,
            "range": "± 419404",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40542470,
            "range": "± 389663",
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
            "value": 58033,
            "range": "± 1793",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 45886,
            "range": "± 340",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 316843,
            "range": "± 887",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 244532,
            "range": "± 506",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2981852,
            "range": "± 56919",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2205685,
            "range": "± 79109",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16466,
            "range": "± 262",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13482,
            "range": "± 309",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 98414,
            "range": "± 2815",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 81385,
            "range": "± 673",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 923024,
            "range": "± 16923",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 747838,
            "range": "± 755",
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
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786780475444,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63393088,
            "range": "± 638530",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 70852571,
            "range": "± 4220083",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 74406471,
            "range": "± 1028040",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 67521372,
            "range": "± 407320",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 145580,
            "range": "± 3752",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 482040,
            "range": "± 10366",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1422736,
            "range": "± 5276",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4861109,
            "range": "± 38158",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14828935,
            "range": "± 91494",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 48363645,
            "range": "± 132679",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 79484,
            "range": "± 960",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 431460,
            "range": "± 1127",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 792389,
            "range": "± 1735",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4725162,
            "range": "± 164186",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 7994828,
            "range": "± 47766",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 55559642,
            "range": "± 343005",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 67085,
            "range": "± 105",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 405847,
            "range": "± 568",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 669380,
            "range": "± 1942",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4049833,
            "range": "± 4234",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6724415,
            "range": "± 17943",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40505671,
            "range": "± 76083",
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
            "value": 57000,
            "range": "± 390",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 45774,
            "range": "± 150",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 315067,
            "range": "± 2910",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 243673,
            "range": "± 1563",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2916160,
            "range": "± 5017",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2208875,
            "range": "± 10062",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16286,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13286,
            "range": "± 73",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 97535,
            "range": "± 124",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 80932,
            "range": "± 90",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 912295,
            "range": "± 2586",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 752205,
            "range": "± 7255",
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
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786866920113,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 70552479,
            "range": "± 170978",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 71919752,
            "range": "± 411477",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 72707571,
            "range": "± 509767",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 72080292,
            "range": "± 254041",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 145972,
            "range": "± 3618",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 525131,
            "range": "± 8277",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1430727,
            "range": "± 5127",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 5233542,
            "range": "± 18076",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14381873,
            "range": "± 70171",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 52476619,
            "range": "± 2599837",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 87775,
            "range": "± 1516",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 477099,
            "range": "± 926",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 878927,
            "range": "± 1136",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4886621,
            "range": "± 33494",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8762250,
            "range": "± 26986",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 61141496,
            "range": "± 333506",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 69488,
            "range": "± 179",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 450349,
            "range": "± 380",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 693206,
            "range": "± 1708",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4497943,
            "range": "± 3354",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6915953,
            "range": "± 16578",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 44985304,
            "range": "± 40083",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 39,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 59241,
            "range": "± 787",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 49375,
            "range": "± 528",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 332733,
            "range": "± 1308",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 251191,
            "range": "± 988",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 3113740,
            "range": "± 4920",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2291965,
            "range": "± 4258",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 17200,
            "range": "± 167",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 14241,
            "range": "± 145",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 104151,
            "range": "± 283",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 83886,
            "range": "± 125",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 981333,
            "range": "± 1871",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 784459,
            "range": "± 1729",
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
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1786954475222,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63315334,
            "range": "± 97386",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 70895477,
            "range": "± 892316",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 73942242,
            "range": "± 1647268",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 67467331,
            "range": "± 792994",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 144500,
            "range": "± 3917",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 480742,
            "range": "± 11802",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1416769,
            "range": "± 6589",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4781227,
            "range": "± 183488",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14211069,
            "range": "± 71021",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 48017083,
            "range": "± 147973",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 79852,
            "range": "± 393",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 430810,
            "range": "± 625",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 794937,
            "range": "± 2963",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4562885,
            "range": "± 148825",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 7991163,
            "range": "± 83676",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 54720731,
            "range": "± 332920",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 67050,
            "range": "± 242",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 405904,
            "range": "± 8294",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 670211,
            "range": "± 25316",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4047483,
            "range": "± 6069",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6728080,
            "range": "± 10390",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40472244,
            "range": "± 645075",
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
            "value": 57266,
            "range": "± 437",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 46249,
            "range": "± 623",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 319531,
            "range": "± 3085",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 248635,
            "range": "± 15942",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2934098,
            "range": "± 7221",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2202170,
            "range": "± 114442",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16287,
            "range": "± 88",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 14010,
            "range": "± 271",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 98710,
            "range": "± 385",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 81381,
            "range": "± 213",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 922650,
            "range": "± 1819",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 748778,
            "range": "± 5910",
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
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1787040187744,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 73791452,
            "range": "± 5315854",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 70927518,
            "range": "± 281496",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 74047522,
            "range": "± 410558",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 67408994,
            "range": "± 142321",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 144231,
            "range": "± 3592",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 481062,
            "range": "± 9044",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1402505,
            "range": "± 7630",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4791292,
            "range": "± 68695",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14148527,
            "range": "± 53645",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 47891588,
            "range": "± 182727",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 85346,
            "range": "± 196",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 435306,
            "range": "± 1278",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 844545,
            "range": "± 4163",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4647286,
            "range": "± 158486",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8188867,
            "range": "± 41685",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 53188261,
            "range": "± 536152",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 71746,
            "range": "± 1055",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 410251,
            "range": "± 1542",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 714977,
            "range": "± 5476",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4094350,
            "range": "± 11056",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 7539682,
            "range": "± 24063",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40922750,
            "range": "± 65520",
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
            "value": 56945,
            "range": "± 337",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 45897,
            "range": "± 168",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 315039,
            "range": "± 7057",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 243691,
            "range": "± 467",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2933116,
            "range": "± 14530",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2208374,
            "range": "± 2587",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16229,
            "range": "± 255",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13508,
            "range": "± 130",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 97619,
            "range": "± 822",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 81915,
            "range": "± 115",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 912284,
            "range": "± 6489",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 752652,
            "range": "± 2490",
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
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1787126606123,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63539451,
            "range": "± 154364",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 70991962,
            "range": "± 462093",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 74420330,
            "range": "± 628762",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 67821523,
            "range": "± 1746957",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 145519,
            "range": "± 7864",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 483156,
            "range": "± 9471",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1419965,
            "range": "± 46148",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4817954,
            "range": "± 12891",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14285300,
            "range": "± 333202",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 48079363,
            "range": "± 138313",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 79529,
            "range": "± 223",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 432024,
            "range": "± 1118",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 802785,
            "range": "± 1812",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4696406,
            "range": "± 292756",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8074755,
            "range": "± 154304",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 55385696,
            "range": "± 332822",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 66993,
            "range": "± 166",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 406558,
            "range": "± 612",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 670490,
            "range": "± 3323",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4055121,
            "range": "± 20776",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6727467,
            "range": "± 21448",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40521228,
            "range": "± 721998",
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
            "value": 56961,
            "range": "± 863",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 46086,
            "range": "± 179",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 315223,
            "range": "± 2397",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 244395,
            "range": "± 16367",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2915486,
            "range": "± 36508",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2213056,
            "range": "± 51361",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16376,
            "range": "± 96",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 14003,
            "range": "± 201",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 97679,
            "range": "± 127",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 81129,
            "range": "± 2216",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 913272,
            "range": "± 7080",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 753085,
            "range": "± 1566",
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
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1787213169203,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 70692462,
            "range": "± 447804",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 72007038,
            "range": "± 1269880",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 73200763,
            "range": "± 208500",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 72223391,
            "range": "± 1220085",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 147099,
            "range": "± 4591",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 529757,
            "range": "± 7595",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1434469,
            "range": "± 4815",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 5224763,
            "range": "± 8910",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14297549,
            "range": "± 483663",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 52712125,
            "range": "± 1130506",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 83261,
            "range": "± 198",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 470978,
            "range": "± 1239",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 829899,
            "range": "± 2067",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4864694,
            "range": "± 20196",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8294096,
            "range": "± 232021",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 63036209,
            "range": "± 305090",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 69217,
            "range": "± 273",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 449487,
            "range": "± 1114",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 687422,
            "range": "± 1151",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4494076,
            "range": "± 5574",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6865949,
            "range": "± 12136",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 44937580,
            "range": "± 130526",
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
            "value": 59121,
            "range": "± 624",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 49549,
            "range": "± 544",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 334773,
            "range": "± 1300",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 251042,
            "range": "± 726",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 3103348,
            "range": "± 6784",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2293494,
            "range": "± 2738",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16923,
            "range": "± 132",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 14058,
            "range": "± 126",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 103680,
            "range": "± 280",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 84049,
            "range": "± 203",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 976031,
            "range": "± 987",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 785424,
            "range": "± 547",
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
          "id": "2d892ff94ae10f50b7a328ea9b1d033b0241b26a",
          "message": "bench(criterion): record v0.6.5 results [skip ci]",
          "timestamp": "2026-08-10T17:01:14Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/2d892ff94ae10f50b7a328ea9b1d033b0241b26a"
        },
        "date": 1787299660638,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 70530670,
            "range": "± 1992798",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 71805075,
            "range": "± 2179115",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 72677195,
            "range": "± 271275",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 72057743,
            "range": "± 448596",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 148041,
            "range": "± 4816",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 529552,
            "range": "± 10546",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1436375,
            "range": "± 3552",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 5238192,
            "range": "± 11512",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14306093,
            "range": "± 115051",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 52496580,
            "range": "± 324743",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 83184,
            "range": "± 1028",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 469598,
            "range": "± 14926",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 827491,
            "range": "± 2331",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4872935,
            "range": "± 118258",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8406895,
            "range": "± 169696",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 61278511,
            "range": "± 400202",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 69528,
            "range": "± 1759",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 450053,
            "range": "± 3533",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 694557,
            "range": "± 1567",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4494873,
            "range": "± 3066",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6919874,
            "range": "± 138967",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 44968219,
            "range": "± 429378",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 39,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 57246,
            "range": "± 189",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 46713,
            "range": "± 146",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 336354,
            "range": "± 669",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 233897,
            "range": "± 826",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 3040004,
            "range": "± 9013",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2110313,
            "range": "± 4823",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16634,
            "range": "± 59",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13619,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 102313,
            "range": "± 467",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 79041,
            "range": "± 263",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 964295,
            "range": "± 5881",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 740020,
            "range": "± 2955",
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
          "id": "7ddb398df7f6299e4fc1155be68642f6de355f90",
          "message": "add harness and agents doc",
          "timestamp": "2026-08-21T13:49:12+01:00",
          "tree_id": "c352bdac10823ac3895315b1a0b47ccd06915a33",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/7ddb398df7f6299e4fc1155be68642f6de355f90"
        },
        "date": 1787317656146,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63219900,
            "range": "± 145097",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 70193859,
            "range": "± 534170",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 75302505,
            "range": "± 455165",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 67459954,
            "range": "± 126959",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 145374,
            "range": "± 3913",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 481796,
            "range": "± 7599",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1394048,
            "range": "± 10587",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4749449,
            "range": "± 19864",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 13828567,
            "range": "± 95451",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 47576531,
            "range": "± 167248",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 79381,
            "range": "± 678",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 430224,
            "range": "± 1058",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 793724,
            "range": "± 3539",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4286605,
            "range": "± 43917",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 7841586,
            "range": "± 38880",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 53575304,
            "range": "± 100020",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 67721,
            "range": "± 521",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 406787,
            "range": "± 552",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 674592,
            "range": "± 1409",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4056701,
            "range": "± 6850",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6763159,
            "range": "± 10927",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40555564,
            "range": "± 73723",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 36,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 55581,
            "range": "± 493",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 45463,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 307000,
            "range": "± 2969",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 234051,
            "range": "± 328",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2840836,
            "range": "± 139048",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2109223,
            "range": "± 14261",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16069,
            "range": "± 137",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13324,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 96253,
            "range": "± 1106",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 79033,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 897399,
            "range": "± 6081",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 733147,
            "range": "± 3259",
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
          "id": "aabee1b34d3e6fc2ecc8a66ca4d656d0916eebca",
          "message": "bench(criterion): record v0.6.7 results [skip ci]",
          "timestamp": "2026-08-21T13:07:40Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/aabee1b34d3e6fc2ecc8a66ca4d656d0916eebca"
        },
        "date": 1787317816549,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63286494,
            "range": "± 1185256",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 70219012,
            "range": "± 133111",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 75268859,
            "range": "± 359071",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 67551998,
            "range": "± 1888666",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 146552,
            "range": "± 4491",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 481744,
            "range": "± 7849",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1415261,
            "range": "± 8397",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4778084,
            "range": "± 13828",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14454759,
            "range": "± 249312",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 48101303,
            "range": "± 1198133",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 79523,
            "range": "± 287",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 430238,
            "range": "± 1216",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 797442,
            "range": "± 11132",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4294145,
            "range": "± 12992",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 7905997,
            "range": "± 70040",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 54156264,
            "range": "± 279985",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 67668,
            "range": "± 421",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 406119,
            "range": "± 748",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 672884,
            "range": "± 3909",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4054923,
            "range": "± 4910",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6746060,
            "range": "± 12735",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40534887,
            "range": "± 50116",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 36,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 55781,
            "range": "± 238",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 45138,
            "range": "± 91",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 304889,
            "range": "± 758",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 234253,
            "range": "± 15588",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2801190,
            "range": "± 3292",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2089789,
            "range": "± 50484",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 15993,
            "range": "± 64",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13229,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 95289,
            "range": "± 228",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 78233,
            "range": "± 369",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 888728,
            "range": "± 4333",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 723054,
            "range": "± 12793",
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
          "id": "41c23ae4e7f6bed4b25042da0a93242e72df99c6",
          "message": "build: criterion 0.8.2, MSRV 1.86, bounded bench runtime; bench smartcore 0.6.10\n\n- criterion 0.5 -> 0.8.2 (0.8 needs Rust 1.86; rust-version raised\n  accordingly - CI runs stable, smartcore itself stays MSRV 1.85).\n  Replace deprecated criterion::black_box with std::hint::black_box.\n- Cap wall-clock cost via CLI (criterion 0.8 removed Criterion.toml):\n  --warm-up-time 0.5 --measurement-time 2 --sample-size 20 keeps the\n  full suite inside ~15 min on shared runners instead of hours.\n- Trim MultiClassSVC::fit grid to 100/500/1000 x 10/100: RBF SVM fit\n  scales ~quadratically; CI measured 7.5 s/fit at 1000x100 and the old\n  10000-sample rows needed minutes per fit (run cancelled after >1 h\n  stuck in svc).\n- Lockfile now resolves smartcore 0.6.10 for the next recording run.",
          "timestamp": "2026-08-21T16:55:03+01:00",
          "tree_id": "89819297b0ca36d44621d48fc88842e78986f2ff",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/41c23ae4e7f6bed4b25042da0a93242e72df99c6"
        },
        "date": 1787329234010,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63423331,
            "range": "± 61708",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 74153136,
            "range": "± 69248",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 76669018,
            "range": "± 97315",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 74383643,
            "range": "± 277326",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 144832,
            "range": "± 1123",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 492245,
            "range": "± 11200",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1425317,
            "range": "± 68247",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4795058,
            "range": "± 71532",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14724755,
            "range": "± 608131",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 50994774,
            "range": "± 527728",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 79950,
            "range": "± 190",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 430550,
            "range": "± 1086",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 804397,
            "range": "± 1391",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4416240,
            "range": "± 133649",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 7964740,
            "range": "± 47102",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 54224753,
            "range": "± 355843",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 67255,
            "range": "± 256",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 405940,
            "range": "± 449",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 671257,
            "range": "± 1917",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4055484,
            "range": "± 2285",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6703628,
            "range": "± 34577",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40576957,
            "range": "± 96814",
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
            "value": 57009,
            "range": "± 779",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 44534,
            "range": "± 243",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 308190,
            "range": "± 2674",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 221741,
            "range": "± 4957",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2819743,
            "range": "± 13866",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 1977412,
            "range": "± 9455",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16352,
            "range": "± 58",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13094,
            "range": "± 116",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 96654,
            "range": "± 959",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 76201,
            "range": "± 1034",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 901245,
            "range": "± 5993",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 701488,
            "range": "± 3751",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/1024",
            "value": 4676283,
            "range": "± 452796",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/1024",
            "value": 49263720,
            "range": "± 479610",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/1024",
            "value": 43142540,
            "range": "± 277100",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/1024",
            "value": 4067352,
            "range": "± 435997",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/4096",
            "value": 75965050,
            "range": "± 397085",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/4096",
            "value": 980298713,
            "range": "± 2718314",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/4096",
            "value": 972161302,
            "range": "± 2243225",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/4096",
            "value": 72571510,
            "range": "± 3477595",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/1024",
            "value": 3614283,
            "range": "± 227831",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/1024",
            "value": 49959372,
            "range": "± 406741",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/4096",
            "value": 70909369,
            "range": "± 282143",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/4096",
            "value": 979787960,
            "range": "± 4055690",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 64, n_features: 16",
            "value": 37281,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 256, n_features: 64",
            "value": 714232,
            "range": "± 1709",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 1024, n_features: 256",
            "value": 16451829,
            "range": "± 105516",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/64",
            "value": 1010604,
            "range": "± 20199",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/256",
            "value": 63452090,
            "range": "± 67709",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/1024",
            "value": 4714764112,
            "range": "± 108670454",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 10",
            "value": 25848,
            "range": "± 258",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 100",
            "value": 142344,
            "range": "± 2426",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 1000",
            "value": 1291865,
            "range": "± 4382",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 10",
            "value": 335109,
            "range": "± 1328",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 100",
            "value": 1501866,
            "range": "± 25491",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 1000",
            "value": 18275305,
            "range": "± 335936",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 10",
            "value": 6024322,
            "range": "± 49066",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 100",
            "value": 26891585,
            "range": "± 420945",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 1000",
            "value": 183827476,
            "range": "± 2142072",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/DenseMatrix",
            "value": 3084975445,
            "range": "± 14150851",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/ndarray",
            "value": 3284677990,
            "range": "± 7924881",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 10",
            "value": 19306755,
            "range": "± 130761",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 100",
            "value": 67593977,
            "range": "± 1700349",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 10",
            "value": 514081589,
            "range": "± 2275248",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 100",
            "value": 1769622804,
            "range": "± 42034266",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 10",
            "value": 2096352411,
            "range": "± 9310900",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 100",
            "value": 7134354319,
            "range": "± 53813527",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/square/128x128",
            "value": 844971,
            "range": "± 1372",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/tall/256x64",
            "value": 591569,
            "range": "± 1781",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/square/128x128",
            "value": 861339,
            "range": "± 3672",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/tall/256x64",
            "value": 517323,
            "range": "± 1746",
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
          "id": "a0b2669a95ea9bc02f12c58c06019c5adbb59652",
          "message": "bench(criterion): record v0.6.10 results [skip ci]",
          "timestamp": "2026-08-21T16:20:36Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/a0b2669a95ea9bc02f12c58c06019c5adbb59652"
        },
        "date": 1787385764616,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63537974,
            "range": "± 207898",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 74101713,
            "range": "± 93973",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 77026666,
            "range": "± 518882",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 69526520,
            "range": "± 860997",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 146543,
            "range": "± 1546",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 496216,
            "range": "± 11401",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1447348,
            "range": "± 28543",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4854788,
            "range": "± 95071",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 17433458,
            "range": "± 115017",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 52090816,
            "range": "± 425746",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 79501,
            "range": "± 373",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 431025,
            "range": "± 2167",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 792338,
            "range": "± 764",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4315399,
            "range": "± 40970",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 7980636,
            "range": "± 31305",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 53859706,
            "range": "± 375797",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 68085,
            "range": "± 98",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 412284,
            "range": "± 1474",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 676407,
            "range": "± 1816",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4114568,
            "range": "± 9250",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6764086,
            "range": "± 32558",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 41157312,
            "range": "± 46434",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 36,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 55653,
            "range": "± 333",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 44405,
            "range": "± 1400",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 305590,
            "range": "± 3735",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 219861,
            "range": "± 2542",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2789438,
            "range": "± 15101",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 1952862,
            "range": "± 15645",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16254,
            "range": "± 96",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 12955,
            "range": "± 203",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 95326,
            "range": "± 1642",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 75014,
            "range": "± 154",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 896421,
            "range": "± 15438",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 686655,
            "range": "± 4776",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/1024",
            "value": 4728006,
            "range": "± 220413",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/1024",
            "value": 47698849,
            "range": "± 150661",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/1024",
            "value": 42169385,
            "range": "± 164857",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/1024",
            "value": 4733942,
            "range": "± 397445",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/4096",
            "value": 72820337,
            "range": "± 177941",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/4096",
            "value": 956526507,
            "range": "± 1660634",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/4096",
            "value": 942749777,
            "range": "± 4000465",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/4096",
            "value": 72037897,
            "range": "± 297835",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/1024",
            "value": 3745843,
            "range": "± 61075",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/1024",
            "value": 48260690,
            "range": "± 162445",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/4096",
            "value": 72694763,
            "range": "± 443284",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/4096",
            "value": 954781726,
            "range": "± 2610279",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 64, n_features: 16",
            "value": 37856,
            "range": "± 80",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 256, n_features: 64",
            "value": 711241,
            "range": "± 889",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 1024, n_features: 256",
            "value": 16309800,
            "range": "± 208786",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/64",
            "value": 961644,
            "range": "± 3077",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/256",
            "value": 63457319,
            "range": "± 72677",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/1024",
            "value": 4511231552,
            "range": "± 4361702",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 10",
            "value": 25668,
            "range": "± 134",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 100",
            "value": 141910,
            "range": "± 1731",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 1000",
            "value": 1295950,
            "range": "± 1940",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 10",
            "value": 342349,
            "range": "± 5315",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 100",
            "value": 2120298,
            "range": "± 19488",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 1000",
            "value": 17133105,
            "range": "± 239966",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 10",
            "value": 5999893,
            "range": "± 208289",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 100",
            "value": 24683315,
            "range": "± 274812",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 1000",
            "value": 179042649,
            "range": "± 710504",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/DenseMatrix",
            "value": 3132356923,
            "range": "± 9850684",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/ndarray",
            "value": 3276916139,
            "range": "± 3963184",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 10",
            "value": 19413389,
            "range": "± 282109",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 100",
            "value": 67622169,
            "range": "± 227764",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 10",
            "value": 517722620,
            "range": "± 1657724",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 100",
            "value": 1767774377,
            "range": "± 4314740",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 10",
            "value": 2131064225,
            "range": "± 3710134",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 100",
            "value": 7127096929,
            "range": "± 9036982",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/square/128x128",
            "value": 852366,
            "range": "± 13201",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/tall/256x64",
            "value": 622635,
            "range": "± 1513",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/square/128x128",
            "value": 858924,
            "range": "± 2029",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/tall/256x64",
            "value": 520226,
            "range": "± 738",
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
          "id": "a0b2669a95ea9bc02f12c58c06019c5adbb59652",
          "message": "bench(criterion): record v0.6.10 results [skip ci]",
          "timestamp": "2026-08-21T16:20:36Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/a0b2669a95ea9bc02f12c58c06019c5adbb59652"
        },
        "date": 1787472265040,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63445348,
            "range": "± 59996",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 74125085,
            "range": "± 653320",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 81226035,
            "range": "± 133039",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 73499915,
            "range": "± 775801",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 145381,
            "range": "± 1014",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 491972,
            "range": "± 11605",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1447819,
            "range": "± 26895",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4807628,
            "range": "± 88143",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 16594548,
            "range": "± 73476",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 51881581,
            "range": "± 1842294",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 79581,
            "range": "± 989",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 430589,
            "range": "± 737",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 794936,
            "range": "± 6320",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4383305,
            "range": "± 194730",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8000057,
            "range": "± 121156",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 54709160,
            "range": "± 404180",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 68285,
            "range": "± 260",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 406881,
            "range": "± 2378",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 681453,
            "range": "± 956",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4058340,
            "range": "± 13084",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6813389,
            "range": "± 17484",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40655026,
            "range": "± 422470",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 36,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 56077,
            "range": "± 651",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 44488,
            "range": "± 202",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 306816,
            "range": "± 959",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 220042,
            "range": "± 7959",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2818395,
            "range": "± 28859",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 1983209,
            "range": "± 127267",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16107,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13071,
            "range": "± 191",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 97438,
            "range": "± 1699",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 75575,
            "range": "± 768",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 899733,
            "range": "± 5669",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 686187,
            "range": "± 10922",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/1024",
            "value": 3774917,
            "range": "± 405451",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/1024",
            "value": 48230826,
            "range": "± 692335",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/1024",
            "value": 42883183,
            "range": "± 383188",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/1024",
            "value": 4731170,
            "range": "± 324729",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/4096",
            "value": 72963923,
            "range": "± 464130",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/4096",
            "value": 970742454,
            "range": "± 7394902",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/4096",
            "value": 940071924,
            "range": "± 8617557",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/4096",
            "value": 71888020,
            "range": "± 360404",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/1024",
            "value": 3747777,
            "range": "± 14622",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/1024",
            "value": 47442871,
            "range": "± 235190",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/4096",
            "value": 73149684,
            "range": "± 279865",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/4096",
            "value": 965077481,
            "range": "± 12203486",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 64, n_features: 16",
            "value": 37797,
            "range": "± 73",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 256, n_features: 64",
            "value": 730559,
            "range": "± 5681",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 1024, n_features: 256",
            "value": 16916731,
            "range": "± 124129",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/64",
            "value": 1084360,
            "range": "± 909",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/256",
            "value": 63202442,
            "range": "± 1302466",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/1024",
            "value": 4591775479,
            "range": "± 56254166",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 10",
            "value": 25994,
            "range": "± 210",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 100",
            "value": 134955,
            "range": "± 3490",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 1000",
            "value": 1223586,
            "range": "± 22580",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 10",
            "value": 334149,
            "range": "± 4183",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 100",
            "value": 2096972,
            "range": "± 7185",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 1000",
            "value": 17550048,
            "range": "± 249496",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 10",
            "value": 6063151,
            "range": "± 69259",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 100",
            "value": 25227833,
            "range": "± 275912",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 1000",
            "value": 180984820,
            "range": "± 2737130",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/DenseMatrix",
            "value": 3155171681,
            "range": "± 26686887",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/ndarray",
            "value": 3286502139,
            "range": "± 4495415",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 10",
            "value": 19688877,
            "range": "± 793927",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 100",
            "value": 67589100,
            "range": "± 507437",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 10",
            "value": 521389552,
            "range": "± 5831763",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 100",
            "value": 1772136289,
            "range": "± 8729891",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 10",
            "value": 2128945568,
            "range": "± 11197676",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 100",
            "value": 7145336713,
            "range": "± 16411906",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/square/128x128",
            "value": 849118,
            "range": "± 1538",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/tall/256x64",
            "value": 599180,
            "range": "± 3336",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/square/128x128",
            "value": 849387,
            "range": "± 1792",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/tall/256x64",
            "value": 510471,
            "range": "± 4227",
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
          "id": "a0b2669a95ea9bc02f12c58c06019c5adbb59652",
          "message": "bench(criterion): record v0.6.10 results [skip ci]",
          "timestamp": "2026-08-21T16:20:36Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/a0b2669a95ea9bc02f12c58c06019c5adbb59652"
        },
        "date": 1787559983793,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 70707926,
            "range": "± 2080884",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 72079577,
            "range": "± 2976247",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 73609593,
            "range": "± 862263",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 72182898,
            "range": "± 280228",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 147676,
            "range": "± 1107",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 528828,
            "range": "± 13064",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1464862,
            "range": "± 9119",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 5293956,
            "range": "± 100753",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 15349919,
            "range": "± 765138",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 55829896,
            "range": "± 411722",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 82558,
            "range": "± 506",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 470653,
            "range": "± 1919",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 817127,
            "range": "± 2229",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 5889137,
            "range": "± 235470",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8363204,
            "range": "± 68123",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 62756978,
            "range": "± 284350",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 70226,
            "range": "± 173",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 451511,
            "range": "± 629",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 700252,
            "range": "± 934",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4518907,
            "range": "± 2405",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 7019838,
            "range": "± 6833",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 45186771,
            "range": "± 30312",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 39,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 57878,
            "range": "± 285",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 46003,
            "range": "± 458",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 321296,
            "range": "± 8954",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 232774,
            "range": "± 3860",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 3003715,
            "range": "± 7243",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2116959,
            "range": "± 12010",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16699,
            "range": "± 116",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13470,
            "range": "± 74",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 100732,
            "range": "± 668",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 79633,
            "range": "± 990",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 954020,
            "range": "± 2294",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 749836,
            "range": "± 5087",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/1024",
            "value": 3871418,
            "range": "± 23637",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/1024",
            "value": 52595872,
            "range": "± 208497",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/1024",
            "value": 46178115,
            "range": "± 467798",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/1024",
            "value": 3862621,
            "range": "± 5977",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/4096",
            "value": 71877647,
            "range": "± 189755",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/4096",
            "value": 951670195,
            "range": "± 12993531",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/4096",
            "value": 950323540,
            "range": "± 5679995",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/4096",
            "value": 70826656,
            "range": "± 186652",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/1024",
            "value": 3857920,
            "range": "± 6315",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/1024",
            "value": 52831952,
            "range": "± 224179",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/4096",
            "value": 71935370,
            "range": "± 265986",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/4096",
            "value": 940806765,
            "range": "± 9535460",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 64, n_features: 16",
            "value": 38548,
            "range": "± 344",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 256, n_features: 64",
            "value": 732539,
            "range": "± 4218",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 1024, n_features: 256",
            "value": 18231907,
            "range": "± 80481",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/64",
            "value": 1036362,
            "range": "± 1505",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/256",
            "value": 70509587,
            "range": "± 56786",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/1024",
            "value": 4759393932,
            "range": "± 11761446",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 10",
            "value": 24321,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 100",
            "value": 129035,
            "range": "± 1883",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 1000",
            "value": 1167020,
            "range": "± 3754",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 10",
            "value": 331089,
            "range": "± 1494",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 100",
            "value": 2187337,
            "range": "± 16765",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 1000",
            "value": 18248624,
            "range": "± 152388",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 10",
            "value": 6303705,
            "range": "± 63775",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 100",
            "value": 28451237,
            "range": "± 220119",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 1000",
            "value": 183770725,
            "range": "± 1040424",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/DenseMatrix",
            "value": 3109157141,
            "range": "± 13568298",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/ndarray",
            "value": 3651584653,
            "range": "± 8847095",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 10",
            "value": 18224500,
            "range": "± 50712",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 100",
            "value": 70580291,
            "range": "± 281997",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 10",
            "value": 487138198,
            "range": "± 6020914",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 100",
            "value": 1847946931,
            "range": "± 66775674",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 10",
            "value": 1986764834,
            "range": "± 5336162",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 100",
            "value": 7462668621,
            "range": "± 18259064",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/square/128x128",
            "value": 766983,
            "range": "± 4720",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/tall/256x64",
            "value": 531574,
            "range": "± 2219",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/square/128x128",
            "value": 778985,
            "range": "± 1015",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/tall/256x64",
            "value": 393914,
            "range": "± 406",
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
          "id": "c9e5a99a1813d61ff3026797cc708737a2d97ea8",
          "message": "bench(criterion): record v0.6.11 results [skip ci]",
          "timestamp": "2026-08-24T08:26:26Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/c9e5a99a1813d61ff3026797cc708737a2d97ea8"
        },
        "date": 1787593124407,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63529544,
            "range": "± 1403660",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 74189840,
            "range": "± 107826",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 77191996,
            "range": "± 150576",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 74428522,
            "range": "± 405851",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 148913,
            "range": "± 2595",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 491566,
            "range": "± 10494",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1454153,
            "range": "± 44770",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4861549,
            "range": "± 90556",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14388947,
            "range": "± 800888",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 51862619,
            "range": "± 304462",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 78998,
            "range": "± 183",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 429335,
            "range": "± 718",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 794106,
            "range": "± 2822",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4335277,
            "range": "± 21693",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 7910318,
            "range": "± 34781",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 55707111,
            "range": "± 341285",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 67223,
            "range": "± 149",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 405826,
            "range": "± 664",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 670140,
            "range": "± 16024",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4049822,
            "range": "± 5815",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6694533,
            "range": "± 4515",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40517199,
            "range": "± 19537",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 37,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 56847,
            "range": "± 421",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 44152,
            "range": "± 119",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 305638,
            "range": "± 2380",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 221030,
            "range": "± 1931",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2822942,
            "range": "± 41035",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 1953882,
            "range": "± 8223",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16239,
            "range": "± 215",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13012,
            "range": "± 295",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 96367,
            "range": "± 1144",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 75098,
            "range": "± 155",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 894429,
            "range": "± 7192",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 688773,
            "range": "± 3262",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/1024",
            "value": 4703504,
            "range": "± 435218",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/1024",
            "value": 49004034,
            "range": "± 270934",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/1024",
            "value": 42929899,
            "range": "± 176808",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/1024",
            "value": 3871824,
            "range": "± 427069",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/4096",
            "value": 76600698,
            "range": "± 204514",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/4096",
            "value": 979355823,
            "range": "± 4262345",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/4096",
            "value": 968016923,
            "range": "± 3508669",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/4096",
            "value": 73065258,
            "range": "± 203355",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/1024",
            "value": 4894822,
            "range": "± 478757",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/1024",
            "value": 50032565,
            "range": "± 534488",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/4096",
            "value": 76547511,
            "range": "± 179806",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/4096",
            "value": 976074910,
            "range": "± 3652518",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 64, n_features: 16",
            "value": 37679,
            "range": "± 57",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 256, n_features: 64",
            "value": 700936,
            "range": "± 5225",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 1024, n_features: 256",
            "value": 16003096,
            "range": "± 123444",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/64",
            "value": 962916,
            "range": "± 867",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/256",
            "value": 63152165,
            "range": "± 676624",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/1024",
            "value": 4564523860,
            "range": "± 58142088",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 10",
            "value": 25235,
            "range": "± 197",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 100",
            "value": 130946,
            "range": "± 305",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 1000",
            "value": 1286784,
            "range": "± 61812",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 10",
            "value": 331313,
            "range": "± 1367",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 100",
            "value": 2071221,
            "range": "± 29336",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 1000",
            "value": 17335667,
            "range": "± 255856",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 10",
            "value": 6001518,
            "range": "± 26983",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 100",
            "value": 25486819,
            "range": "± 466124",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 1000",
            "value": 179375291,
            "range": "± 2809739",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/DenseMatrix",
            "value": 3259203632,
            "range": "± 11770102",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/ndarray",
            "value": 3287773435,
            "range": "± 4311443",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 10",
            "value": 19447786,
            "range": "± 142010",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 100",
            "value": 68232317,
            "range": "± 408544",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 10",
            "value": 515205394,
            "range": "± 4445309",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 100",
            "value": 1814190589,
            "range": "± 89959557",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 10",
            "value": 2087351687,
            "range": "± 10832107",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 100",
            "value": 7496719936,
            "range": "± 306798934",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/square/128x128",
            "value": 846118,
            "range": "± 1766",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/tall/256x64",
            "value": 593389,
            "range": "± 9174",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/square/128x128",
            "value": 838936,
            "range": "± 2519",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/tall/256x64",
            "value": 514128,
            "range": "± 765",
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
          "id": "d28e8cea28566ca0127bf981e8d3fd75b9130478",
          "message": "bench(criterion): record v0.6.12 results [skip ci]",
          "timestamp": "2026-08-24T17:38:48Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/d28e8cea28566ca0127bf981e8d3fd75b9130478"
        },
        "date": 1787645754521,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 67068623,
            "range": "± 648159",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 74247883,
            "range": "± 73120",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 84601342,
            "range": "± 1366665",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 75508116,
            "range": "± 833504",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 145673,
            "range": "± 1415",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 495077,
            "range": "± 15254",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1440143,
            "range": "± 33983",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4849280,
            "range": "± 110491",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 18174026,
            "range": "± 136823",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 51362677,
            "range": "± 481153",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 79177,
            "range": "± 291",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 430556,
            "range": "± 612",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 796676,
            "range": "± 2425",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 5109835,
            "range": "± 200424",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8294835,
            "range": "± 19390",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 56011961,
            "range": "± 520176",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 67318,
            "range": "± 73",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 405981,
            "range": "± 550",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 669888,
            "range": "± 2529",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4055662,
            "range": "± 13183",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6711033,
            "range": "± 7908",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40573620,
            "range": "± 505106",
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
            "value": 57022,
            "range": "± 318",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 44653,
            "range": "± 1236",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 305139,
            "range": "± 3356",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 219804,
            "range": "± 461",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2794968,
            "range": "± 28210",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 1953365,
            "range": "± 8720",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16249,
            "range": "± 108",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 12941,
            "range": "± 164",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 95323,
            "range": "± 640",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 74982,
            "range": "± 142",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 884655,
            "range": "± 7419",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 688703,
            "range": "± 3891",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/1024",
            "value": 4004268,
            "range": "± 304005",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/1024",
            "value": 50465733,
            "range": "± 201449",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/1024",
            "value": 44326527,
            "range": "± 581732",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/1024",
            "value": 3836228,
            "range": "± 475850",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/4096",
            "value": 77350399,
            "range": "± 290242",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/4096",
            "value": 994525684,
            "range": "± 5592358",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/4096",
            "value": 982303808,
            "range": "± 4490049",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/4096",
            "value": 72814833,
            "range": "± 639342",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/1024",
            "value": 3924486,
            "range": "± 17582",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/1024",
            "value": 50362895,
            "range": "± 182913",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/4096",
            "value": 76669731,
            "range": "± 627763",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/4096",
            "value": 988266797,
            "range": "± 4776553",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 64, n_features: 16",
            "value": 37268,
            "range": "± 371",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 256, n_features: 64",
            "value": 684392,
            "range": "± 2273",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 1024, n_features: 256",
            "value": 16150719,
            "range": "± 491586",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/64",
            "value": 971948,
            "range": "± 25157",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/256",
            "value": 63297322,
            "range": "± 71609",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/1024",
            "value": 4680507164,
            "range": "± 38015372",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 10",
            "value": 25257,
            "range": "± 204",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 100",
            "value": 130863,
            "range": "± 2020",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 1000",
            "value": 1204886,
            "range": "± 33496",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 10",
            "value": 333617,
            "range": "± 727",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 100",
            "value": 2059225,
            "range": "± 8657",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 1000",
            "value": 17588688,
            "range": "± 115363",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 10",
            "value": 6062003,
            "range": "± 63269",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 100",
            "value": 25852167,
            "range": "± 232309",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 1000",
            "value": 180599411,
            "range": "± 1983377",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/DenseMatrix",
            "value": 3280438785,
            "range": "± 9091952",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/ndarray",
            "value": 3290890398,
            "range": "± 7599227",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 10",
            "value": 19335111,
            "range": "± 152433",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 100",
            "value": 67913287,
            "range": "± 3293796",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 10",
            "value": 513776775,
            "range": "± 2390518",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 100",
            "value": 1942880140,
            "range": "± 73011367",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 10",
            "value": 2087197839,
            "range": "± 13343623",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 100",
            "value": 7267868436,
            "range": "± 322911312",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/square/128x128",
            "value": 842913,
            "range": "± 4557",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/tall/256x64",
            "value": 594547,
            "range": "± 4875",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/square/128x128",
            "value": 839148,
            "range": "± 1896",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/tall/256x64",
            "value": 513091,
            "range": "± 7450",
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
          "id": "d28e8cea28566ca0127bf981e8d3fd75b9130478",
          "message": "bench(criterion): record v0.6.12 results [skip ci]",
          "timestamp": "2026-08-24T17:38:48Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/d28e8cea28566ca0127bf981e8d3fd75b9130478"
        },
        "date": 1787655243220,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 70577209,
            "range": "± 95445",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 71912793,
            "range": "± 1079191",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 72987517,
            "range": "± 1063339",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 72241929,
            "range": "± 86652",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 145543,
            "range": "± 650",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 528752,
            "range": "± 12220",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1436013,
            "range": "± 10863",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 5243097,
            "range": "± 93238",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 14937889,
            "range": "± 650131",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 56080746,
            "range": "± 428221",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 82679,
            "range": "± 193",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 472552,
            "range": "± 546",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 828777,
            "range": "± 2843",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4866234,
            "range": "± 7629",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8202057,
            "range": "± 31975",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 61535079,
            "range": "± 1059443",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 69005,
            "range": "± 337",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 450438,
            "range": "± 3828",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 688064,
            "range": "± 3020",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4493591,
            "range": "± 4832",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6914546,
            "range": "± 18695",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 44970979,
            "range": "± 40711",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 39,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 57415,
            "range": "± 664",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 45975,
            "range": "± 429",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 321951,
            "range": "± 1278",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 230284,
            "range": "± 1258",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 3012918,
            "range": "± 4816",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2114848,
            "range": "± 21930",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16696,
            "range": "± 96",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13700,
            "range": "± 86",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 101563,
            "range": "± 298",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 79238,
            "range": "± 462",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 963208,
            "range": "± 8067",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 746816,
            "range": "± 4381",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/1024",
            "value": 3856377,
            "range": "± 49579",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/1024",
            "value": 50797405,
            "range": "± 130499",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/1024",
            "value": 44668157,
            "range": "± 128395",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/1024",
            "value": 3856269,
            "range": "± 8395",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/4096",
            "value": 71662128,
            "range": "± 135556",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/4096",
            "value": 948151021,
            "range": "± 10006063",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/4096",
            "value": 941334648,
            "range": "± 8327731",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/4096",
            "value": 70710426,
            "range": "± 737248",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/1024",
            "value": 3853967,
            "range": "± 6202",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/1024",
            "value": 51533941,
            "range": "± 226468",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/4096",
            "value": 71562533,
            "range": "± 116708",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/4096",
            "value": 940606150,
            "range": "± 10229472",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 64, n_features: 16",
            "value": 38475,
            "range": "± 570",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 256, n_features: 64",
            "value": 740169,
            "range": "± 1223",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 1024, n_features: 256",
            "value": 17842028,
            "range": "± 59630",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/64",
            "value": 1036369,
            "range": "± 1018",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/256",
            "value": 70313320,
            "range": "± 158726",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/1024",
            "value": 4789579695,
            "range": "± 25364660",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 10",
            "value": 24404,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 100",
            "value": 139069,
            "range": "± 4777",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 1000",
            "value": 1142484,
            "range": "± 18390",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 10",
            "value": 327731,
            "range": "± 884",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 100",
            "value": 2147637,
            "range": "± 12156",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 1000",
            "value": 16996886,
            "range": "± 61958",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 10",
            "value": 6330371,
            "range": "± 174462",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 100",
            "value": 26451867,
            "range": "± 281812",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 1000",
            "value": 180329823,
            "range": "± 491383",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/DenseMatrix",
            "value": 3426992726,
            "range": "± 11645823",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/ndarray",
            "value": 3624073892,
            "range": "± 10403408",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 10",
            "value": 18115861,
            "range": "± 132857",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 100",
            "value": 70449604,
            "range": "± 83569",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 10",
            "value": 479602425,
            "range": "± 1538426",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 100",
            "value": 1846428826,
            "range": "± 3594542",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 10",
            "value": 1975119506,
            "range": "± 5167194",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 100",
            "value": 7466801173,
            "range": "± 18885124",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/square/128x128",
            "value": 768825,
            "range": "± 2009",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/tall/256x64",
            "value": 533851,
            "range": "± 3213",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/square/128x128",
            "value": 843371,
            "range": "± 2819",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/tall/256x64",
            "value": 409342,
            "range": "± 2211",
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
          "id": "1b2eb0fde4141dcade56ace88e9726713cf0de22",
          "message": "bench(criterion): record v0.6.13 results [skip ci]",
          "timestamp": "2026-08-25T10:54:07Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/1b2eb0fde4141dcade56ace88e9726713cf0de22"
        },
        "date": 1787732264528,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63867323,
            "range": "± 641375",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 74306742,
            "range": "± 83629",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 80050463,
            "range": "± 448766",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 75956804,
            "range": "± 251070",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 145434,
            "range": "± 1427",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 501412,
            "range": "± 33242",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1436745,
            "range": "± 32428",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4865770,
            "range": "± 91507",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 15121845,
            "range": "± 917457",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 52245555,
            "range": "± 388033",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 78839,
            "range": "± 138",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 432120,
            "range": "± 993",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 792450,
            "range": "± 18563",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 5497895,
            "range": "± 65043",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8436283,
            "range": "± 30875",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 55598358,
            "range": "± 207704",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 67209,
            "range": "± 955",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 406117,
            "range": "± 3727",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 669813,
            "range": "± 934",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4058991,
            "range": "± 2627",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6712694,
            "range": "± 21361",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40593681,
            "range": "± 784533",
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
            "value": 55382,
            "range": "± 177",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 44988,
            "range": "± 1140",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 304494,
            "range": "± 934",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 221739,
            "range": "± 878",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2798788,
            "range": "± 28710",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 1961087,
            "range": "± 127348",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 15901,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13124,
            "range": "± 330",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 95171,
            "range": "± 959",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 75789,
            "range": "± 975",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 890126,
            "range": "± 21465",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 689641,
            "range": "± 12902",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/1024",
            "value": 4924831,
            "range": "± 223024",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/1024",
            "value": 49601758,
            "range": "± 677185",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/1024",
            "value": 43545252,
            "range": "± 326035",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/1024",
            "value": 4743036,
            "range": "± 214507",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/4096",
            "value": 76528316,
            "range": "± 235106",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/4096",
            "value": 981351071,
            "range": "± 4234064",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/4096",
            "value": 975914005,
            "range": "± 4366130",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/4096",
            "value": 72770529,
            "range": "± 380145",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/1024",
            "value": 3585992,
            "range": "± 7782",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/1024",
            "value": 50018883,
            "range": "± 353536",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/4096",
            "value": 70987225,
            "range": "± 2353017",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/4096",
            "value": 970582959,
            "range": "± 6076395",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 64, n_features: 16",
            "value": 37297,
            "range": "± 225",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 256, n_features: 64",
            "value": 715728,
            "range": "± 6678",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 1024, n_features: 256",
            "value": 16373369,
            "range": "± 93312",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/64",
            "value": 1010197,
            "range": "± 1195",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/256",
            "value": 63241686,
            "range": "± 627107",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/1024",
            "value": 4747910083,
            "range": "± 73773937",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 10",
            "value": 25267,
            "range": "± 442",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 100",
            "value": 131003,
            "range": "± 853",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 1000",
            "value": 1188324,
            "range": "± 20759",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 10",
            "value": 332253,
            "range": "± 5012",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 100",
            "value": 2058114,
            "range": "± 6508",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 1000",
            "value": 17220607,
            "range": "± 70241",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 10",
            "value": 6036651,
            "range": "± 54806",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 100",
            "value": 25734287,
            "range": "± 217775",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 1000",
            "value": 178156547,
            "range": "± 3709521",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/DenseMatrix",
            "value": 3266565139,
            "range": "± 17943305",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/ndarray",
            "value": 3286446634,
            "range": "± 8989117",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 10",
            "value": 20057858,
            "range": "± 278701",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 100",
            "value": 67811221,
            "range": "± 202495",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 10",
            "value": 538508662,
            "range": "± 4659537",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 100",
            "value": 1780280978,
            "range": "± 79060999",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 10",
            "value": 2173068101,
            "range": "± 15345934",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 100",
            "value": 7272912631,
            "range": "± 329683742",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/square/128x128",
            "value": 849192,
            "range": "± 5926",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/tall/256x64",
            "value": 591751,
            "range": "± 1958",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/square/128x128",
            "value": 861573,
            "range": "± 4921",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/tall/256x64",
            "value": 504802,
            "range": "± 8226",
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
          "id": "1b2eb0fde4141dcade56ace88e9726713cf0de22",
          "message": "bench(criterion): record v0.6.13 results [skip ci]",
          "timestamp": "2026-08-25T10:54:07Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/1b2eb0fde4141dcade56ace88e9726713cf0de22"
        },
        "date": 1787747967275,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 63354034,
            "range": "± 70280",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 74164608,
            "range": "± 2050900",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 76563727,
            "range": "± 411918",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 69615665,
            "range": "± 145462",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 144406,
            "range": "± 1610",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 493018,
            "range": "± 12061",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1461068,
            "range": "± 29590",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4852668,
            "range": "± 88932",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 18027789,
            "range": "± 484448",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 51453733,
            "range": "± 426231",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 78733,
            "range": "± 177",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 431436,
            "range": "± 1369",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 793473,
            "range": "± 1542",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4388148,
            "range": "± 186733",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8144939,
            "range": "± 49440",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 55612980,
            "range": "± 261780",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 68208,
            "range": "± 178",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 406947,
            "range": "± 976",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 682278,
            "range": "± 2192",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4060827,
            "range": "± 54741",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6798566,
            "range": "± 27853",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40642081,
            "range": "± 304928",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 36,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 57041,
            "range": "± 576",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 44002,
            "range": "± 228",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 305160,
            "range": "± 4698",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 222049,
            "range": "± 1208",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2818983,
            "range": "± 27615",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 1953376,
            "range": "± 11138",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16290,
            "range": "± 186",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 12948,
            "range": "± 111",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 96239,
            "range": "± 799",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 75059,
            "range": "± 312",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 893185,
            "range": "± 8182",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 689514,
            "range": "± 2754",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/1024",
            "value": 4887726,
            "range": "± 451951",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/1024",
            "value": 48490057,
            "range": "± 347405",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/1024",
            "value": 43337613,
            "range": "± 429385",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/1024",
            "value": 4736043,
            "range": "± 431931",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/4096",
            "value": 76491012,
            "range": "± 525382",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/4096",
            "value": 980538055,
            "range": "± 5627715",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/4096",
            "value": 968640012,
            "range": "± 7193883",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/4096",
            "value": 73505015,
            "range": "± 3850424",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/1024",
            "value": 3943096,
            "range": "± 20039",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/1024",
            "value": 49208354,
            "range": "± 302290",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/4096",
            "value": 76635168,
            "range": "± 780674",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/4096",
            "value": 992142876,
            "range": "± 4648994",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 64, n_features: 16",
            "value": 38688,
            "range": "± 737",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 256, n_features: 64",
            "value": 721942,
            "range": "± 680",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 1024, n_features: 256",
            "value": 17710851,
            "range": "± 245798",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/64",
            "value": 964052,
            "range": "± 33944",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/256",
            "value": 63349272,
            "range": "± 1191695",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/1024",
            "value": 4791107432,
            "range": "± 96546921",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 10",
            "value": 25609,
            "range": "± 208",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 100",
            "value": 135133,
            "range": "± 4695",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 1000",
            "value": 1298692,
            "range": "± 21576",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 10",
            "value": 341104,
            "range": "± 4829",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 100",
            "value": 2194096,
            "range": "± 38371",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 1000",
            "value": 18134812,
            "range": "± 229373",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 10",
            "value": 6149759,
            "range": "± 119072",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 100",
            "value": 27022425,
            "range": "± 660885",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 1000",
            "value": 188388829,
            "range": "± 1397649",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/DenseMatrix",
            "value": 3058853704,
            "range": "± 28573332",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/ndarray",
            "value": 3283623788,
            "range": "± 5617388",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 10",
            "value": 19298175,
            "range": "± 98205",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 100",
            "value": 67631213,
            "range": "± 768734",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 10",
            "value": 514665455,
            "range": "± 2778425",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 100",
            "value": 1787593131,
            "range": "± 35997102",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 10",
            "value": 2095583599,
            "range": "± 6531411",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 100",
            "value": 7196612030,
            "range": "± 80128857",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/square/128x128",
            "value": 853682,
            "range": "± 7217",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/tall/256x64",
            "value": 608878,
            "range": "± 5340",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/square/128x128",
            "value": 865895,
            "range": "± 2779",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/tall/256x64",
            "value": 520342,
            "range": "± 1942",
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
          "id": "147686f018600f3be5f4f6c46b824e183fd6a89d",
          "message": "bench(criterion): record v0.6.14 results [skip ci]",
          "timestamp": "2026-08-26T12:39:31Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/147686f018600f3be5f4f6c46b824e183fd6a89d"
        },
        "date": 1787855903770,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 70645647,
            "range": "± 335565",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 77418145,
            "range": "± 2565392",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 73600975,
            "range": "± 97366",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 72130376,
            "range": "± 79228",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 148836,
            "range": "± 723",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 528042,
            "range": "± 11984",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1474115,
            "range": "± 12635",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 5246518,
            "range": "± 99530",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 15319586,
            "range": "± 574562",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 55795033,
            "range": "± 526832",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 82400,
            "range": "± 282",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 471275,
            "range": "± 491",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 826419,
            "range": "± 2759",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4935627,
            "range": "± 42390",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8487758,
            "range": "± 29766",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 61558378,
            "range": "± 407146",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 70563,
            "range": "± 1278",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 450882,
            "range": "± 11916",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 706350,
            "range": "± 21714",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4504040,
            "range": "± 17436",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 7030962,
            "range": "± 7901",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 45025673,
            "range": "± 2029660",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 39,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 57247,
            "range": "± 256",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 46043,
            "range": "± 539",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 322603,
            "range": "± 834",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 230958,
            "range": "± 3128",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 3015003,
            "range": "± 10688",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 2111493,
            "range": "± 30854",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 16618,
            "range": "± 68",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 13554,
            "range": "± 112",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 101577,
            "range": "± 282",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 78765,
            "range": "± 1049",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 961333,
            "range": "± 2202",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 745218,
            "range": "± 7391",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/1024",
            "value": 4225105,
            "range": "± 47869",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/1024",
            "value": 50093187,
            "range": "± 176020",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/1024",
            "value": 44651935,
            "range": "± 185790",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/1024",
            "value": 4223557,
            "range": "± 8261",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/4096",
            "value": 77423171,
            "range": "± 429432",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/4096",
            "value": 931622208,
            "range": "± 8526101",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/4096",
            "value": 936396511,
            "range": "± 8918526",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/4096",
            "value": 76407817,
            "range": "± 86835",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/1024",
            "value": 4223136,
            "range": "± 9018",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/1024",
            "value": 50570824,
            "range": "± 183601",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/4096",
            "value": 77410541,
            "range": "± 159966",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/4096",
            "value": 933508725,
            "range": "± 3281351",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 64, n_features: 16",
            "value": 38735,
            "range": "± 65",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 256, n_features: 64",
            "value": 766773,
            "range": "± 8175",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 1024, n_features: 256",
            "value": 18319825,
            "range": "± 65438",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/64",
            "value": 1036559,
            "range": "± 2526",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/256",
            "value": 70404619,
            "range": "± 2369580",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/1024",
            "value": 4955078055,
            "range": "± 57592335",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 10",
            "value": 24349,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 100",
            "value": 128778,
            "range": "± 9007",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 1000",
            "value": 1166503,
            "range": "± 12663",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 10",
            "value": 332067,
            "range": "± 1113",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 100",
            "value": 2161655,
            "range": "± 9518",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 1000",
            "value": 17195887,
            "range": "± 172360",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 10",
            "value": 6277657,
            "range": "± 58521",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 100",
            "value": 26485813,
            "range": "± 805777",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 1000",
            "value": 183543817,
            "range": "± 1794766",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/DenseMatrix",
            "value": 3103693343,
            "range": "± 14263995",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/ndarray",
            "value": 3672908042,
            "range": "± 8169878",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 10",
            "value": 18293804,
            "range": "± 378383",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 100",
            "value": 70478769,
            "range": "± 122281",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 10",
            "value": 480779233,
            "range": "± 1425951",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 100",
            "value": 1851926885,
            "range": "± 4139878",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 10",
            "value": 1968571110,
            "range": "± 25661883",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 100",
            "value": 7439661407,
            "range": "± 12416915",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/square/128x128",
            "value": 766054,
            "range": "± 3492",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/tall/256x64",
            "value": 533028,
            "range": "± 12517",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/square/128x128",
            "value": 785191,
            "range": "± 1268",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/tall/256x64",
            "value": 395907,
            "range": "± 532",
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
          "id": "147686f018600f3be5f4f6c46b824e183fd6a89d",
          "message": "bench(criterion): record v0.6.14 results [skip ci]",
          "timestamp": "2026-08-26T12:39:31Z",
          "url": "https://github.com/smartcorelib/smartcore-benches/commit/147686f018600f3be5f4f6c46b824e183fd6a89d"
        },
        "date": 1787946768618,
        "tool": "cargo",
        "benches": [
          {
            "name": "Array2::ab/a_t=false,b_t=false/256",
            "value": 46038390,
            "range": "± 105704",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=false,b_t=true/256",
            "value": 60105491,
            "range": "± 324173",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=false/256",
            "value": 67434233,
            "range": "± 199599",
            "unit": "ns/iter"
          },
          {
            "name": "Array2::ab/a_t=true,b_t=true/256",
            "value": 62128386,
            "range": "± 162859",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x10/build",
            "value": 126796,
            "range": "± 990",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/1000x100/build",
            "value": 481971,
            "range": "± 8831",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x10/build",
            "value": 1245439,
            "range": "± 74402",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/10000x100/build",
            "value": 4988851,
            "range": "± 153169",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x10/build",
            "value": 17241921,
            "range": "± 159375",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::new/100000x100/build",
            "value": 55046770,
            "range": "± 458938",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x10/k=10",
            "value": 75862,
            "range": "± 127",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/1000x100/k=10",
            "value": 417244,
            "range": "± 536",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x10/k=10",
            "value": 757688,
            "range": "± 1629",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/10000x100/k=10",
            "value": 4212988,
            "range": "± 76458",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x10/k=10",
            "value": 8211751,
            "range": "± 77706",
            "unit": "ns/iter"
          },
          {
            "name": "CoverTree::find/100000x100/k=10",
            "value": 45476456,
            "range": "± 122567",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x10/k=10",
            "value": 62962,
            "range": "± 679",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/1000x100/k=10",
            "value": 405542,
            "range": "± 393",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x10/k=10",
            "value": 612656,
            "range": "± 1360",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/10000x100/k=10",
            "value": 4059640,
            "range": "± 3855",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x10/k=10",
            "value": 6142074,
            "range": "± 11734",
            "unit": "ns/iter"
          },
          {
            "name": "LinearKNNSearch::find/100000x100/k=10",
            "value": 40588450,
            "range": "± 26188",
            "unit": "ns/iter"
          },
          {
            "name": "Euclidean Distance",
            "value": 29,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 10",
            "value": 47628,
            "range": "± 147",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 10",
            "value": 37879,
            "range": "± 175",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 100",
            "value": 263331,
            "range": "± 1819",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 100",
            "value": 202667,
            "range": "± 1105",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 20, n_features: 1000",
            "value": 2433998,
            "range": "± 16867",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 20, n_features: 1000",
            "value": 1794529,
            "range": "± 10218",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 10",
            "value": 13815,
            "range": "± 114",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 10",
            "value": 11228,
            "range": "± 73",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 100",
            "value": 82626,
            "range": "± 813",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 100",
            "value": 67760,
            "range": "± 436",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/fastpair --- n_samples: 10, n_features: 1000",
            "value": 764759,
            "range": "± 8723",
            "unit": "ns/iter"
          },
          {
            "name": "FastPair/brute --- n_samples: 10, n_features: 1000",
            "value": 623064,
            "range": "± 4679",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/1024",
            "value": 4290492,
            "range": "± 106244",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/1024",
            "value": 55419040,
            "range": "± 959303",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/1024",
            "value": 49456481,
            "range": "± 719394",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/1024",
            "value": 4104168,
            "range": "± 82047",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=0/4096",
            "value": 101061069,
            "range": "± 1308414",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/row-major,axis=1/4096",
            "value": 2470302980,
            "range": "± 64501520",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=0/4096",
            "value": 1947613570,
            "range": "± 38360455",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::iterator_mut/col-major,axis=1/4096",
            "value": 94648528,
            "range": "± 1291481",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/1024",
            "value": 4459384,
            "range": "± 100067",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/1024",
            "value": 56720046,
            "range": "± 723416",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=0/4096",
            "value": 96211921,
            "range": "± 546660",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrixMutView::iterator_mut/axis=1/4096",
            "value": 2461675554,
            "range": "± 34733725",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 64, n_features: 16",
            "value": 36649,
            "range": "± 152",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 256, n_features: 64",
            "value": 642883,
            "range": "± 1621",
            "unit": "ns/iter"
          },
          {
            "name": "LinearRegression::fit/n_samples: 1024, n_features: 256",
            "value": 14967140,
            "range": "± 101580",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/64",
            "value": 725347,
            "range": "± 491",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/256",
            "value": 48762386,
            "range": "± 237888",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::matmul/1024",
            "value": 4852847027,
            "range": "± 87402626",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 10",
            "value": 22652,
            "range": "± 67",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 100",
            "value": 109126,
            "range": "± 290",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 100, n_features: 1000",
            "value": 980630,
            "range": "± 1137",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 10",
            "value": 265124,
            "range": "± 976",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 100",
            "value": 1591015,
            "range": "± 7737",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 1000, n_features: 1000",
            "value": 14016728,
            "range": "± 237694",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 10",
            "value": 5244267,
            "range": "± 15432",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 100",
            "value": 21431314,
            "range": "± 219637",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB::fit/n_samples: 10000, n_features: 1000",
            "value": 143893998,
            "range": "± 273424",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/DenseMatrix",
            "value": 9227186974,
            "range": "± 6683194",
            "unit": "ns/iter"
          },
          {
            "name": "GaussianNB/ndarray",
            "value": 9473249992,
            "range": "± 3878940",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 10",
            "value": 16908007,
            "range": "± 17438",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 100, n_features: 100",
            "value": 65094325,
            "range": "± 117622",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 10",
            "value": 454306365,
            "range": "± 781729",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 500, n_features: 100",
            "value": 1698061871,
            "range": "± 1764728",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 10",
            "value": 1840575400,
            "range": "± 1601710",
            "unit": "ns/iter"
          },
          {
            "name": "MultiClassSVC::fit/n_samples: 1000, n_features: 100",
            "value": 6837952460,
            "range": "± 9515882",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/square/128x128",
            "value": 831609,
            "range": "± 7417",
            "unit": "ns/iter"
          },
          {
            "name": "DenseMatrix::svd/tall/256x64",
            "value": 585956,
            "range": "± 1895",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/square/128x128",
            "value": 799132,
            "range": "± 500",
            "unit": "ns/iter"
          },
          {
            "name": "ndarray::Array2::svd/tall/256x64",
            "value": 480573,
            "range": "± 508",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}