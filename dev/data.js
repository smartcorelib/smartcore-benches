window.BENCHMARK_DATA = {
  "lastUpdate": 1787472266017,
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
      }
    ]
  }
}