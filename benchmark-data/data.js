window.BENCHMARK_DATA = {
  "lastUpdate": 1786381653883,
  "repoUrl": "https://github.com/DiogoRibeiro7/pinn",
  "entries": {
    "training": [
      {
        "commit": {
          "author": {
            "name": "Diogo Ribeiro",
            "username": "DiogoRibeiro7",
            "email": "diogo_dj@hotmail.com"
          },
          "committer": {
            "name": "Diogo Ribeiro",
            "username": "DiogoRibeiro7",
            "email": "diogo_dj@hotmail.com"
          },
          "id": "38eba34c9dc2b0ae3cde9e22b744ae3e4d8b00f5",
          "message": "Emit benchmark results in the format the action requires\n\nWith the earlier steps fixed, the Performance job reached its final step and\nfailed there:\n\n    Output file for 'custom-(bigger|smaller)-is-better' must be JSON file\n    containing an array of entries in BenchmarkResult format\n    ... \"message\": \"Invalid input: expected array, received object\"\n\nThe conversion step wrapped the entries in a {\"benchmarks\": [...]} object,\nbut customSmallerIsBetter expects a bare array of {name, unit, value}.\nWrite the array directly, and record the constraint in a comment so the\nwrapper does not come back.\n\nVerified against a report produced by scripts/benchmark_training.py: the\noutput is a list whose entries carry exactly those keys.",
          "timestamp": "2026-08-10T17:05:15Z",
          "url": "https://github.com/DiogoRibeiro7/pinn/commit/38eba34c9dc2b0ae3cde9e22b744ae3e4d8b00f5"
        },
        "date": 1786381652322,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "main.<locals>.run_train",
            "value": 1.070253887000007,
            "unit": "s"
          }
        ]
      }
    ]
  }
}