window.BENCHMARK_DATA = {
  "lastUpdate": 1786709431161,
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
          "id": "8f657a977e80e3be6a4a31932ba9763bec6a50ac",
          "message": "Add a composable nonlinear Schrodinger problem\n\nThe second multi-output problem, and the one that shows the two outputs\nneed not be physically distinct fields: here they are the real and\nimaginary parts of one complex field, split by the problem while the\ncore stays unaware.\n\nROADMAP.md recorded this as blocked on \"complex-valued residuals, which\nthe current StrongFormResidual contract does not express\". It was not\nblocked, and that is the same wrong assumption the Navier-Stokes entry\ncarried until yesterday.\n\nThree things were checked rather than assumed. The familiar benchmark\ninitial condition, 2 sech(x), is a breather with no closed form -- it\nleaves a residual of order 1, asserted directly -- while sech(x) is the\nfundamental soliton with the exact solution sech(x) exp(i t / 2),\nverified to 3e-16. So the default amplitude is the soliton, and the\nbreather amplitude reports reference_kind \"unavailable\" with exact()\nraising, rather than quietly scoring against the wrong thing.\n\nBoundary conditions follow from the solution. sech is even, so value\nperiodicity is exact to the last bit; sech' is odd, so derivative\nperiodicity is violated and no flux constraint is imposed; Dirichlet\nwould contradict the exact solution by 0.0135, which is over a percent\nof the peak. All three are asserted so the reasoning cannot erode.\n\nWeighting used the per-component balance the Navier-Stokes work\nestablished rather than rediscovering the 4x it costs to get wrong.\nReaches |h| relative L2 of 4.2e-3, the best of any composable problem\nhere, which is a fact about the soliton and not the framework.\n\nAlso fixes a flaky test surfaced while running the gate.\ntest_mlp_forward_speed asserts 100 forward passes finish inside a\nsecond; it failed once in an 18-minute contended run and passed on a\nquiet one. addopts now deselects the performance marker and\nperformance.yml re-enables it, since without that the job collects\nnothing and exits 5. Both paths verified rather than assumed -- the\nfirst attempt broke the performance workflow silently.\n\n547 passed at 67.50%.",
          "timestamp": "2026-08-14T12:06:26Z",
          "url": "https://github.com/DiogoRibeiro7/pinn/commit/8f657a977e80e3be6a4a31932ba9763bec6a50ac"
        },
        "date": 1786709430352,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "burgers_continuous_pinn_train",
            "value": 1.2182070789999955,
            "unit": "s"
          }
        ]
      }
    ]
  }
}