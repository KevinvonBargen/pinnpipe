# Reference solutions: provenance, generation, and accuracy

This document records, for every benchmark in `benchmarks/`, how the reference
solution was obtained, how to regenerate it, and how accurate it is.

Benchmarks fall into three classes:

1. **Analytical / manufactured** (`heat`, `poisson`, `wave`): the reference is
   an exact closed-form solution evaluated on the test grid; reference error
   is zero up to floating-point roundoff (float64, ~1e-16). No data files are
   needed; the expressions live in each benchmark's `utils.py`.
2. **Spectral time-stepping references** (`allen_cahn`, `burgers`,
   `ginzburg_landau`, `kdv`, `ks_chaotic`): Fourier spectral in space, ETDRK4
   exponential integrator in time, float64. `ginzburg_landau`, `kdv` and
   `ks_chaotic` were generated with Chebfun's `spin`/`spin2` (MATLAB; the
   `gen_*.m` scripts are in each `data/` folder). `burgers` and `allen_cahn`
   use refined references (`burgers_refined.mat`, `allen_cahn_refined.mat`)
   generated with the independent Python solver of
   `verify_reference_solutions.py` by `data/gen_*_refined.py`; they replace
   the original 511-mode Chebfun files, whose accuracy (4e-5 and 1.2e-5) was
   at the level of the best PINN errors (see below). Accuracy of all
   references was verified with `verify_reference_solutions.py`.
3. **External solver data** (`rayleigh_taylor`): inherited from the upstream
   jaxpi repository (PredictiveIntelligenceLab/jaxpi, commit `75d3fee`,
   Sep 2025); the generating script is not included upstream. It was
   validated independently (see below).

## Summary table

| Benchmark | Reference type | Solver | Spatial resolution | Time step | Output grid | Estimated reference accuracy (rel. L2) |
|---|---|---|---|---|---|---|
| Poisson (5-D) | manufactured, u(x)=Σᵢ sin(π/2 xᵢ) on [0,1]⁵ | exact expression (float64) | 65,536 test points | — | random cloud | ~1e-16 (roundoff) |
| Heat (5-D) | exact, u(x,t)=exp(‖x‖²/2+t), unit ball | exact expression (float64) | 65,536 test points | — | random cloud | ~1e-16 (roundoff) |
| Wave (1-D) | exact, u=sin(πx)cos(2πt)+½sin(4πx)cos(8πt) | exact expression (float64) | 200×128 grid | — | tensor grid | ~1e-16 (roundoff) |
| Burgers | Fourier + ETDRK4 (`verify_reference_solutions.py` solver) | `data/gen_burgers_refined.py` | 2044 Fourier modes on [-1,1] | 2.5e-6 | 201×512 | 1.9e-11 (certified vs 4088 modes, dt 1.25e-6) |
| Allen–Cahn | Fourier + ETDRK4 (`verify_reference_solutions.py` solver) | `data/gen_allen_cahn_refined.py` | 32704 Fourier modes on [-1,1] | 5e-6 | 201×512 | 2.7e-9 (certified vs 65408 modes) |
| KdV | Chebfun `spin` (Fourier + ETDRK4) | `data/gen_kdv.m` | 511 Fourier modes on [-1,1] | 1e-5 | 201×512 | ≈5e-12 |
| Kuramoto–Sivashinsky | Chebfun `spin` (Fourier + ETDRK4) | `data/gen_ks_chaotic.m` | 511 Fourier modes on [0,2π] | 1e-5 | 251×512 | ≈3e-9 |
| Ginzburg–Landau (2-D, complex) | Chebfun `spin2` (Fourier + ETDRK4) | `data/gen_ginzburg_landau_square.m` | 200×200 Fourier modes on [-1,1]² | 1e-3 | 101×200×200 | ≈3e-8 |
| Rayleigh–Taylor (2-D) | external solver run (upstream jaxpi) | generation script not published; independently validated (see below) | 100×200 stretched tensor grid on [0,1]×[0,2] | output Δt=0.25, 40 snapshots | Ra=1e6, Pr=0.71 | ≈5e-5 (early) to ≈3e-3 (developed regime), per output interval in T |

## Verification methodology (spectral references)

`verify_reference_solutions.py` re-solves each PDE with an independent NumPy
implementation of ETDRK4 (Kassam & Trefethen, SIAM J. Sci. Comput. 2005) and
reports three numbers per benchmark:

- **repro** — relative L2 distance between the shipped `.mat` file and the
  independent solver at the *same* resolution and time step. Values at
  roundoff level (~1e-14) demonstrate the shipped data is exactly what the
  `gen_*.m` scripts produce (verified independently of MATLAB/Chebfun).
- **acc** — relative L2 distance between the shipped reference and a refined
  run (2× spatial modes, ½ time step). This is the estimated accuracy of the
  reference itself, to be quoted against the reported PINN errors
  (~1e-7–1e-8).
- **conv** — distance between the refined and doubly-refined (4×, ¼)
  runs, confirming the refined run is converged and can serve as ground truth
  for **acc**.

Measured results (full space-time relative L2, float64). The Burgers and
Allen–Cahn rows below were measured on the **original** 511-mode files
(`burgers.mat`, `allen_cahn.mat`, no longer shipped); they motivated the
refined references, whose accuracy is given in the summary table and in
`benchmarks/{burgers,allen_cahn}/data/README_refined.md`:

| Benchmark | repro (shipped vs independent, same res.) | acc (shipped vs 2× refined) | conv (2× vs 4× refined) |
|---|---|---|---|
| Burgers (original 511-mode file) | 1.06e-14 | 4.01e-05 | 1.24e-08 |
| Allen–Cahn (original 511-mode file) | 3.98e-14 | 8.34e-06 | 2.37e-06 |
| KdV | 4.19e-13 | 5.48e-12 | 1.28e-11 |
| Kuramoto–Sivashinsky | 1.70e-11 | 2.99e-09 | 9.12e-10 |
| Ginzburg–Landau | 1.41e-09 | 3.17e-08 | 2.39e-09 |

Interpretation:

- **repro ~1e-14–1e-11** for all four 1-D benchmarks: the shipped `.mat`
  files are exactly what the included `gen_*.m` scripts produce, confirmed by
  an independent solver implementation (no MATLAB required).
- **KdV (~5e-12)**, **KS (~3e-9)** and **Ginzburg–Landau (~3e-8)** references
  are well below the best reported PINN errors on those benchmarks (KdV best
  ~1.3e-6, KS best ~9e-6). GL's repro value (~1e-9 rather than 1e-14)
  reflects the chebfun2 off-grid interpolation used when sampling the .mat's
  output grid, not a scheme difference.
- **Burgers (~4e-5)**: the reference itself is only accurate to ~4e-5 in
  global relative L2, dominated by the steep front at x=0 (front width ~ν =
  0.01/π ≈ 3.2e-3 vs grid spacing 2/511 ≈ 3.9e-3). Any PINN error reported
  below ~1e-4 against this file measures agreement with the reference, not
  with the true solution. The benchmark therefore uses a refined reference
  (2044 modes, dt=2.5e-6, certified to rel. L2 = 1.9e-11 against a 4088-mode
  run), `burgers/data/burgers_refined.mat` (drop-in, same grid; generated by
  `gen_burgers_refined.py`).
- **Allen–Cahn (~1e-5, order of magnitude)**: refinement converges slowly
  (conv only ~3.5× below acc). The cause is the initial condition
  u0 = x²cos(πx), which is only C⁰ across the periodic seam, so Fourier
  convergence is algebraic (~N⁻²·²) and all of the error sits in a thin
  boundary layer at |x| > 0.9, t < 0.1. The benchmark therefore uses a
  refined reference (32704 modes, dt=5e-6, certified to rel. L2 = 2.7e-9
  against a 65408-mode run), `allen_cahn/data/allen_cahn_refined.mat`
  (drop-in, same grid; generated by `gen_allen_cahn_refined.py`).

### Reference-limited PINN results (resolved by the refined references)

Against the original files, the best PINN runs on Burgers reached rel. L2 =
4.0118e-5 — numerically indistinguishable from the measured reference error
floor of 4.005e-5. The best Allen–Cahn runs (~9.3e-6) likewise sat at the
~8e-6 floor. This is
exactly the signature of a PINN that is *more* accurate than the reference:
if the PINN is close to the true solution, its measured "error" equals the
reference's own error. Both benchmarks now load the refined references
(`burgers_refined.mat`, `allen_cahn_refined.mat`), so errors reported by the
sweeps are measured against the true solution down to ~1e-9. Scores against
the refined files can differ from scores against the old files by up to the
old files' own error (~4e-5 Burgers, ~1.5e-5 Allen–Cahn), so results
computed against the old files are not comparable.

## Known reproducibility caveats

- `gen_kdv.m` currently sets `steps = 250` (251 output times) while the
  shipped `kdv.mat` contains 201 output times (`steps = 200`). The PDE solve
  is identical (`dt = 1e-5` fixed); only the output sampling differs. To
  reproduce the shipped file exactly, set `steps = 200`. The solution values
  at coinciding times agree to roundoff (verified numerically).
- The `gen_*.m` scripts require MATLAB with Chebfun
  (https://www.chebfun.org). The Python verifier requires only
  NumPy/SciPy and reproduces the data to ~1e-14, so it doubles as an
  open-source regeneration path.

## Rayleigh–Taylor: independent validation

`data/rayleigh_taylor_high_Ra.npy` (fields u, v, p, T; 40 snapshots at
Δt=0.25; 100×200 sin-stretched tensor grid; Ra=1e6, Pr=0.71) was committed to
upstream jaxpi without a generation script. Since it cannot be regenerated,
its accuracy was validated independently
(`benchmarks/rayleigh_taylor/evaluate_reference_accuracy.py`, results in
`reference_accuracy_results.txt`):

**Consistency diagnostics.** Boundary conditions check out (no-slip walls,
T=0 walls, x-periodicity). The solution is exactly mirror-symmetric in x at
all times (asymmetry grows only from 1e-16 to ~1e-13 by t=9.76), i.e. a
symmetric single-mode RT configuration whose perturbation amplification over
the whole window is only ~10³ — the flow is deterministic enough for
short-horizon validation to be meaningful.

**Re-solve validation.** Each tested snapshot was used as an initial
condition for an independent solver of the stated Boussinesq system
(2nd-order MAC staggered finite volumes, exact FFT/DCT pressure projection,
explicit SSP-RK2), integrated one and two output intervals, and compared with
the dataset at the next snapshots. Relative L2 discrepancy of T (converged
across solver resolutions 192×384 → 512×1024, i.e. limited by the dataset,
not by the validator):

| IC snapshot | regime | after 0.25 t.u. | after 0.5 t.u. | persistence baseline |
|---|---|---|---|---|
| t=1.26 | smooth spike growth | 5.3e-5 | 1.2e-4 | 2.2e-2 / 4.5e-2 |
| t=3.76 | developed plumes | 2.8e-3 | 4.2e-3 | 1.7e-1 / 3.2e-1 |
| t=7.51 | late mixing | 1.9e-3 | 2.8e-3 | 1.8e-1 / 3.2e-1 |

The dataset therefore evolves consistently with the stated PDE to ≈5e-5
(early) and ≈2–4e-3 (developed regime) per output interval; these numbers are
upper bounds on the dataset's local error (they include the validator's
interpolation error at sharp plume fronts). Best reported PINN errors on this
benchmark (rel. L2 of T ≈ 8.6e-3–1.1e-2) sit a factor ~2–3 above this floor —
the evaluation is meaningful, with a margin of a factor ~2–3. PINN errors
below ~1e-3 on RT cannot be resolved with this reference.

## Provenance

All `gen_*.m` scripts and the `.mat`/`.npy` reference files of `kdv`,
`ks_chaotic`, `ginzburg_landau` and `rayleigh_taylor` originate from the
upstream jaxpi repository (PredictiveIntelligenceLab/jaxpi) and its PirateNets
extensions; they are mirrored here unchanged so the benchmark suite is
self-contained. The refined `burgers`/`allen_cahn` references with their
`gen_*_refined.py` scripts, the verification script and this document were
added in PINNPipe.
