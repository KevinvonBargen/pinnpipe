# Refined Allen–Cahn reference

The original Allen–Cahn reference `allen_cahn.mat` (from
`gen_allen_cahn.m`: Chebfun `spin`, 511 Fourier modes, dt = 1e-5; no longer
shipped) has a measured accuracy of only **1.17e-5 relative L2** on the
benchmark's 201×512 evaluation grid. The cause is the initial condition
u0 = x²cos(πx): it is only C⁰ across the periodic boundary
(u0'(-1) = +2, u0'(1) = -2). The true solution lifts that corner tip by
~2e-3 within t < 0.01 over a boundary layer of width √(10⁻⁴ t) ≈ 1e-3, less
than half a cell of the 512 grid. All of the old file's error sits at
|x| > 0.9, t < 0.1.

Because of that corner the Fourier method converges only algebraically,
~N⁻²·², so the refined reference needs many modes. Same PDE / initial
condition / output protocol, generated with the independent Fourier-spectral
ETDRK4 solver (`verify_reference_solutions.py` at the repository root,
Kassam–Trefethen, float64) at **N = 32704 = 64×511 modes, dt = 5e-6**:

- `allen_cahn_refined.mat` — the file `utils.get_dataset` loads, a drop-in
  replacement for `allen_cahn.mat`: identical keys (`t` 1×201, `x` 1×512,
  `usol` 201×512) and identical grid (the 512 points are an exact subset, no
  interpolation). md5 `c6e04bf9e24d0b2c9090913defbc2da4`.
- `allen_cahn_refined_hires.mat` — full resolution (`usol` 201×32705, 53 MB);
  not shipped, regenerate it with the script below.

**Certified accuracy: rel. L2 = 2.7e-9** on the evaluation grid against a
65408-mode run (2.3e-9 on the full 32704 grid); Richardson extrapolation with
the observed order puts the file's true error near 7e-10. Convergence on the
evaluation grid: 511 → 1.17e-5, 16352 → 1.33e-8, 32704 → 2.7e-9 against the
65408-mode run (`gen_allen_cahn_refined.log`). The time step is irrelevant at
this level (dt = 1e-5 vs 5e-6 differ by 5e-12 at 16352 modes).

Regenerate (from this folder): `python gen_allen_cahn_refined.py` (~1 h,
single core; `--reuse <dir>` takes stored solves written by
`python solve_allen_cahn_N.py <N> <dt> <dir>/ac_N<N>_dt<dt>.npz`). Extra
`.mat` keys `gen_modes`, `gen_dt`, `method` record provenance.

Note: relative L2 errors of runs evaluated against `allen_cahn_refined.mat`
differ from values computed against `allen_cahn.mat` by up to ~1.5e-5 (the
original reference's own error, with the duplicated x = ±1 column counted as
the evaluator does). The benchmark network receives the initial condition as
512 grid values and cannot resolve the corner layer, so scores near 1e-5 are
expected against the refined reference.
