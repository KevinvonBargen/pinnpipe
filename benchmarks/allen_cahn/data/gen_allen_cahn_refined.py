"""Generate the refined Allen-Cahn reference (target: rel L2
accuracy <= 1e-8 on the benchmark's 201 x 512 evaluation grid).

Same PDE / IC / output protocol as gen_allen_cahn.m (u_t = 1e-4 u_xx + 5u - 5u^3,
u0 = x^2 cos(pi x), x in [-1,1] periodic, t in [0,1], 201 outputs), solved with
the independent Fourier-spectral ETDRK4 solver of
verify_reference_solutions.py (repo root) (Kassam-Trefethen, float64) at
N = 32704 = 64 x 511 modes and dt = 5e-6, so the 512-point evaluation grid is
an exact subset (drop-in replacement for the original allen_cahn.mat, no interpolation).

Why so many modes: u0 is only C^0 across the periodic boundary
(u0'(-1) = +2, u0'(1) = -2), so the Fourier coefficients decay like 1/k^2 and
the reference converges algebraically, ~N^-2, not spectrally; the error is
confined to |x| > 0.9, t < 0.1 where diffusion (1e-4) has not yet smoothed the
corner (511 modes: 1.2e-5, 4088: 2.1e-7, 8176: 4.3e-8, 16352: ~1e-8). The time
step is irrelevant at this level (dt 1e-5 vs 5e-6 differ by 5e-12 at N=16352).
Certification: a 65408-mode run; its distance to the saved file bounds the
file's error (Richardson with the observed order gives the estimate).

    python gen_allen_cahn_refined.py                       # ~1 h, single core
    python gen_allen_cahn_refined.py --reuse <dir-with-ac_N*_dt*.npz>
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))   # repo root: verify_reference_solutions.py
import verify_reference_solutions as V

OUT = os.path.dirname(os.path.abspath(__file__))
N_OUT = 200                              # intervals -> 201 outputs
METHOD = "Fourier spectral + ETDRK4 (verify_reference_solutions.py at the repo root)"
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))


def solve(N, dt, reuse):
    if reuse:
        for f in glob.glob(os.path.join(reuse, f"ac_N{N}_dt*.npz")):
            z = np.load(f)
            if abs(float(z["dt"]) - dt) < 1e-12:
                print(f"  reusing {f}", flush=True)
                return z["usol"]
    _, u0, L, nl = V.make_problem("allen_cahn", N)
    n_steps = round(1.0 / dt)
    assert n_steps % N_OUT == 0
    t0 = time.time()
    print(f"  solving N={N}, dt={dt} ({n_steps} steps) ...", flush=True)
    u = V.solve_1d(u0, L, nl, dt, n_steps, n_steps // N_OUT)
    print(f"  done in {time.time() - t0:.0f} s", flush=True)
    return u


def on_eval_grid(u):                     # (201, N) -> (201, 511), N = m*511
    m = u.shape[1] // 511
    assert u.shape[1] == m * 511, u.shape
    return u[:, ::m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32704)
    ap.add_argument("--dt", type=float, default=5e-6)
    ap.add_argument("--cert_N", type=int, default=65408)
    ap.add_argument("--cert_dt", type=float, default=5e-6)
    ap.add_argument("--reuse", default=None, help="dir with ac_N<N>_dt<dt>.npz solves")
    args = ap.parse_args()
    N, SUB = args.N, args.N // 511
    assert N == SUB * 511 and args.cert_N % 511 == 0

    print(f"production run N={N}, dt={args.dt}", flush=True)
    u_fine = solve(N, args.dt, args.reuse)                      # (201, N)
    t = np.linspace(0.0, 1.0, N_OUT + 1)
    prov = {"gen_modes": N, "gen_dt": args.dt, "method": METHOD}

    x_hi = np.linspace(-1.0, 1.0, N + 1)
    sio.savemat(f"{OUT}/allen_cahn_refined_hires.mat",
                {"t": t, "x": x_hi, "usol": np.concatenate([u_fine, u_fine[:, :1]], 1), **prov})
    x_lo = np.linspace(-1.0, 1.0, 512)
    u_lo = u_fine[:, ::SUB]
    sio.savemat(f"{OUT}/allen_cahn_refined.mat",
                {"t": t, "x": x_lo, "usol": np.concatenate([u_lo, u_lo[:, :1]], 1), **prov})
    print("saved allen_cahn_refined.mat (201x512) and allen_cahn_refined_hires.mat "
          f"(201x{N + 1})", flush=True)

    # sanity: distance to the original 511-mode allen_cahn.mat (expect ~1.2e-5);
    # that file is no longer shipped, so the check runs only if it is present
    old = None
    if os.path.exists(f"{OUT}/allen_cahn.mat"):
        old = sio.loadmat(f"{OUT}/allen_cahn.mat")["usol"][:, :511]
        print(f"rel L2 (original allen_cahn.mat vs refined) = {rel(old, u_lo):.3e}  "
              f"<- the OLD file's error", flush=True)

    print(f"certification run N={args.cert_N}, dt={args.cert_dt}", flush=True)
    u_cert = solve(args.cert_N, args.cert_dt, args.reuse)
    np.savez_compressed(f"{OUT}/allen_cahn_refined_cert{args.cert_N}.npz", t=t,
                        x=np.linspace(-1.0, 1.0, args.cert_N + 1)[:-1], usol=u_cert,
                        gen_modes=args.cert_N, gen_dt=args.cert_dt)
    m = args.cert_N // N
    err_full = rel(u_fine, u_cert[:, ::m]) if args.cert_N == m * N else float("nan")
    err_eval = rel(u_lo, on_eval_grid(u_cert))
    print(f"rel L2 (saved {N}-mode file vs {args.cert_N}-mode run): "
          f"{err_full:.3e} on the {N}-grid, {err_eval:.3e} on the 512 evaluation grid",
          flush=True)
    d = u_lo - on_eval_grid(u_cert)
    e2 = (d ** 2).sum(1); tot = e2.sum()
    xg = x_lo[:511]
    print("  share of squared error: t<0.025 {:.0%}, 0.025-0.1 {:.0%}, 0.1-0.3 {:.0%}, "
          ">0.3 {:.0%}; |x|>0.9: {:.0%}; max|diff| {:.2e}".format(
              e2[:5].sum() / tot, e2[5:20].sum() / tot, e2[20:60].sum() / tot,
              e2[60:].sum() / tot, (d[:, np.abs(xg) > 0.9] ** 2).sum() / tot,
              np.abs(d).max()), flush=True)

    # convergence table from every available solve (plus the original file, if present)
    seq = {511: old} if old is not None else {}
    if args.reuse:
        for f in glob.glob(os.path.join(args.reuse, "ac_N*_dt*.npz")):
            z = np.load(f)
            seq.setdefault(int(z["N"]), z["usol"])
    seq[N] = u_fine
    ref = on_eval_grid(u_cert)
    Ns = sorted(n for n in seq if n < args.cert_N)
    errs = {n: rel(on_eval_grid(seq[n]), ref) for n in Ns}
    print(f"convergence vs the {args.cert_N}-mode run (512 evaluation grid):", flush=True)
    for n in Ns:
        print(f"  N={n:6d}: {errs[n]:.3e}", flush=True)
    if len(Ns) >= 2:
        a, b = Ns[-2], Ns[-1]
        p = np.log(errs[a] / errs[b]) / np.log(b / a)
        est = errs[b] / ((args.cert_N / b) ** p - 1)
        print(f"  observed order {a}->{b}: p = {p:.2f}; Richardson estimate of the "
              f"saved file's error: {est:.1e}, of the certification run: {est / (args.cert_N / b) ** p:.1e}",
              flush=True)
    print("=> accuracy of the saved refined reference is at or below the certification distance.",
          flush=True)


if __name__ == "__main__":
    main()
