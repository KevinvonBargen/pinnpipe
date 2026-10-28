"""Independent verification of the spectral reference solutions used in
pinnpipe/benchmarks (allen_cahn, burgers, kdv, ks_chaotic, ginzburg_landau).

The kdv, ks_chaotic and ginzburg_landau .mat files were generated with
Chebfun's spin/spin2 (Fourier spectral in space, ETDRK4 in time); the burgers
and allen_cahn references (burgers_refined.mat: 2044 modes, dt 2.5e-6;
allen_cahn_refined.mat: 32704 modes, dt 5e-6) were generated with this
script's solver by data/gen_*_refined.py. This script re-solves each PDE with
an independent NumPy implementation of ETDRK4 (Kassam & Trefethen, 2005) and
reports:

  1. repro : rel. L2 distance between the shipped .mat and this solver at the
             SAME resolution/timestep  -> reproducibility of the file
  2. acc   : rel. L2 distance between the shipped .mat and a refined run
             (2x space, 2x time)       -> estimated accuracy of the reference
  3. conv  : rel. L2 distance between the refined and doubly-refined runs
             -> confirms the refined run is converged (trust anchor for 2.);
             for the two refined files only with --full (4x-mode solves)

Run:  python verify_reference_solutions.py [--fast] [--full]
      (allen_cahn alone takes ~1 h: repro at 32704 and acc at 65408 modes)
"""
import sys
import os
import numpy as np
import scipy.io as sio

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")


def etdrk4_coeffs(L, dt, M=32):
    """ETDRK4 scalar coefficients via contour integral (Kassam-Trefethen)."""
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None].astype(complex) + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1))
    return E, E2, Q, f1, f2, f3


def etdrk4_coeffs_complex(L, dt, M=64):
    """Same but for complex L (Ginzburg-Landau): contour around each dt*L."""
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)
    r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[..., None] + r
    Q = dt * np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=-1)
    f1 = dt * np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=-1)
    f2 = dt * np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=-1)
    f3 = dt * np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=-1)
    return E, E2, Q, f1, f2, f3


def solve_1d(u0, L, nonlin_hat, dt, n_steps, save_every):
    """ETDRK4 for u_t = L u + N(u), periodic, real field. Returns (n_saves+1, N)."""
    E, E2, Q, f1, f2, f3 = etdrk4_coeffs(L, dt)
    v = np.fft.fft(u0)
    out = [u0.copy()]
    for n in range(1, n_steps + 1):
        Nv = nonlin_hat(v)
        a = E2 * v + Q * Nv
        Na = nonlin_hat(a)
        b = E2 * v + Q * Na
        Nb = nonlin_hat(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlin_hat(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        if n % save_every == 0:
            out.append(np.real(np.fft.ifft(v)))
    return np.array(out)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ----------------------------------------------------------------------------
# 1D problem definitions (mirroring the gen_*.m scripts)
# ----------------------------------------------------------------------------
def make_problem(name, N):
    """Return (x, u0, L, nonlin_hat) on an N-point periodic grid."""
    if name in ("burgers", "allen_cahn", "kdv"):
        dom_len, x0 = 2.0, -1.0
    else:  # ks_chaotic
        dom_len, x0 = 2.0 * np.pi, 0.0
    x = x0 + dom_len * np.arange(N) / N
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0 / N) / dom_len
    ik = 1j * k

    if name == "burgers":
        nu = 0.01 / np.pi
        L = -nu * k**2
        u0 = -np.sin(np.pi * x)

        def nonlin_hat(v):
            u = np.real(np.fft.ifft(v))
            return -0.5 * ik * np.fft.fft(u * u)

    elif name == "allen_cahn":
        L = 5.0 - 1e-4 * k**2
        u0 = x**2 * np.cos(np.pi * x)

        def nonlin_hat(v):
            u = np.real(np.fft.ifft(v))
            return np.fft.fft(-5.0 * u**3)

    elif name == "kdv":
        L = -(0.022**2) * ik**3  # purely imaginary (dispersive)
        u0 = np.cos(np.pi * x)

        def nonlin_hat(v):
            u = np.real(np.fft.ifft(v))
            return -0.5 * ik * np.fft.fft(u * u)

    elif name == "ks_chaotic":
        L = (100.0 / 16.0**2) * k**2 - (100.0 / 16.0**4) * k**4
        u0 = np.cos(x) * (1.0 + np.sin(x))

        def nonlin_hat(v):
            u = np.real(np.fft.ifft(v))
            return -(100.0 / 16.0) * 0.5 * ik * np.fft.fft(u * u)

    else:
        raise ValueError(name)
    return x, u0, L, nonlin_hat


def solve_1d_kdv_complexL(u0, L, nonlin_hat, dt, n_steps, save_every):
    """KdV has purely imaginary L -> use complex coefficient variant."""
    E, E2, Q, f1, f2, f3 = etdrk4_coeffs_complex(np.asarray(L, complex), dt)
    v = np.fft.fft(u0)
    out = [u0.copy()]
    for n in range(1, n_steps + 1):
        Nv = nonlin_hat(v)
        a = E2 * v + Q * Nv
        Na = nonlin_hat(a)
        b = E2 * v + Q * Na
        Nb = nonlin_hat(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlin_hat(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        if n % save_every == 0:
            out.append(np.real(np.fft.ifft(v)))
    return np.array(out)


def run_1d(name, matfile, fast=False, full=False):
    d = sio.loadmat(matfile)
    usol = d["usol"]                      # (n_out, 512) incl. duplicated endpoint
    ref = usol[:, :511]                   # periodic grid values
    n_out = usol.shape[0] - 1             # intervals
    T = float(d["t"].ravel()[-1])

    # generating resolution/timestep: the refined files (gen_*_refined.py) record
    # it as gen_modes/gen_dt; the Chebfun gen_*.m files used 511 modes, dt 1e-5
    N0 = int(np.ravel(d["gen_modes"])[0]) if "gen_modes" in d else 511
    dt0 = float(np.ravel(d["gen_dt"])[0]) if "gen_dt" in d else 1e-5
    m = N0 // 511                         # the 511-point grid is every m-th point
    assert N0 == m * 511, N0
    solver = solve_1d_kdv_complexL if name == "kdv" else solve_1d

    res = {}
    # 1) reproduction at identical resolution/timestep
    _, u0, L, nl = make_problem(name, N0)
    n_steps = round(T / dt0)
    mine = solver(u0, L, nl, dt0, n_steps, n_steps // n_out)
    res["repro"] = rel_l2(mine[:, ::m], ref)

    # 2) refined run: 2x modes, 2x smaller dt (grid contains the coarse grid)
    _, u0f, Lf, nlf = make_problem(name, 2 * N0)
    fine = solver(u0f, Lf, nlf, dt0 / 2, 2 * n_steps, 2 * n_steps // n_out)
    res["acc"] = rel_l2(ref, fine[:, ::2 * m])

    # 3) doubly refined: convergence confirmation. For the refined files this
    #    is a 4x-mode solve (hours for Allen-Cahn); their gen_*_refined.py
    #    certification runs already provide it, so it needs --full.
    if not fast and (N0 == 511 or full):
        _, u0f2, Lf2, nlf2 = make_problem(name, 4 * N0)
        fine2 = solver(u0f2, Lf2, nlf2, dt0 / 4, 4 * n_steps, 4 * n_steps // n_out)
        res["conv"] = rel_l2(fine, fine2[:, ::2])
    return res


# ----------------------------------------------------------------------------
# 2D Ginzburg-Landau (complex field)
# ----------------------------------------------------------------------------
def solve_gl(N, dt, n_steps, save_every, x_target):
    """Solve GL on N-point periodic grid; save snapshots Fourier-evaluated at
    the tensor grid x_target (matching the .mat's linspace sampling)."""
    kk = 10.0
    eps = kk / (50.0 * 50.0)
    x = -1.0 + 2.0 * np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = (1j * X * 10.0 + Y * 10.0) * np.exp(-0.01 * (X**2 * 2500.0 + Y**2 * 2500.0))
    k1 = np.pi * np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(k1, k1, indexing="ij")
    L = -eps * (KX**2 + KY**2)
    E, E2, Q, f1, f2, f3 = etdrk4_coeffs(L.ravel(), dt)
    E, E2, Q, f1, f2, f3 = (a.reshape(N, N) for a in (E, E2, Q, f1, f2, f3))
    # evaluation matrix  P[a, j] = exp(i k_j (x_a + 1)) / N   (grid starts at -1)
    P = np.exp(1j * np.outer(x_target + 1.0, k1)) / N

    def nonlin_hat(v):
        u = np.fft.ifft2(v)
        return np.fft.fft2(kk * u - kk * (1.0 + 1.5j) * u * np.abs(u) ** 2)

    def sample(v):
        return P @ v @ P.T

    v = np.fft.fft2(u)
    out = [sample(v)]
    for n in range(1, n_steps + 1):
        Nv = nonlin_hat(v)
        a = E2 * v + Q * Nv
        Na = nonlin_hat(a)
        b = E2 * v + Q * Na
        Nb = nonlin_hat(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlin_hat(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        if n % save_every == 0:
            out.append(sample(v))
    return np.array(out)


def run_gl(fast=False):
    d = sio.loadmat(f"{BASE}/ginzburg_landau/data/ginzburg_landau_square.mat")
    # MATLAB meshgrid in gen script stores snapshots as (t, y, x); transpose
    # to (t, x, y) to match this solver's orientation (verified vs t=0 init).
    ref = (d["usol"] + 1j * d["vsol"]).transpose(0, 2, 1)
    xt = np.linspace(-1.0, 1.0, 200)
    n_out = ref.shape[0] - 1
    dt0 = 1.0 / 200.0 / 5.0               # = 1/1000, as in gen script
    T = 1.0
    n_steps = round(T / dt0)

    res = {}
    mine = solve_gl(200, dt0, n_steps, n_steps // n_out, xt)
    res["repro"] = rel_l2(mine, ref)

    fine = solve_gl(400, dt0 / 2.0, 2 * n_steps, 2 * n_steps // n_out, xt)
    res["acc"] = rel_l2(ref, fine)

    if not fast:
        fine2 = solve_gl(800, dt0 / 4.0, 4 * n_steps, 4 * n_steps // n_out, xt)
        res["conv"] = rel_l2(fine, fine2)
    return res


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    full = "--full" in sys.argv
    jobs = {
        "burgers": lambda: run_1d("burgers", f"{BASE}/burgers/data/burgers_refined.mat", fast, full),
        "allen_cahn": lambda: run_1d("allen_cahn", f"{BASE}/allen_cahn/data/allen_cahn_refined.mat", fast, full),
        "kdv": lambda: run_1d("kdv", f"{BASE}/kdv/data/kdv.mat", fast),
        "ks_chaotic": lambda: run_1d("ks_chaotic", f"{BASE}/ks_chaotic/data/ks_chaotic.mat", fast),
        "ginzburg_landau": run_gl,
    }
    for name, job in jobs.items():
        try:
            r = job()
            print(f"{name:16s} " + "  ".join(f"{k}={v:.3e}" for k, v in r.items()), flush=True)
        except Exception as e:
            print(f"{name:16s} FAILED: {e!r}", flush=True)
