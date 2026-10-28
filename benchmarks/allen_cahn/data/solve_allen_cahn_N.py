"""Solve the Allen-Cahn benchmark at N Fourier modes / time step dt with the
independent ETDRK4 solver and store the 201 output slices.

    python solve_allen_cahn_N.py <N> <dt> <out.npz>
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))   # repo root: verify_reference_solutions.py
import verify_reference_solutions as V

N, dt, out = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
N_OUT = 200
_, u0, L, nl = V.make_problem("allen_cahn", N)
n_steps = round(1.0 / dt)
assert n_steps % N_OUT == 0, "dt must divide 0.005"
t0 = time.time()
print(f"N={N} dt={dt} steps={n_steps} ...", flush=True)
u = V.solve_1d(u0, L, nl, dt, n_steps, n_steps // N_OUT)
np.savez(out, t=np.linspace(0, 1, N_OUT + 1), usol=u, N=N, dt=dt)
print(f"N={N} dt={dt}: done in {time.time() - t0:.0f} s -> {out}  "
      f"(u range {u.min():.4f}..{u.max():.4f}, finite={np.isfinite(u).all()})", flush=True)
