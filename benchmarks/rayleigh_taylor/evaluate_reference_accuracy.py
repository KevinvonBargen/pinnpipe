"""Independent accuracy evaluation of rayleigh_taylor_high_Ra.npy.

Method: take dataset snapshot k as initial condition, integrate the stated
Boussinesq system

    u_t + u.grad(u) = -grad(p) + sqrt(Pr/Ra) lap(u) + T e_y
    div(u) = 0
    T_t + u.grad(T) = 1/sqrt(Pr Ra) lap(T)

on [0,1]x[0,2] (periodic in x, u=v=T=0 at y=0,2; Pr=0.71, Ra=1e6) with an
independent solver (2nd-order MAC staggered finite volumes, explicit SSP-RK2,
exact FFTx/DCTy pressure projection), and compare against the dataset at the
next snapshot times (dt_out = 0.25). Discrepancy after one/two intervals,
measured at two solver resolutions, bounds the dataset's local accuracy.
"""
import numpy as np
from scipy.fft import fft, ifft, dct, idct
from scipy.interpolate import RectBivariateSpline

import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rayleigh_taylor_high_Ra.npy")
Pr, Ra = 0.71, 1.0e6
NU = np.sqrt(Pr / Ra)
KAPPA = 1.0 / np.sqrt(Pr * Ra)
LX, LY = 1.0, 2.0

d = np.load(DATA, allow_pickle=True).item()
cx = np.unique(d["coords"][:, 0])
cy = np.unique(d["coords"][:, 1])
NXD, NYD = len(cx), len(cy)  # 100, 200
T_data = d["temperature"].reshape(40, NYD, NXD).transpose(0, 2, 1)  # (t, x, y)
u_data = d["velocity"][..., 0].reshape(40, NYD, NXD).transpose(0, 2, 1)
v_data = d["velocity"][..., 1].reshape(40, NYD, NXD).transpose(0, 2, 1)
t_data = d["t"]


def spline_from_data(F, wall_zero=True):
    """Periodic-in-x spline of a (NXD, NYD) data field; rows for the walls
    (y=0, y=LY, value 0) are appended so near-wall evaluation interpolates."""
    xp = np.concatenate([cx - LX, cx, cx + LX])
    Fp = np.vstack([F, F, F])
    if wall_zero:
        yp = np.concatenate([[0.0], cy, [LY]])
        Fp = np.hstack([np.zeros((len(xp), 1)), Fp, np.zeros((len(xp), 1))])
    else:
        yp = cy
    return RectBivariateSpline(xp, yp, Fp, kx=3, ky=3)


class Solver:
    def __init__(self, Nx, Ny):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = LX / Nx, LY / Ny
        self.xf = self.dx * np.arange(Nx)              # u faces
        self.xc = self.dx * (np.arange(Nx) + 0.5)      # centers
        self.yf = self.dy * np.arange(Ny + 1)          # v faces
        self.yc = self.dy * (np.arange(Ny) + 0.5)      # centers
        # Poisson eigenvalues (2nd-order modified wavenumbers)
        kx = np.arange(Nx)
        self.lam_x = (2.0 * np.cos(2.0 * np.pi * kx / Nx) - 2.0) / self.dx**2
        j = np.arange(Ny)
        self.lam_y = (2.0 * np.cos(np.pi * j / Ny) - 2.0) / self.dy**2
        lam = self.lam_x[:, None] + self.lam_y[None, :]
        lam[0, 0] = 1.0  # zero mode: arbitrary constant
        self.inv_lam = 1.0 / lam
        self.inv_lam[0, 0] = 0.0

    # ---- operators ------------------------------------------------------
    def poisson(self, rhs):
        """Solve lap(phi) = rhs, periodic x, Neumann y, cell-centered."""
        h = fft(rhs, axis=0)
        h = dct(h, type=2, axis=1)
        h *= self.inv_lam
        h = idct(h, type=2, axis=1)
        return np.real(ifft(h, axis=0))

    def divergence(self, u, v):
        return ((np.roll(u, -1, axis=0) - u) / self.dx
                + (v[:, 1:] - v[:, :-1]) / self.dy)

    def project(self, u, v, dt):
        phi = self.poisson(self.divergence(u, v) / dt)
        u = u - dt * (phi - np.roll(phi, 1, axis=0)) / self.dx
        v = v.copy()
        v[:, 1:-1] -= dt * (phi[:, 1:] - phi[:, :-1]) / self.dy
        return u, v

    def rhs(self, u, v, T):
        dx, dy = self.dx, self.dy
        # ghost rows (walls): tangential no-slip, T=0 Dirichlet
        ug = np.concatenate([-u[:, :1], u, -u[:, -1:]], axis=1)      # (Nx, Ny+2)
        Tg = np.concatenate([-T[:, :1], T, -T[:, -1:]], axis=1)
        # corner values (x-face, y-face): (Nx, Ny+1)
        u_cor = 0.5 * (ug[:, :-1] + ug[:, 1:])
        v_cor = 0.5 * (v + np.roll(v, 1, axis=0))
        # --- u momentum (at x-faces, cell-centered in y) ---
        ue = 0.5 * (u + np.roll(u, -1, axis=0))                      # at centers
        duu = (ue**2 - np.roll(ue, 1, axis=0)**2) / dx
        Fuv = u_cor * v_cor                                          # (Nx, Ny+1)
        duv = (Fuv[:, 1:] - Fuv[:, :-1]) / dy
        lap_u = ((np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dx**2
                 + (ug[:, 2:] - 2 * u + ug[:, :-2]) / dy**2)
        Ru = -duu - duv + NU * lap_u
        # --- v momentum (at y-faces; walls stay 0) ---
        Guv = u_cor * v_cor
        dvu = (np.roll(Guv, -1, axis=0) - Guv) / dx                  # (Nx, Ny+1)
        vc = 0.5 * (v[:, :-1] + v[:, 1:])                            # at centers
        dvv_i = (vc[:, 1:]**2 - vc[:, :-1]**2) / dy                  # interior faces
        lap_v_i = ((np.roll(v, -1, axis=0) - 2 * v + np.roll(v, 1, axis=0))[:, 1:-1] / dx**2
                   + (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]) / dy**2)
        buoy_i = 0.5 * (T[:, 1:] + T[:, :-1])
        Rv = np.zeros_like(v)
        Rv[:, 1:-1] = -dvu[:, 1:-1] - dvv_i + NU * lap_v_i + buoy_i
        # --- temperature (cell centers) ---
        Tx_face = 0.5 * (T + np.roll(T, 1, axis=0))                  # at x-faces
        adv_x = (np.roll(u * Tx_face, -1, axis=0) - u * Tx_face) / dx
        Ty_face = np.zeros_like(v)
        Ty_face[:, 1:-1] = 0.5 * (T[:, 1:] + T[:, :-1])
        Fy = v * Ty_face
        adv_y = (Fy[:, 1:] - Fy[:, :-1]) / dy
        lap_T = ((np.roll(T, -1, axis=0) - 2 * T + np.roll(T, 1, axis=0)) / dx**2
                 + (Tg[:, 2:] - 2 * T + Tg[:, :-2]) / dy**2)
        RT = -adv_x - adv_y + KAPPA * lap_T
        return Ru, Rv, RT

    def step(self, u, v, T, dt):
        """SSP-RK2 with projection after each stage."""
        Ru, Rv, RT = self.rhs(u, v, T)
        u1, v1 = self.project(u + dt * Ru, v + dt * Rv, dt)
        T1 = T + dt * RT
        Ru1, Rv1, RT1 = self.rhs(u1, v1, T1)
        u2, v2 = self.project(0.5 * (u + u1 + dt * Ru1), 0.5 * (v + v1 + dt * Rv1), dt)
        T2 = 0.5 * (T + T1 + dt * RT1)
        return u2, v2, T2

    # ---- data transfer --------------------------------------------------
    def init_from_snapshot(self, k):
        su = spline_from_data(u_data[k])
        sv = spline_from_data(v_data[k])
        sT = spline_from_data(T_data[k])
        u = su(self.xf, self.yc)
        v = np.zeros((self.Nx, self.Ny + 1))
        v[:, 1:-1] = sv(self.xc, self.yf[1:-1])
        T = sT(self.xc, self.yc)
        u, v = self.project(u, v, 1.0)  # make divergence-free on this grid
        return u, v, T

    def sample_at_data_points(self, u, v, T):
        """Interpolate solver fields to the dataset's (cx, cy) tensor grid."""
        def sp(F, x_nodes, y_nodes, wall_zero_pad):
            xp = np.concatenate([x_nodes - LX, x_nodes, x_nodes + LX])
            Fp = np.vstack([F, F, F])
            if wall_zero_pad:
                yp = np.concatenate([[0.0], y_nodes, [LY]])
                Fp = np.hstack([np.zeros((len(xp), 1)), Fp, np.zeros((len(xp), 1))])
            else:
                yp = y_nodes
            return RectBivariateSpline(xp, yp, Fp, kx=3, ky=3)(cx, cy)
        ui = sp(u, self.xf, self.yc, True)
        vi = sp(v, self.xc, self.yf, False)
        Ti = sp(T, self.xc, self.yc, True)
        return ui, vi, Ti


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def evaluate(k, n_intervals, Nx, Ny, cfl=0.3):
    s = Solver(Nx, Ny)
    u, v, T = s.init_from_snapshot(k)
    umax = max(np.abs(u).max(), np.abs(v).max(), 0.5) * 1.6
    h = min(s.dx, s.dy)
    dt_stab = min(cfl * h / umax, 0.2 * h**2 / max(NU, KAPPA))
    n_sub = int(np.ceil(0.25 / dt_stab))
    dt = 0.25 / n_sub
    out = []
    for m in range(1, n_intervals + 1):
        for _ in range(n_sub):
            u, v, T = s.step(u, v, T, dt)
        ui, vi, Ti = s.sample_at_data_points(u, v, T)
        kk = k + m
        out.append({
            "t": t_data[kk],
            "relL2_T": rel_l2(Ti, T_data[kk]),
            "relL2_u": rel_l2(ui, u_data[kk]),
            "relL2_v": rel_l2(vi, v_data[kk]),
            "persistence_T": rel_l2(T_data[k], T_data[kk]),
        })
    return out, dt


if __name__ == "__main__":
    import sys
    fast = "--fast" in sys.argv
    resolutions = [(192, 384)] if fast else [(192, 384), (384, 768)]
    snapshots = [5, 15, 30]        # benchmark window starts at index 5 (t->0)
    n_intervals = 2
    for (Nx, Ny) in resolutions:
        print(f"--- solver resolution {Nx}x{Ny} ---", flush=True)
        for k in snapshots:
            out, dt = evaluate(k, n_intervals, Nx, Ny)
            for o in out:
                print(f"  IC snap {k} (t={t_data[k]:.2f}) -> t={o['t']:.2f}: "
                      f"relL2(T)={o['relL2_T']:.3e}  relL2(u)={o['relL2_u']:.3e}  "
                      f"relL2(v)={o['relL2_v']:.3e}  [persistence T: {o['persistence_T']:.2e}]  dt={dt:.1e}",
                      flush=True)
