"""
Lephad MMC engine.

Topology:  one tau leg decays leptonically (vis = e/mu, two invisible neutrinos
with non-zero combined invariant mass m_inv_lep), the other decays hadronically
(vis = pi/rho, single invisible neutrino with m_inv_had ~= 0).

Strategy:
  * Hadronic side is fully constrained by the m_tau and m_miss_had = 0 condition
    -> reuse the algebraic compute_neutrino_momenta() with m_miss_had_grid = [0].
  * Leptonic side has an undetermined m_inv_lep -> scan a 1-D grid in
    m_inv_lep in [0, m_tau - eps] and pick the grid point with the highest
    leptonic dR PDF likelihood.

The hadronic side does not need its own dR PDF because it is exactly determined
by the kinematic constraints once m_inv_lep is fixed. Only the leptonic-side
(electron or muon) dR PDF from mmc/mmc_lep_lep_parameters.h5 is consulted.
"""
import numpy as np
import vector

from mmc.MMC_util import mixture_pdf, get_ptau_bin_id


class MMCLepHad:
    def __init__(
            self,
            lep_hist_array: dict,
            sqrt_s: float = 91.2,
            m_inv_grid_points: int = 30,
    ):
        self.M_tau = 1.777
        self.lep_hist_array = lep_hist_array
        self.sqrt_s = sqrt_s
        # Grid spans [0, m_tau - vis_lep_mass]; safe upper bound just below m_tau
        self.m_inv_grid_points = m_inv_grid_points
        self.E_tau = sqrt_s / 2.0

    def _P_dR(self, dR: np.ndarray, p_tau: np.ndarray) -> np.ndarray:
        """Look up the leptonic-side dR PDF at (dR, p_tau)."""
        bin_idx = get_ptau_bin_id(p_tau)
        max_idx = len(self.lep_hist_array["w"]) - 1
        bin_idx = np.clip(bin_idx, 0, max_idx)
        w = self.lep_hist_array["w"][bin_idx]
        mu = self.lep_hist_array["mu"][bin_idx]
        sigma = self.lep_hist_array["sigma"][bin_idx]
        h = self.lep_hist_array["h"][bin_idx]
        A = self.lep_hist_array["A"][bin_idx]
        B = self.lep_hist_array["B"][bin_idx]
        return mixture_pdf(dR, w, mu, sigma, h, A, B)

    def calculation(self, vis_had, vis_lep):
        """
        Reconstruct (nu_had, nu_lep) for each event.

        Returns
        -------
        nu_had : vector.array, shape (N,)
        nu_lep : vector.array, shape (N,)
        likelihood : np.ndarray, shape (N,)  (best-grid-point lep dR PDF value)

        Notes
        -----
        compute_neutrino_momenta is called with vis1 = vis_had, vis2 = vis_lep,
        m_miss1_grid = 0 (single column), m_miss2_grid = scanned (G columns).
        """
        # Local import to avoid an import cycle (NeutrinoReconstructionProcessor
        # itself imports mmc.MMC).
        from processor.NeutrinoReconstructionProcessor import compute_neutrino_momenta

        n_events = len(vis_had)
        # m_inv grid for the leptonic side.
        # Upper limit is m_tau minus a safety margin so the algebraic equations remain physical.
        m_max = self.M_tau - 0.05
        m_grid = np.linspace(0.0, m_max, self.m_inv_grid_points)
        # compute_neutrino_momenta requires m_miss1_grid and m_miss2_grid to have
        # the same column count, so broadcast m_miss_had to G columns of zeros.
        m_miss_had_grid = np.zeros((n_events, self.m_inv_grid_points))
        m_miss_lep_grid = np.broadcast_to(m_grid, (n_events, self.m_inv_grid_points)).copy()

        nu_had_grid, nu_lep_grid, flag_grid = compute_neutrino_momenta(
            vis1_p4=vis_had,
            vis2_p4=vis_lep,
            m_miss1_grid=m_miss_had_grid,
            m_miss2_grid=m_miss_lep_grid,
        )

        # Both grids are now (n_events, m_inv_grid_points). For the had side
        # all columns share m_miss_had_grid = 0, but the algebraic solution still
        # depends on m_miss_lep, so keep the full grid.
        # Compute reco tau on the lep side and dR(tau_lep, vis_lep) per grid point.
        vis_lep_px = np.asarray(vis_lep.px)[:, None]
        vis_lep_py = np.asarray(vis_lep.py)[:, None]
        vis_lep_pz = np.asarray(vis_lep.pz)[:, None]
        vis_lep_E = np.asarray(vis_lep.E)[:, None]

        tau_lep_px = vis_lep_px + np.asarray(nu_lep_grid.px)
        tau_lep_py = vis_lep_py + np.asarray(nu_lep_grid.py)
        tau_lep_pz = vis_lep_pz + np.asarray(nu_lep_grid.pz)
        tau_lep_E = vis_lep_E + np.asarray(nu_lep_grid.E)
        tau_lep = vector.array({
            "px": tau_lep_px, "py": tau_lep_py, "pz": tau_lep_pz, "E": tau_lep_E,
        })

        # dR(vis_lep, nu_lep) -- broadcast vis to grid shape via vector.array
        vis_lep_grid = vector.array({
            "px": np.broadcast_to(vis_lep_px, tau_lep_px.shape).copy(),
            "py": np.broadcast_to(vis_lep_py, tau_lep_py.shape).copy(),
            "pz": np.broadcast_to(vis_lep_pz, tau_lep_pz.shape).copy(),
            "E":  np.broadcast_to(vis_lep_E, tau_lep_E.shape).copy(),
        })
        dR_lep = vis_lep_grid.deltaR(nu_lep_grid)
        p_tau_lep = tau_lep.p

        L = self._P_dR(np.asarray(dR_lep), np.asarray(p_tau_lep))
        # Mask out unphysical grid points (no valid solution -> NaN momenta).
        valid = (flag_grid > 0) & np.isfinite(L)
        L = np.where(valid, L, 0.0)

        # Pick the grid point with the highest likelihood for each event.
        best = np.argmax(L, axis=1)
        any_valid = np.any(valid, axis=1)

        idx = best[:, None]
        nu_had = vector.array({
            "px": np.take_along_axis(np.asarray(nu_had_grid.px), idx, axis=1).squeeze(1),
            "py": np.take_along_axis(np.asarray(nu_had_grid.py), idx, axis=1).squeeze(1),
            "pz": np.take_along_axis(np.asarray(nu_had_grid.pz), idx, axis=1).squeeze(1),
            "E":  np.take_along_axis(np.asarray(nu_had_grid.E),  idx, axis=1).squeeze(1),
        })
        nu_lep = vector.array({
            "px": np.take_along_axis(np.asarray(nu_lep_grid.px), idx, axis=1).squeeze(1),
            "py": np.take_along_axis(np.asarray(nu_lep_grid.py), idx, axis=1).squeeze(1),
            "pz": np.take_along_axis(np.asarray(nu_lep_grid.pz), idx, axis=1).squeeze(1),
            "E":  np.take_along_axis(np.asarray(nu_lep_grid.E),  idx, axis=1).squeeze(1),
        })

        likelihood = np.where(any_valid, np.take_along_axis(L, idx, axis=1).squeeze(1), 0.0)

        # Zero out events with no valid grid point (caller's flag_valid check).
        zero_mask = ~any_valid
        if np.any(zero_mask):
            for arr in (nu_had.px, nu_had.py, nu_had.pz, nu_had.E,
                        nu_lep.px, nu_lep.py, nu_lep.pz, nu_lep.E):
                arr[zero_mask] = np.nan

        return nu_had, nu_lep, likelihood
