# adapted from https://github.com/UW-EPE-ML/Quantum_Informaton_Analysis/blob/main/mmc/MMC_func.py
import sys
import multiprocessing as mp
import numpy as np
import vector
from scipy.stats import norm, landau
import matplotlib.pyplot as plt
from MMC_util import mixture_pdf, get_ptau_bin_edges, get_ptau_bin_id


class MissingMassCalculator:
    def __init__(
            self, hist_array: dict,
            phi_grid_points: int = 30,
            phi_search_range: float = 0.3,
            sigma_ET: float = 1.0,
    ):

        self.phi_grid_points = phi_grid_points
        self.phi_search_range = phi_search_range * np.pi
        self.M_tau = 1.777  # GeV
        self.hist_array = hist_array
        self.sigma_ET = sigma_ET
        self.likelihood_scale = 1000

    def validate_function(self, save_path=None):
        # Delta R values
        deltaR_values = np.linspace(0, 1.0, 100)

        # Define number of subplots
        num_bins = len(self.hist_array['tau1'])
        num_cols = 5
        num_rows = (num_bins + num_cols - 1) // num_cols

        # Create the figure and subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 18))
        axes = axes.flatten()

        # Plot each pT bin
        for i, (pt, params) in enumerate(self.hist_array['tau1'].items()):
            mu = params["mu"]
            sigma = params["sigma"]
            A = params["A"]
            B = params["B"]
            w = params["w"]
            h = params["h"]

            y_values = mixture_pdf(deltaR_values, w, mu, sigma, h, A, B)

            axes[i].plot(deltaR_values, y_values, label=f'pT {pt}')
            # axes[i].set_title(f'pT {pt}')
            # axes[i].set_xlabel(r'$\Delta R$')
            # axes[i].set_ylabel('PDF')
            axes[i].legend()

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


    def P_deltaR(self, delta_R: np.ndarray, p_tau: np.ndarray, is_tau_1: bool):
        # TODO: Implement vectorized version of P_deltaR using get_ptau_bin_id and direct indexing into hist_array
        # TODO: Then compute the mixture PDF for all events and candidates at once using broadcasting

        # # Select tau dictionary
        # params = self.hist_array["tau1"] if is_tau_1 else self.hist_array["tau2"]

        # num_events, num_candidates = p_tau.shape
        # # Convert pT bins to a NumPy array for fast indexing
        # pt_bins = np.array([
        #     [sorted(map(int, params.keys()))] * num_candidates for _ in range(num_events)
        # ])

        # pt_indices = np.abs(pt_bins - p_tau[:, :, None]).argmin(axis=2)

        # pt_values = np.take_along_axis(pt_bins, pt_indices[:, :, None], axis=2).squeeze(-1)  # (100, 900)

        # # Retrieve parameters for each selected pT bin
        # param_array = np.vectorize(lambda pt: params[str(pt)], otypes=[dict])(pt_values)

        # # Extract parameters as NumPy arrays
        # mu = np.array([[p["mu"] for p in row] for row in param_array])
        # sigma = np.array([[p["sigma"] for p in row] for row in param_array])
        # A = np.array([[p["A"] for p in row] for row in param_array])
        # B = np.array([[p["B"] for p in row] for row in param_array])
        # w = np.array([[p["w"] for p in row] for row in param_array])
        # h = np.array([[p["h"] for p in row] for row in param_array])

        # # Compute vectorized probability
        # return mixture_pdf(delta_R, w, mu, sigma, h, A, B)

    def event_likelihood(self, delta_R1: np.ndarray, p_tau1: np.ndarray, delta_R2: np.ndarray, p_tau2: np.ndarray):
        """Vectorized version of event_likelihood using NumPy broadcasting."""

        # Create a mask for NaN values
        nan_mask = np.isnan(delta_R1) | np.isnan(delta_R2) | np.isnan(p_tau1) | np.isnan(p_tau2)

        # Compute P_deltaR values using broadcasting
        P1 = self.P_deltaR(delta_R1, p_tau1, True) * self.likelihood_scale
        P2 = self.P_deltaR(delta_R2, p_tau2, False) * self.likelihood_scale

        # Compute final likelihood while handling invalid values
        likelihoods = np.where(
            (P1 * P2 > 0) & (~nan_mask),  # Only compute when valid
            P1 * P2,
            np.inf  # Assign infinity for invalid cases
        )

        return likelihoods

    def compute_neutrino_momenta(
            self,
            phi_mis1: np.ndarray, phi_mis2: np.ndarray,
            Etx: np.ndarray, Ety: np.ndarray,
            vis_1_mass: np.ndarray, vis_2_mass: np.ndarray,
            vis_1_pz: np.ndarray, vis_2_pz: np.ndarray,
            vis_1_phi: np.ndarray, vis_2_phi: np.ndarray,
            vis_1_p: np.ndarray, vis_2_p: np.ndarray,
            vis_1_theta: np.ndarray, vis_2_theta: np.ndarray,
            theta_mis2: np.ndarray = None,
    ):
        pass

    def calculation(
            self,
            Etx: np.ndarray, Ety: np.ndarray,
            vis_1: vector.MomentumNumpy4D,
            vis_2: vector.MomentumNumpy4D,
            eventID: np.ndarray,
    ):
        pass