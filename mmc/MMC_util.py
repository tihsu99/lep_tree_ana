# adapted from  https://github.com/UW-EPE-ML/Quantum_Informaton_Analysis/blob/main/mmc/MMC_util.py
import multiprocessing as mp
import numpy as np
import vector
from tqdm import tqdm
from scipy.stats import norm, landau

phi_grid_pts = 50
theta_grid_pts = 50

def get_ptau_bin_edges():
    """Return the edges of p_tau bins."""
    ptau_edges = np.concatenate([
        np.arange(10, 40, 10),
        np.arange(40, 44, 0.5),
        np.arange(44, 45, 0.1),
        np.arange(45, 45.6+0.001, 0.01),
    ])
    return ptau_edges



def get_ptau_bin_id(ptau: np.ndarray) -> np.ndarray:
    """Return the p_tau bin ID for each value in ptau."""
    ptau_edges = get_ptau_bin_edges()
    bin_idx = np.digitize(ptau, bins=ptau_edges) - 1  # Subtract 1 for 0-based indexing
    
    # Safely clip to ensure no index goes out of bounds
    max_idx = len(ptau_edges) - 2
    return np.clip(bin_idx, 0, max_idx)


def gaussian_pdf(x, mu, sigma):
    """Normalized Gaussian PDF."""
    sigma = np.clip(sigma, 1e-6, None)
    return norm.pdf(x, loc=mu, scale=sigma)


def landau_pdf(x, A, B):
    """
    Landau PDF.
    Uses scipy.stats.landau if available.
    Falls back to moyal approximation otherwise.
    """
    B = np.clip(B, 1e-6, None)
    return landau.pdf(x, loc=A, scale=B)


def mixture_pdf(x, w, mu, sigma, h, A, B):
    """
    Model:
        w * Gaussian(mu, sigma) + h * Landau(A, B)

    Here w and h are free amplitudes, exactly as requested.
    """
    return w * gaussian_pdf(x, mu, sigma) + h * landau_pdf(x, A, B)


def parallel_worker(args):
    """Wrapper function to process a chunk of data"""
    mmc, vis_1, vis_2 = args
    return mmc.calculation(vis_1, vis_2)


def parallel_calculation(mmc, vis_1, vis_2, num_workers=None, batch_size=10000):
    """
    Parallel execution of calculation() using multiprocessing.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()  # Use all available CPUs

    if num_workers > batch_size:
        num_workers = batch_size

    num_events = len(vis_1)
    total_batches = (num_events + batch_size - 1) // batch_size  # Calculate total batches
    mini_size = batch_size // num_workers

    # Initialize lists to store final merged results
    nu1_list, nu2_list, likelihoods_list = [], [], []

    # tqdm progress bar
    with tqdm(total=total_batches, desc="Processing Batches", unit="batch") as pbar:
        # Process in batches
        for batch_start in range(0, num_events, batch_size):
            batch_end = min(batch_start + batch_size, num_events)

            # Further split batch into parallel chunks, handling remainder properly
            chunks = []
            for i in range(batch_start, batch_end, mini_size):
                mini_end = min(i + mini_size, batch_end)  # Ensure no missing events
                chunks.append(
                    (
                        mmc, vis_1[i:mini_end], vis_2[i:mini_end]
                    )
                )

            with mp.Pool(num_workers) as pool:
                results = pool.map(parallel_worker, chunks)

            # Unpack results and merge them
            for nu1, nu2, likelihood in results:
                nu1_list.append(nu1)
                nu2_list.append(nu2)
                likelihoods_list.append(likelihood)

            pbar.update(1)  # Update progress bar after processing a batch

    # Concatenate all components using list comprehension
    def concat_attr(attr, nu_list):
        return np.concatenate([getattr(nu, attr) for nu in nu_list])

    nu1 = vector.arr({
        "px": concat_attr("px", nu1_list),
        "py": concat_attr("py", nu1_list),
        "pz": concat_attr("pz", nu1_list),
        "E": concat_attr("E", nu1_list)
    })

    nu2 = vector.arr({
        "px": concat_attr("px", nu2_list),
        "py": concat_attr("py", nu2_list),
        "pz": concat_attr("pz", nu2_list),
        "E": concat_attr("E", nu2_list)
    })

    # Concatenate likelihoods
    likelihoods = np.concatenate(likelihoods_list)

    return nu1, nu2, likelihoods