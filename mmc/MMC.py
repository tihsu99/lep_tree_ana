import numpy as np
import vector
import awkward as ak
import h5py
import os
from mmc.mmc_diunknown import MMCDiUnknown 
from mmc.MMC_util import parallel_calculation
from utils.common_functions import cme

def load_h5_group(filepath, group_name):
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF file missing: {filepath}")
    with h5py.File(filepath, "r") as f:
        if group_name in f:
            grp = f[group_name]
            return {k: grp[k][:] for k in grp.keys()}
        raise KeyError(f"Could not find '{group_name}' in {filepath}")

class MMC:
    def __init__(self, config):
        self.config = config
        self.mmc_regions = self.config.get('mmc_regions', [])
        self.mmc_workers = self.config.get('mmc_workers', 4)
        
        self.pdfs = {}
        if len(self.mmc_regions) > 0:
            pdf_path = self.config.get('mmc_pdf_path', 'mmc_lep_lep_parameters_master.h5')
            print(f"[MMC] Loading Master PDF parameters from {pdf_path}...")
            self.pdfs['electron'] = load_h5_group(pdf_path, 'electron')
            self.pdfs['muon'] = load_h5_group(pdf_path, 'muon')

    def calculate(self, vis1_p4, vis2_p4, region_name, events):
        """
        Executes the MMC calculation dynamically and returns reconstructed neutrino objects.
        """
        num_events = len(events)
        
        # Initialize output arrays
        nu1_px, nu1_py, nu1_pz, nu1_E = np.zeros(num_events), np.zeros(num_events), np.zeros(num_events), np.zeros(num_events)
        nu2_px, nu2_py, nu2_pz, nu2_E = np.zeros(num_events), np.zeros(num_events), np.zeros(num_events), np.zeros(num_events)
        mmc_likelihood = np.zeros(num_events)

        masks = []
        pdf_list = []

        # Setup configurations based on region
        if region_name == 'ee':
            masks = [np.ones(num_events, dtype=bool)]
            pdf_list = [(self.pdfs['electron'], self.pdfs['electron'])]
        elif region_name == 'mumu':
            masks = [np.ones(num_events, dtype=bool)]
            pdf_list = [(self.pdfs['muon'], self.pdfs['muon'])]
        elif region_name == 'emu':
            is_a_e = ak.to_numpy(events['lead_a_is_electron'])
            is_a_mu = ak.to_numpy(events['lead_a_is_muon'])
            masks = [is_a_e, is_a_mu]
            pdf_list = [
                (self.pdfs['electron'], self.pdfs['muon']),
                (self.pdfs['muon'], self.pdfs['electron'])
            ]
        else:
            raise ValueError(f"Unknown region {region_name}")

        phi_grid_pts = 50
        theta_grid_pts = 50

        # Harmonized execution loop
        for mask, pdfs in zip(masks, pdf_list):
            if not np.any(mask):
                continue
                
            engine = MMCDiUnknown(
                hist_array_1=pdfs[0], hist_array_2=pdfs[1], sqrt_s=cme,
                phi_grid_points=phi_grid_pts, theta_grid_points=theta_grid_pts,
                phi_search_range=0.3, theta_search_range=0.3
            )
            
            # Notice how we pass vis1_p4[mask] directly without recreating it!
            n1, n2, likelihood = parallel_calculation(
                engine, vis1_p4[mask], vis2_p4[mask], num_workers=self.mmc_workers
            )
            
            # Map the results back to the original array sizing using the mask
            nu1_px[mask], nu1_py[mask], nu1_pz[mask], nu1_E[mask] = n1.px, n1.py, n1.pz, n1.E
            nu2_px[mask], nu2_py[mask], nu2_pz[mask], nu2_E[mask] = n2.px, n2.py, n2.pz, n2.E
            mmc_likelihood[mask] = likelihood

        # Zip final contiguous arrays
        reco_mis_negativep4 = vector.zip({
            "px": np.ascontiguousarray(nu1_px), 
            "py": np.ascontiguousarray(nu1_py), 
            "pz": np.ascontiguousarray(nu1_pz), 
            "E":  np.ascontiguousarray(nu1_E)
        })
        reco_mis_positivep4 = vector.zip({
            "px": np.ascontiguousarray(nu2_px), 
            "py": np.ascontiguousarray(nu2_py), 
            "pz": np.ascontiguousarray(nu2_pz), 
            "E":  np.ascontiguousarray(nu2_E)
        })

        mmc_likelihood = np.ascontiguousarray(mmc_likelihood)
        flags_valid = np.ascontiguousarray(np.where((mmc_likelihood > 0) & np.isfinite(mmc_likelihood), 1, 0))
        
        return reco_mis_negativep4, reco_mis_positivep4, flags_valid, mmc_likelihood