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
            this_dir = os.path.dirname(os.path.abspath(__file__))
            default_pdf_path = os.path.join(this_dir, 'mmc_lep_lep_parameters.h5')
            pdf_path = self.config.get('mmc_pdf_path', default_pdf_path)
            print(f"[MMC] Loading Master PDF parameters from {pdf_path}...")
            self.pdfs['electron'] = load_h5_group(pdf_path, 'electron')
            self.pdfs['muon'] = load_h5_group(pdf_path, 'muon')

    def calculate(self, vis_a_p4, vis_b_p4, region_name, events):
        """
        Executes the MMC calculation dynamically and returns reconstructed neutrino objects.
        """
        num_events = len(events)
        
        # Initialize output arrays
        nu_a_px, nu_a_py, nu_a_pz, nu_a_E = np.zeros(num_events), np.zeros(num_events), np.zeros(num_events), np.zeros(num_events)
        nu_b_px, nu_b_py, nu_b_pz, nu_b_E = np.zeros(num_events), np.zeros(num_events), np.zeros(num_events), np.zeros(num_events)
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
        elif region_name in {'pi_el', 'el_pi', 'pi_mu', 'mu_pi',
                             'rho_el', 'el_rho', 'rho_mu', 'mu_rho'}:
            # Lephad regions: hadronic side is determined by the algebraic
            # m_miss = 0 condition (handled inside MMCLepHad via
            # compute_neutrino_momenta); leptonic side scans m_inv_lep against
            # the electron or muon dR PDF. Delegate and return early.
            return self._calculate_lephad(vis_a_p4, vis_b_p4, region_name)
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
            
            # Notice how we pass vis_a_p4[mask] directly without recreating it!
            n1, n2, likelihood = parallel_calculation(
                engine, vis_a_p4[mask], vis_b_p4[mask], num_workers=self.mmc_workers
            )
            
            # Map the results back to the original array sizing using the mask
            nu_a_px[mask], nu_a_py[mask], nu_a_pz[mask], nu_a_E[mask] = n1.px, n1.py, n1.pz, n1.E
            nu_b_px[mask], nu_b_py[mask], nu_b_pz[mask], nu_b_E[mask] = n2.px, n2.py, n2.pz, n2.E
            mmc_likelihood[mask] = likelihood

        # Zip final contiguous arrays
        reco_mis_positivep4 = vector.zip({
            "px": np.ascontiguousarray(nu_a_px), 
            "py": np.ascontiguousarray(nu_a_py), 
            "pz": np.ascontiguousarray(nu_a_pz), 
            "E":  np.ascontiguousarray(nu_a_E)
        })
        reco_mis_negativep4 = vector.zip({
            "px": np.ascontiguousarray(nu_b_px), 
            "py": np.ascontiguousarray(nu_b_py), 
            "pz": np.ascontiguousarray(nu_b_pz), 
            "E":  np.ascontiguousarray(nu_b_E)
        })

        mmc_likelihood = np.ascontiguousarray(mmc_likelihood)
        flags_valid = np.ascontiguousarray(np.where((mmc_likelihood > 0) & np.isfinite(mmc_likelihood), 1, 0))

        return reco_mis_positivep4, reco_mis_negativep4, flags_valid, mmc_likelihood

    def _calculate_lephad(self, vis_a_p4, vis_b_p4, region_name):
        """Lephad reconstruction. region_name encodes the (hadron, lepton)
        ordering and the lepton flavor: e.g. 'pi_el' = hadron(pi) on a-side,
        lepton(electron) on b-side; 'el_rho' = lepton(electron) on a-side,
        hadron(rho) on b-side.
        """
        from mmc.mmc_lephad import MMCLepHad

        had_a, lep_a = region_name.split('_')
        had_on_a = had_a in ('pi', 'rho')
        lep_flavor = lep_a if had_on_a else had_a
        lep_pdf = self.pdfs['electron'] if lep_flavor == 'el' else self.pdfs['muon']
        engine = MMCLepHad(lep_hist_array=lep_pdf, sqrt_s=cme)

        if had_on_a:
            vis_had, vis_lep = vis_a_p4, vis_b_p4
        else:
            vis_had, vis_lep = vis_b_p4, vis_a_p4

        nu_had, nu_lep, likelihood = parallel_calculation(
            engine, vis_had, vis_lep, num_workers=self.mmc_workers
        )

        # Map back to (a, b) ordering. Caller expects (mis_positive=a, mis_negative=b).
        if had_on_a:
            nu_a, nu_b = nu_had, nu_lep
        else:
            nu_a, nu_b = nu_lep, nu_had

        reco_mis_positivep4 = vector.zip({
            "px": np.ascontiguousarray(nu_a.px),
            "py": np.ascontiguousarray(nu_a.py),
            "pz": np.ascontiguousarray(nu_a.pz),
            "E":  np.ascontiguousarray(nu_a.E),
        })
        reco_mis_negativep4 = vector.zip({
            "px": np.ascontiguousarray(nu_b.px),
            "py": np.ascontiguousarray(nu_b.py),
            "pz": np.ascontiguousarray(nu_b.pz),
            "E":  np.ascontiguousarray(nu_b.E),
        })
        likelihood = np.ascontiguousarray(likelihood)
        flags_valid = np.ascontiguousarray(
            np.where((likelihood > 0) & np.isfinite(likelihood), 1, 0)
        )
        return reco_mis_positivep4, reco_mis_negativep4, flags_valid, likelihood