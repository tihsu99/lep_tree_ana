import numpy as np
import pandas as pd
import uproot as ur
import matplotlib.pyplot as plt
import logging
import vector
import glob
import os
import awkward as ak
import copy
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events, get_all_p4_from_ak_events, cme

log = logging.getLogger(__name__)

def filter_leplep_channel(events: ak.Array, filter_log_dict: dict):
    filter_log_dict['leplep channel initial'] = filter_log_dict.get('leplep channel initial', 0) + len(events)
    recpart_pdgid = events['Part_pdgId']
    recpart_abspdgid = abs(recpart_pdgid)
    recpart_charge = events['Part_charge']

    pass_filter = ak.ones_like(events['evtNumber'], dtype=bool)

    flag_is_mu = (recpart_abspdgid == 6)
    flag_is_el = (recpart_abspdgid == 2)
    flag_is_lepton = flag_is_mu | flag_is_el

    # only contains two leptons in opposite charge
    pass_filter = (ak.sum(flag_is_lepton, axis=1) == 2) & pass_filter
    filter_log_dict['2 leptons'] = filter_log_dict.get('2 leptons', 0) + ak.sum(pass_filter)

    charge_ary_of_leptons = recpart_charge[flag_is_lepton]
    flag_opposite_charge = (ak.sum(charge_ary_of_leptons, axis=1) == 0)
    pass_filter = flag_opposite_charge & pass_filter
    filter_log_dict['opposite charge'] = filter_log_dict.get('opposite charge', 0) + ak.sum(pass_filter)

    events_leplep = events[pass_filter & flag_opposite_charge]

    return events_leplep, filter_log_dict
    


def filter_pipi_channel(events: ak.Array, filter_log_dict: dict):
    filter_log_dict['pipi channel initial'] = filter_log_dict.get('pipi channel initial', 0) + len(events)
    recpart_pdgid = events['Part_pdgId']
    recpart_abspdgid = abs(recpart_pdgid)
    recpart_charge = events['Part_charge']

    pass_filter = ak.ones_like(events['evtNumber'], dtype=bool)

    # no other hadronic particles in reco particles
    pass_filter = (ak.sum(
        (recpart_abspdgid == 47) |  # pi0
        (recpart_abspdgid == 42) |  # kaon+
        (recpart_abspdgid == 61) |  # KS
        (recpart_abspdgid == 62) |  # KL
        (recpart_abspdgid == 65) |  # proton
        (recpart_abspdgid == 66) |  # neutron
        (recpart_abspdgid == 81) |  # lambda
        (recpart_abspdgid == 0),   # undefined
      axis=1) == 0) & pass_filter
    filter_key = 'no other hadrons'
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter)

    # no electrons or muons in reco particles
    pass_filter = (ak.sum(
        (recpart_abspdgid == 2) |  # electron
        (recpart_abspdgid == 6),  # muon
      axis=1) == 0) & pass_filter
    filter_key = 'no e/mu'    
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter)

    # no less than two pions (regardless of charge for now) in reco particles
    pass_filter = (ak.sum((recpart_abspdgid == 41), axis=1) >= 2) & pass_filter
    # exactly one pi+ and one pi-
    charge_ary_of_pions = recpart_charge[recpart_abspdgid == 41]
    flag_pi_plus_and_pi_minus = (ak.sum(charge_ary_of_pions == 1, axis=1) == 1 ) & (ak.sum(charge_ary_of_pions == -1, axis=1) == 1)
    pass_filter = flag_pi_plus_and_pi_minus & pass_filter
    filter_key = '1 pi+ and 1 pi-'
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter)

    events_pipi = events[pass_filter & flag_pi_plus_and_pi_minus]
    # define some new variables for pipi events
    p4_piplus = get_p4_from_ak_events(events_pipi, (abs(events_pipi['Part_pdgId']) == 41) & (events_pipi['Part_charge'] == 1))
    p4_piminus = get_p4_from_ak_events(events_pipi, (abs(events_pipi['Part_pdgId']) == 41) & (events_pipi['Part_charge'] == -1))
    P_rad = ((p4_piplus.px**2 + p4_piplus.py**2 + p4_piplus.pz**2) + (p4_piminus.px**2 + p4_piminus.py**2 + p4_piminus.pz**2))**0.5
    events_pipi['P_rad'] = P_rad
    # filtered_events['pipi'] = events_pipi


    ##########################
    # define SR
    ##########################
    reco_abs_pdgId = np.abs(events_pipi['Part_pdgId'])
    reco_charge = events_pipi['Part_charge']
    flag_pion = (reco_abs_pdgId == 41)
    flag_piplus = flag_pion & (reco_charge == 1)
    flag_piminus = flag_pion & (reco_charge == -1)
    p4_piplus = get_p4_from_ak_events(events_pipi, flag_piplus)
    p4_piminus = get_p4_from_ak_events(events_pipi, flag_piminus)
    p4_dipion = p4_piplus + p4_piminus

    pass_filter_sr = ak.ones_like(events_pipi['evtNumber'], dtype=bool)
    # only two reconstructed particles
    pass_filter_sr = (ak.num(events_pipi['Part_pdgId']) == 2) & pass_filter_sr
    filter_key = 'nParticles == 2'
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter_sr)

    # angle between dipions
    angle_between_pions = p4_piplus.deltaangle(p4_piminus)
    pass_filter_sr = (angle_between_pions > 2.99) & (angle_between_pions < 3.1) & pass_filter_sr
    filter_key = 'Pions angle between 2.99 and 3.1'
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter_sr)

    # dipion invariant mass
    dipion_mass = p4_dipion.mass
    pass_filter_sr = (dipion_mass > 10) & (dipion_mass < 85) & pass_filter_sr
    filter_key = 'Dipion mass between 10 and 85 GeV'
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter_sr)

    # total energy
    total_energy = ak.sum(events_pipi['Part_fourMomentum_fCoordinates_fT'], axis=-1)
    pass_filter_sr = (total_energy < 80) & (total_energy > 20) & pass_filter_sr
    filter_key = 'Total energy between 20 and 80 GeV'
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter_sr)

    # missing momentum 
    missing_px = -ak.sum(events_pipi['Part_fourMomentum_fCoordinates_fX'], axis=-1)
    missing_py = -ak.sum(events_pipi['Part_fourMomentum_fCoordinates_fY'], axis=-1)
    missing_pz = - ak.sum(events_pipi['Part_fourMomentum_fCoordinates_fZ'], axis=-1)
    missing_p = np.sqrt(missing_px**2 + missing_py**2 + missing_pz**2)
    pass_filter_sr = (missing_p < 40) & pass_filter_sr
    filter_key = 'Missing p < 40 GeV'
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter_sr)

    # P_rad
    pass_filter_sr = (events_pipi['P_rad'] < cme/2) & pass_filter_sr
    filter_key = 'P_rad < cme/2'
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter_sr)

    # log10_1mthrust
    log10_1mthrust = np.log10(1 - events_pipi['thrust_Mag'])
    pass_filter_sr = (log10_1mthrust < -2.5) & pass_filter_sr
    filter_key = 'log10(1 - thrust) < -2.5'
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter_sr)

    events_sr = events_pipi[pass_filter_sr]

    return events_sr, filter_log_dict


def filter_event(events: ak.Array, filter_log_dict: dict):
    filtered_events_dict = {
        'raw': events,
    }
    events_copy = copy.deepcopy(events)
    filtered_events, filter_log_dict = filter_pipi_channel(events_copy, filter_log_dict)
    filtered_events_dict['pipi'] = filtered_events

    filtered_events, filter_log_dict = filter_leplep_channel(events_copy, filter_log_dict)
    filtered_events_dict['leplep'] = filtered_events


    return filtered_events_dict, filter_log_dict


class DataLoader:
    def __init__(self, config, output_dir):
        self.config = config
        # load all config into member variables
        for key, value in config.items():
            setattr(self, key, value)
        self.norm_factor = config.get("norm_factor", 1.0)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.tree_name = self.config.get("tree_name", "t")
        self.input_files = self.config.get("input_files", [])
        self.region_of_interest = self.config.get("region_of_interest", "pipi")
        self.is_data = self.config.get("is_data", False)

        if not self.input_files:
            raise ValueError("Input files must be specified.")
        elif isinstance(self.input_files, str):
            self.input_files = glob.glob(self.input_files)
        else:
            all_files = []
            for pattern in self.input_files:
                all_files.extend(glob.glob(pattern))
            # sort files for consistency
            all_files = sorted(all_files)
            self.input_files = all_files

        self.data = {}
        self.filter_results = {
            'initial_total_num_events': 0,
        }

        _data_loaded = False
        if len(glob.glob(self.output_dir + "/filtered___*.parquet")) > 0:
            log.info("Loading existing filtered data.")
            file = self.output_dir + "/filtered___" + self.region_of_interest + ".parquet"
            if os.path.exists(file):
                self.data[self.region_of_interest] = ak.from_parquet(file)
                self.initial_total_num_events = self.data[self.region_of_interest]['initial_total_num_events'][0]
                _data_loaded = True
            else:
                log.warning(f"Filtered data file for region {self.region_of_interest} does not exist. Re-loading and filtering data from input files.")

        if not _data_loaded:
            self.load_data()
            self.save_data()
            keys = list(self.data.keys())
            for key in keys:
                if not (key == self.region_of_interest):
                    del self.data[key]
            _data_loaded = True

        log.info(f"DataLoader initialization complete. Loaded {len(all_files)} files.")

    

    def load_data(self) -> pd.DataFrame:
        # Identify branches to load
        f = ur.open(self.input_files[0])
        tree = f[self.tree_name]

        common_evt_branches = ["Event_evtNumber", "Event_totalChargedEnergy", "Event_totalEMEnergy", "Event_totalHadronicEnergy", "thrust_Mag", "thrust_x", "thrust_y", "thrust_z", "nGoodPart", 
        # "event_category"
        ]
        gen_part_branches = ["pdgId", "status", "vector_fCoordinates_fX", "vector_fCoordinates_fY", "vector_fCoordinates_fZ", "vector_fCoordinates_fT"]
        gen_part_branches = [f"GenPart_{b}" for b in gen_part_branches]
        
        part_branches = ["charge", "pdgId", "fourMomentum_fCoordinates_fX", "fourMomentum_fCoordinates_fY", "fourMomentum_fCoordinates_fZ", "fourMomentum_fCoordinates_fT", "isGood"]
        part_branches = [f'Part_{b}' for b in part_branches]

        particleID_branches = [
            # "Elid_partIdx", "Elid_tag", "Elid_gammaConversion",
            # "Muid_partIdx", "Muid_tag",
            # "Haid_pionRich", "Haidn_pionTag", "Haidr_pionTag", "Haide_pionTag", "Haidc_pionTag"
        ]

        branches_to_load = common_evt_branches + part_branches + particleID_branches
        if not self.is_data:
            branches_to_load += gen_part_branches

        # Load data from all files
        initial_total_num_events = 0
        for file in self.input_files:
            log.info(f"Loading data from file: {file}")
            try:
                f = ur.open(file)
                tree = f[self.tree_name]
                # load all events as awkward array 
                events = tree.arrays(branches_to_load, library="ak")

                # adjust event index to be unique across files
                # the original Event_evtNumber starts from 1 for each file
                if len(events) == 0:
                    continue
                events['evtNumber'] = events['Event_evtNumber'] + initial_total_num_events
                events['initial_total_num_events'] = len(events)
                initial_total_num_events += len(events)

                # select Part_xxx via isGood flag
                part_abscosth = abs(events['Part_fourMomentum_fCoordinates_fZ']) / ((events['Part_fourMomentum_fCoordinates_fX'])**2 + (events['Part_fourMomentum_fCoordinates_fY'])**2 + (events['Part_fourMomentum_fCoordinates_fZ'])**2)**0.5
                flag_not_0pdgid = (events['Part_pdgId'] != 0)
                events['Part_isGood'] = (events['Part_isGood']==1) & (part_abscosth < 0.732) # & flag_not_0pdgid
                for part_branch in part_branches + particleID_branches:
                    if part_branch != 'Part_isGood':
                        events[part_branch] = events[part_branch][events['Part_isGood']] 

                if not self.is_data:
                    # get truth info of tau pair and tau neutrinos
                    dict_part_pdg = {
                        'tau': 15,
                        'anti_tau': -15,
                        'nu_tau': 16,
                        'anti_nu_tau': -16,
                    }
                    for key, pdgid in dict_part_pdg.items():
                        flag = (events['GenPart_pdgId'] == pdgid)
                        events[f'truth_{key}_px'] = ak.firsts(events['GenPart_vector_fCoordinates_fX'][flag][...,::-1])
                        events[f'truth_{key}_py'] = ak.firsts(events['GenPart_vector_fCoordinates_fY'][flag][...,::-1])
                        events[f'truth_{key}_pz'] = ak.firsts(events['GenPart_vector_fCoordinates_fZ'][flag][...,::-1])
                        events[f'truth_{key}_E'] = ak.firsts(events['GenPart_vector_fCoordinates_fT'][flag][...,::-1])


                # filter events
                self.filter_results['initial_total_num_events'] += len(events)
                events_pass_filter, self.filter_results = filter_event(events, self.filter_results)

                # # save filtered events
                # log.info(f"Saving filtered data to {path_filtered_single_file_prefix}.")
                # for key, evt in events_pass_filter.items():
                #     path_filtered_single_file = path_filtered_single_file_prefix + f"___{key}.parquet"
                #     ak.to_parquet(evt, path_filtered_single_file)

                # record filtered events into self.data
                for key, evt in events_pass_filter.items():
                    self.data.setdefault(key, []).append(evt)

            except Exception as e:
                log.error(f"Error reading file {file} or tree {self.tree_name}: {e}")
                continue

        # Concatenate data from all files
        for key in self.data:
            self.data[key] = ak.concatenate(self.data[key], axis=0)
            self.initial_total_num_events = initial_total_num_events
            self.data[key]['initial_total_num_events'] = initial_total_num_events

        # Log filter results
        if self.filter_results['initial_total_num_events'] > 0:
            with open(self.output_dir + "/cutflow.txt", "w") as f:
                f.write(f"{'Cut':<40} {'Events':<20} {'Efficiency':<20} {'Relative Efficiency':<20}\n")
                previous_count = self.filter_results['initial_total_num_events']
                for key, value in self.filter_results.items():
                    log.info(f"Filter result - {key}: {value}. Filter efficiency: {value / self.filter_results['initial_total_num_events']:.4f}")
                    efficiency = value / self.filter_results['initial_total_num_events']
                    relative_efficiency = value / previous_count if previous_count > 0 else 1.0
                    f.write(f"{key:<40} {value:<20} {efficiency:<20.4f} {relative_efficiency:<20.4f}\n")
                    previous_count = value

            # plot filter results
            cutflow_labels = list(self.filter_results.keys())
            cutflow_values = [self.filter_results[key] for key in cutflow_labels]
            fig, ax = plt.subplots(dpi=300, figsize=(8,8))
            p = ax.bar(cutflow_labels, cutflow_values)
            ax.bar_label(p, labels=[f"{v}" for v in cutflow_values], padding=3)
            ax.set_ylabel('Number of Events')
            ax.set_title('Event Cutflow')
            ax.set_yscale('log')
            # rotate x, fontsize to small
            plt.xticks(rotation=45, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(self.output_dir + "/cutflow.pdf")

            cutflow_normalized = [v / self.filter_results['initial_total_num_events'] for v in cutflow_values]
            fig, ax = plt.subplots(dpi=300, figsize=(8,8))
            p = ax.bar(cutflow_labels, cutflow_normalized)
            ax.bar_label(p, labels=[f"{v:.4f}" for v in cutflow_normalized], padding=3)
            ax.set_ylabel('Efficiency')
            ax.set_title('Event Cutflow Efficiency')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(self.output_dir + "/cutflow_efficiency.pdf")

            # cutflow_relative = [cutflow_values[i] / cutflow_values[i-1] if i > 0 else 1.0 for i in range(len(cutflow_values))]
            cutflow_relative = [1.0]
            tmp_cutflow_label = ['initial_totoal_num_events']
            for i in range(1, len(cutflow_values)):
                rel = cutflow_values[i] / cutflow_values[i-1] 
                label = cutflow_labels[i]
                if rel>1:
                    # if eff>1 then calculate ratio relative to initial num
                    rel = cutflow_values[i] / cutflow_values[0]
                    label = f"{label}/initialNoE"
                cutflow_relative.append(rel)
                tmp_cutflow_label.append(label)

            fig, ax = plt.subplots(dpi=300, figsize=(8,8))
            p = ax.bar(tmp_cutflow_label, cutflow_relative)
            ax.bar_label(p, labels=[f"{v:.4f}" for v in cutflow_relative], padding=3)
            ax.set_ylabel('Relative Efficiency')
            ax.set_title('Event Cutflow Relative Efficiency')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(self.output_dir + "/cutflow_relative_efficiency.pdf")

        return self.data

    
    def save_data(self):
        output_file_prefix = self.output_dir + "/filtered"
        for key, evt in self.data.items():
            output_file = output_file_prefix + f"___{key}.parquet"
            log.info(f"Saving data for channel {key} to {output_file}.")
            ak.to_parquet(evt, output_file)

            log.info(f"Data saved to {output_file}.")

        # structured_output_file = self.output_dir + f"/filtered_{self.region_of_interest}_structured.npy"
        # np.save(structured_output_file, self.structured_data)
        # # To load the structured data, use: np.load(structured_output_file, allow_pickle=True).item()
        # log.info(f"Structured data saved to {structured_output_file}.")


    def run(self, dl):
        pass


    def finalize(self):
        log.info("DataLoader finalization complete.")


if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG, format = ">>> [%(levelname)s]: %(message)s")
    config = {
        "tree_name": "t",
        "input_files": "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251029_Ztautau_singlePionDecay/simana_job_17827112_*_ttree.root",
        # "input_files": "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251031_Ztautau_singlePi/simana_job_*_ttree.root",
        # "input_files": "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251031_Ztautau_singlePi/simana_job_17841601_*_ttree.root",

        # "input_files": "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251031_Ztautau_singlePi/merged/simana_ttree.root",
    }

    loader = DataLoader(config)

    # import numpy as np
    # import pandas as pd
    # # test reading output file
    # df = pd.read_hdf("filtered_data.h5")
    # print(df)
    # ary = np.load("filtered_data_structured.npy", allow_pickle=True).item()
    # print(ary)
