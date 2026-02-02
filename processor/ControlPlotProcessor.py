import numpy as np
from BaseProcessor import BaseProcessor
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events, get_all_p4_from_ak_events, cme
from utils.plotter import do_control_plot

def make_control_plots_pirho(dl_dict, luminosity, normalize, output_dir, channel_name="pirho"):
    # apply cuts for leplep channel if needed
    for dl_name, dl in dl_dict.items():
        events_pirho = ak.from_parquet("/afs/cern.ch/user/c/clamore/lep_tree_ana/output-pirho/Ztautau/filtered___raw.parquet")
        pass_filter = ak.ones_like(events_pirho['evtNumber'], dtype=bool)
        dl.data[channel_name] = events_pirho[pass_filter]

    ########################################################
    # total energy
    ########################################################
    def get_total_energy(dl):
        events_pirho = ak.from_parquet("/afs/cern.ch/user/c/clamore/lep_tree_ana/output-pirho/Ztautau/filtered___raw.parquet")
        p4_piplus = get_all_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == 211))
        p4_piminus = get_all_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == -211))
        p4_pi0 = get_all_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == 111)) 
        p4_piplus_filled = ak.fill_none(p4_piplus, 0)
        p4_piminus_filled = ak.fill_none(p4_piminus, 0)
        p4_pi0_filled = ak.fill_none(p4_pi0, 0)
        piplus_energy = ak.sum(p4_piplus_filled.energy, axis=-1)
        piminus_energy = ak.sum(p4_piminus_filled.energy, axis=-1)
        pi0_energy = ak.sum(p4_pi0_filled.energy, axis=-1)
        total_energy = piplus_energy + piminus_energy + pi0_energy
        # total_energy = ak.sum(events_pirho['GenPart_vector_fCoordinates_fT'], axis=1)
        return total_energy

    bin_edges = np.linspace(0, 100, 31)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_total_energy,
        bin_edges=bin_edges,
        x_label='Total Energy [GeV]',
        title='Control Plot: Total Energy',
        luminosity=luminosity, normalize=normalize,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_total_energy.png")

    ########################################################
    # p_rad and E_rad
    ########################################################
    # def get_p_rad(dl):
    #     events = dl.data.get(channel_name)
    #     return events['P_rad']
    # bin_edges = np.linspace(0, 40, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_p_rad,
    #     bin_edges=bin_edges,
    #     x_label='P_rad [GeV]',
    #     title='Control Plot: P_rad',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # # ax.vlines(x=cme/2, ymin=0, ymax=ax.get_ylim()[1], colors='b', linestyles='dashed', label='cme/2')
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_p_rad.png")
    # def get_e_rad(dl):
    #     events = dl.data.get(channel_name)
    #     return events['E_rad']
    # bin_edges = np.linspace(0, 100, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_e_rad,
    #     bin_edges=bin_edges,
    #     x_label='E_rad [GeV]',
    #     title='Control Plot: E_rad',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # ax.vlines(x=cme/2, ymin=0, ymax=ax.get_ylim()[1], colors='b', linestyles='dashed', label='cme/2')
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_e_rad.png")

    ########################################################
    # theta of pions
    ########################################################
    # def get_pion_theta(dl):
    #     events = dl.data.get(channel_name)
    #     genpart_pdgId = events['GenPart_pdgId']
    #     genpart_abs_pdgId = np.abs(genpart_pdgId)
    #     flag_pion = (genpart_abs_pdgId == 211) | (genpart_pdgId == 111)
    #     p4_pion = get_sum_p4_from_ak_events(events, flag_pion)
    #     theta = p4_pion.theta * 180 / np.pi  # convert to degrees
    #     return theta

    # bin_edges = np.linspace(0, 180, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_pion_theta,
    #     bin_edges=bin_edges,
    #     x_label='sum Pion Theta [degrees]',
    #     title='Control Plot: Pion Theta',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_sum_pion_theta.png")

    # def get_piplus_theta(dl):
    #     events = dl.data.get(channel_name)
    #     genpart_pdgId = events['GenPart_pdgId']
    #     # genpart_charge = events['GenPart_charge']
    #     genpart_abs_pdgId = np.abs(genpart_pdgId)
    #     flag_piplus = (genpart_pdgId == 211)
    #     p4_piplus = get_p4_from_ak_events(events, flag_piplus)
    #     theta = p4_piplus.theta * 180 / np.pi  # convert to degrees
    #     return theta
    # bin_edges = np.linspace(0, 180, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_piplus_theta,
    #     bin_edges=bin_edges,
    #     x_label='Pion+ Theta [degrees]',
    #     title='Control Plot: Pion+ Theta',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_pion_plus_theta.png")
    # def get_piminus_theta(dl):
    #     events = dl.data.get(channel_name)
    #     genpart_pdgId = events['GenPart_pdgId']
    #     # reco_charge = events['Part_charge']
    #     genpart_abs_pdgId = np.abs(genpart_pdgId)
    #     # flag_pion = (genpart_pdgId == 41)
    #     flag_piminus = (genpart_pdgId == -211)
    #     p4_piminus = get_p4_from_ak_events(events, flag_piminus)
    #     theta = p4_piminus.theta * 180 / np.pi  # convert to degrees
    #     return theta
    # bin_edges = np.linspace(0, 180, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_piminus_theta,
    #     bin_edges=bin_edges,
    #     x_label='Pion- Theta [degrees]',
    #     title='Control Plot: Pion- Theta',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_pion_minus_theta.png")
    # def get_pi0_theta(dl):
    #     events = dl.data.get(channel_name)
    #     genpart_pdgid = events['GenPart_pdgId']
    #     # reco_charge = events['Part_charge']
    #     genpart_abs_pdgid = np.abs(genpart_pdgid)
    #     # flag_pion = (genpart_pdgId == 41)
    #     flag_pi0 = (genpart_pdgid == 111)
    #     p4_pi0 = get_p4_from_ak_events(events, flag_pi0)
    #     theta = p4_pi0.theta * 180 / np.pi  # convert to degrees
    #     return theta
    # bin_edges = np.linspace(0, 180, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_pi0_theta,
    #     bin_edges=bin_edges,
    #     x_label='Pion0 Theta [degrees]',
    #     title='Control Plot: Pion0 Theta',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_pion0_theta.png")


    # def get_piplus_phi(dl):
    #     events = dl.data.get(channel_name)
    #     genpart_pdgid = events['GenPart_pdgId']
    #     genpart_abs_pdgid = np.abs(genpart_pdgid)
    #     flag_piplus = (genpart_pdgid == 211)
    #     p4_piplus = get_p4_from_ak_events(events, flag_piplus)
    #     phi = p4_piplus.phi * 180 / np.pi  # convert to degrees
    #     return phi
    # bin_edges = np.linspace(-180, 180, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_piplus_phi,
    #     bin_edges=bin_edges,
    #     x_label='Pion+ Phi [degrees]',
    #     title='Control Plot: Pion+ Phi',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_pion_plus_phi.png")
    # def get_piminus_phi(dl):
    #     events = dl.data.get(channel_name)
    #     genpart_pdgid = events['GenPart_pdgId']
    #     genpart_abs_pdgid = np.abs(genpart_pdgid)
    #     flag_piminus = (genpart_pdgid == -211)
    #     p4_piminus = get_p4_from_ak_events(events, flag_piminus)
    #     phi = p4_piminus.phi * 180 / np.pi  # convert to degrees
    #     return phi
    # bin_edges = np.linspace(-180, 180, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_piminus_phi,
    #     bin_edges=bin_edges,
    #     x_label='Pion- Phi [degrees]',
    #     title='Control Plot: Pion- Phi',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_pion_minus_phi.png")
    # def get_pi0_phi(dl):
    #     events = dl.data.get(channel_name)
    #     genpart_pdgid = events['GenPart_pdgId']
    #     genpart_abs_pdgid = np.abs(genpart_pdgid)
    #     flag_pi0 = (genpart_pdgid == 111)
    #     p4_pi0 = get_p4_from_ak_events(events, flag_pi0)
    #     phi = p4_pi0.phi * 180 / np.pi  # convert to degrees
    #     return phi
    # bin_edges = np.linspace(-180, 180, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_pi0_phi,
    #     bin_edges=bin_edges,
    #     x_label='Pion0 Phi [degrees]',
    #     title='Control Plot: Pion0 Phi',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_pion0_phi.png")


    # ########################################################
    # # pion pm pt
    # #########################################################
    # def get_pion_plus_pt(dl):
    #     events = dl.data.get(channel_name)
    #     reco_pdgId = events['Part_pdgId']
    #     reco_charge = events['Part_charge']
    #     reco_abs_pdgId = np.abs(reco_pdgId)
    #     flag_pion = (reco_abs_pdgId == 41)
    #     flag_piplus = flag_pion & (reco_charge == 1)
    #     p4_piplus = get_p4_from_ak_events(events, flag_piplus)
    #     return p4_piplus.pt
    # bin_edges = np.linspace(0, 70, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_pion_plus_pt,
    #     bin_edges=bin_edges,
    #     x_label='Transverse Momentum of Pion+ [GeV]',
    #     title='Control Plot: Pion+ Transverse Momentum',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_pion_plus_pt.png")
    # def get_pion_minus_pt(dl):
    #     events = dl.data.get(channel_name)
    #     reco_pdgId = events['Part_pdgId']
    #     reco_charge = events['Part_charge']
    #     reco_abs_pdgId = np.abs(reco_pdgId)
    #     flag_pion = (reco_abs_pdgId == 41)
    #     flag_piminus = flag_pion & (reco_charge == -1)
    #     p4_piminus = get_p4_from_ak_events(events, flag_piminus)
    #     return p4_piminus.pt
    # bin_edges = np.linspace(0, 70, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_pion_minus_pt,
    #     bin_edges=bin_edges,
    #     x_label='Transverse Momentum of Pion- [GeV]',
    #     title='Control Plot: Pion- Transverse Momentum',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_pion_minus_pt.png")
    
    # for charge in ['plus', 'minus', 'neutral']:
    #     def get_pion_pt(dl, charge=charge):
    #         events = dl.data.get(channel_name)
    #         genpart_pdgid = events['GenPart_pdgId']
    #         # genpart_charge = events['GenPart_charge']
    #         genpart_abs_pdgid = np.abs(genpart_pdgid)
    #         # flag_pion = (genpart_abs_pdgid == 211)
    #         if charge == 'plus':
    #             flag = (genpart_pdgid == 211)
    #         elif charge == 'minus':
    #             flag = (genpart_pdgid == -211)
    #         elif charge == 'neutral':
    #             flag = (genpart_pdgid == 111)
    #         p4_pion = get_p4_from_ak_events(events, flag)
    #         return p4_pion.pt
    #     bin_edges = np.linspace(0, 70, 21)
    #     fig, ax, ax_ratio = do_control_plot(
    #         dl_dict,
    #         func_get_variable=lambda dl, charge=charge: get_pion_pt(dl, charge),
    #         bin_edges=bin_edges,
    #         x_label=f'Transverse Momentum of Pion{charge} [GeV]',
    #         title=f'Control Plot: Pion{charge} Transverse Momentum',
    #         luminosity=luminosity, normalize=normalize,
    #     )
    #     plt.tight_layout()
    #     plt.savefig(f"{output_dir}/control_plot_pion_{charge}_pt.png") 

    #     def get_pion_px(dl, charge=charge):
    #         events = dl.data.get(channel_name)
    #         genpart_pdgId = events['GenPart_pdgId']
    #         # reco_charge = events['Part_charge']
    #         genpart_abs_pdgId = np.abs(genpart_pdgId)
    #         # flag_pion = (reco_abs_pdgId == 41)
    #         if charge == 'plus':
    #             flag = (genpart_pdgId == 211)
    #         elif charge == 'minus':
    #             flag = (genpart_pdgId == -211)
    #         else:
    #             charge == 'neutral'
    #             flag = (genpart_pdgId == 111)
    #         p4_pion = get_p4_from_ak_events(events, flag)
    #         return p4_pion.px
    #     bin_edges = np.linspace(-70, 70, 21)
    #     fig, ax, ax_ratio = do_control_plot(
    #         dl_dict,
    #         func_get_variable=lambda dl, charge=charge: get_pion_px(dl, charge),
    #         bin_edges=bin_edges,
    #         x_label=f'Px of Pion{charge} [GeV]',
    #         title=f'Control Plot: Pion{charge} Px',
    #         luminosity=luminosity, normalize=normalize,
    #     )
    #     plt.tight_layout()
    #     plt.savefig(f"{output_dir}/control_plot_pion_{charge}_px.png")
    #     def get_pion_py(dl, charge=charge):
    #         events = dl.data.get(channel_name)
    #         genpart_pdgid = events['GenPart_pdgId']
    #         # reco_charge = events['Part_charge']
    #         genpart_abs_pdgid = np.abs(genpart_pdgid)
    #         # flag_pion = (genpart_abs_pdgId == 41)
    #         if charge == 'plus':
    #             flag = (genpart_pdgid == 211)
    #         elif charge == 'minus':
    #             flag = (genpart_pdgid == -211)
    #         else:
    #             flag = (genpart_pdgid == 111)
    #         p4_pion = get_p4_from_ak_events(events, flag)
    #         return p4_pion.py
    #     bin_edges = np.linspace(-70, 70, 21)
    #     fig, ax, ax_ratio = do_control_plot(
    #         dl_dict,
    #         func_get_variable=lambda dl, charge=charge: get_pion_py(dl, charge),
    #         bin_edges=bin_edges,
    #         x_label=f'Py of Pion{charge} [GeV]',
    #         title=f'Control Plot: Pion{charge} Py',
    #         luminosity=luminosity, normalize=normalize,
    #     )
    #     plt.tight_layout()
    #     plt.savefig(f"{output_dir}/control_plot_pion_{charge}_py.png")
    #     def get_pion_pz(dl, charge=charge):
    #         events = dl.data.get(channel_name)
    #         genpart_pdgId = events['GenPart_pdgId']
    #         # reco_charge = events['Part_charge']
    #         genpart_abs_pdgId = np.abs(genpart_pdgId)
    #         # flag_pion = (genpart_abs_pdgId == 41)
    #         if charge == 'plus':
    #             flag = (genpart_pdgId == 211)
    #         elif charge == 'minus':
    #             flag = (genpart_pdgId == -211)
    #         elif charge == 'neutral':
    #             flag = (genpart_pdgId == 111)
    #         p4_pion = get_p4_from_ak_events(events, flag)
    #         return p4_pion.pz
    #     bin_edges = np.linspace(-100, 100, 21)
    #     fig, ax, ax_ratio = do_control_plot(
    #         dl_dict,
    #         func_get_variable=lambda dl, charge=charge: get_pion_pz(dl, charge),
    #         bin_edges=bin_edges,
    #         x_label=f'Pz of Pion{charge} [GeV]',
    #         title=f'Control Plot: Pion{charge} Pz',
    #         luminosity=luminosity, normalize=normalize,
    #     )
    #     plt.tight_layout()
    #     plt.savefig(f"{output_dir}/control_plot_pion_{charge}_pz.png")

        

    # ########################################################
    # # num truth particles
    # #########################################################
    # def get_num_truth_particles(dl):
    #     events = dl.data.get(channel_name)
    #     num_particles = ak.num(events['GenPart_pdgId'])
    #     return num_particles
    # bin_edges = np.array([2, 3])
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_num_truth_particles,
    #     bin_edges=bin_edges,
    #     x_label='Number of Truth Particles',
    #     title='Control Plot: Number of Truth Particles',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_num_truth_particles.png")

    # ########################################################
    # #  Invariant mass of pi system
    # ########################################################
    def get_pi_mass(dl):
        events_pirho = ak.from_parquet("/afs/cern.ch/user/c/clamore/lep_tree_ana/output-pirho/Ztautau/filtered___raw.parquet")
        p4_pi = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(abs(events_pirho['GenPart_pdgId']) == 211))
        pion_mass = p4_pi.mass
        return pion_mass

    bin_edges = np.linspace(0.1, 0.16, 21)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_pi_mass,
        bin_edges=bin_edges,
        x_label='Invariant Mass [GeV]',
        title='Control Plot: Invariant Mass of Charged Pion',
        luminosity=luminosity, normalize=normalize,
    )
    # ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_pi_mass.png")

    def get_pi0_mass(dl):
        events_pirho = ak.from_parquet("/afs/cern.ch/user/c/clamore/lep_tree_ana/output-pirho/Ztautau/filtered___raw.parquet")
        p4_pi0 = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == 111))
        p4_pi0_filled = ak.fill_none(p4_pi0, 0)
        pi0_mass = p4_pi0_filled.mass
        return pi0_mass

    bin_edges = np.linspace(0.1, 0.16, 21)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_pi0_mass,
        bin_edges=bin_edges,
        x_label='Invariant Mass [GeV]',
        title='Control Plot: Invariant Mass of Neutral Pion',
        luminosity=luminosity, normalize=normalize,
    )
    # ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_pi0_mass.png")

    # ################################################
    # # Invariant Rho mass
    # ################################################

    def get_rho_mass(dl):
        events_pirho = ak.from_parquet("/afs/cern.ch/user/c/clamore/lep_tree_ana/output-pirho/Ztautau/filtered___raw.parquet")
        p4_piplus = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == 211))
        p4_piminus = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == -211))
        p4_pi0 = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == 111)) 
        p4_piplus_filled = ak.fill_none(p4_piplus, 0)
        p4_pi0_filled = ak.fill_none(p4_pi0, 0)
        p4_rho = p4_piplus_filled + p4_pi0_filled
        
        rho_mass = p4_rho.mass
        rho_mass_filtered = rho_mass[rho_mass > 0.2]
        return rho_mass_filtered

    bin_edges = np.linspace(0, 1.5, 30)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_rho_mass,
        bin_edges=bin_edges,
        x_label='Invariant Mass [GeV]',
        title='Control Plot: Invariant Mass of Rho Meson',
        luminosity=luminosity, normalize=normalize,
    )
    # ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_rho_mass.png")


    # ################################################
    # # Invariant pirho mass
    # ################################################

    def get_pirho_mass(dl):
        events_pirho = ak.from_parquet("/afs/cern.ch/user/c/clamore/lep_tree_ana/output-pirho/Ztautau/filtered___raw.parquet")
        p4_piplus = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == 211))
        p4_piminus = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == -211))
        p4_pi0 = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == 111))
        p4_piplus_filled = ak.fill_none(p4_piplus, 0)
        p4_piminus_filled = ak.fill_none(p4_piminus, 0)
        p4_pi0_filled = ak.fill_none(p4_pi0, 0)
        p4_rho = p4_piplus_filled + p4_pi0_filled
        p4_pirho = p4_rho + p4_piminus_filled

        pirho_mass = p4_pirho.mass
        pirho_mass_filtered = pirho_mass[(pirho_mass > 2.5) & (pirho_mass < 100)]
        return pirho_mass_filtered

    bin_edges = np.linspace(0, 100, 30)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_pirho_mass,
        bin_edges=bin_edges,
        x_label='Invariant Mass [GeV]',
        title='Control Plot: Invariant Mass of Pi-Rho System',
        luminosity=luminosity, normalize=normalize,
    )
    # ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_pirho_mass.png")

    # ################################################
    # # Angle between two pions
    # ################################################
    def get_angle_between_pirho(dl):
        events_pirho = ak.from_parquet("/afs/cern.ch/user/c/clamore/lep_tree_ana/output-pirho/Ztautau/filtered___raw.parquet")
        p4_piplus = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == 211))
        p4_piminus = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == -211))
        p4_pi0 = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == 111))
        p4_piplus_filled = ak.fill_none(p4_piplus, 0)
        p4_piminus_filled = ak.fill_none(p4_piminus, 0)
        p4_pi0_filled = ak.fill_none(p4_pi0, 0)
        p4_rho = p4_piplus_filled + p4_pi0_filled
        angles = p4_piminus_filled.deltaangle(p4_rho)
        return angles

    bin_edges = np.linspace(0, 3.14, 30)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_angle_between_pirho,
        bin_edges=bin_edges,
        x_label='Pi-Rho Angle [rad]',
        title='Control Plot: Angle between Pi-Rho',
        luminosity=luminosity, normalize=normalize,
    )
    plt.tight_layout()
    plt.xlim(0, np.pi),
    plt.xticks([0, np.pi/2, np.pi], ["0", "π/2", "π"]),
    plt.savefig(f"{output_dir}/control_plot_angle_between_pirho.png")

    # # ################################################
    # # # Thrust costh
    # # ################################################
    # # def get_thrust_costh(dl):
    # #     events = dl.data.get(channel_name)
    # #     thrust_magnitude = events['thrust_Mag'] # avoid division by zero
    # #     thrust_z = events['thrust_z']
    # #     thrust_cosine = (thrust_z / thrust_magnitude)
    # #     return thrust_cosine
    # # bin_edges = np.linspace(-1, 1, 21)
    # # fig, ax, ax_ratio = do_control_plot(
    # #     dl_dict,
    # #     func_get_variable=get_thrust_costh,
    # #     bin_edges=bin_edges,
    # #     x_label=r'cos(\theta_{thrust})',
    # #     title=r'Control Plot: Thrust cos(\theta)',
    # #     luminosity=luminosity, normalize=normalize,
    # # )
    # # plt.tight_layout()
    # # plt.savefig(f"{output_dir}/control_plot_thrust_costh.png")

    # # def get_thrust_phi(dl):
    # #     events = dl.data.get(channel_name)
    # #     thrust_x = events['thrust_x']
    # #     thrust_y = events['thrust_y']
    # #     thrust_phi = np.arctan2(thrust_y, thrust_x)
    # #     return thrust_phi
    # # bin_edges = np.linspace(-np.pi, np.pi, 21)
    # # fig, ax, ax_ratio = do_control_plot(
    # #     dl_dict,
    # #     func_get_variable=get_thrust_phi,
    # #     bin_edges=bin_edges,
    # #     x_label=r'$\phi_{thrust}$ [rad]',
    # #     title=r'Control Plot: Thrust $\phi$',
    # #     luminosity=luminosity, normalize=normalize,
    # # )
    # # plt.tight_layout()
    # # plt.savefig(f"{output_dir}/control_plot_thrust_phi.png")

    # # def get_thrust_x(dl):
    # #     events = dl.data.get(channel_name)
    # #     thrust_x = events['thrust_x']
    # #     return thrust_x
    # # bin_edges = np.linspace(-1, 1, 21)
    # # fig, ax, ax_ratio = do_control_plot(
    # #     dl_dict,
    # #     func_get_variable=get_thrust_x,
    # #     bin_edges=bin_edges,
    # #     x_label=r'Thrust X',
    # #     title=r'Control Plot: Thrust X',
    # #     luminosity=luminosity, normalize=normalize,
    # # )
    # # plt.tight_layout()
    # # plt.savefig(f"{output_dir}/control_plot_thrust_x.png")
    # # def get_thrust_y(dl):
    # #     events = dl.data.get(channel_name)
    # #     thrust_y = events['thrust_y']
    # #     return thrust_y
    # # bin_edges = np.linspace(-1, 1, 21)
    # # fig, ax, ax_ratio = do_control_plot(
    # #     dl_dict,
    # #     func_get_variable=get_thrust_y,
    # #     bin_edges=bin_edges,
    # #     x_label=r'Thrust Y',
    # #     title=r'Control Plot: Thrust Y',
    # #     luminosity=luminosity, normalize=normalize,
    # # )
    # # plt.tight_layout()
    # # plt.savefig(f"{output_dir}/control_plot_thrust_y.png")
    # # def get_thrust_z(dl):
    # #     events = dl.data.get(channel_name)
    # #     thrust_z = events['thrust_z']
    # #     return thrust_z
    # # bin_edges = np.linspace(-1, 1, 21)
    # # fig, ax, ax_ratio = do_control_plot(
    # #     dl_dict,
    # #     func_get_variable=get_thrust_z,
    # #     bin_edges=bin_edges,
    # #     x_label=r'Thrust Z',
    # #     title=r'Control Plot: Thrust Z',
    # #     luminosity=luminosity, normalize=normalize,
    # # )
    # # plt.tight_layout()
    # # plt.savefig(f"{output_dir}/control_plot_thrust_z.png")


    # ################################################
    # # -log10(1-thrust)
    # ################################################
    # def get_neglog1mthrust(dl):
    #     events = dl.data.get(channel_name)
    #     thrust_magnitude = events['thrust_Mag']
    #     neglog1mthrust = -np.log10(1 - thrust_magnitude + 1e-10) # avoid log(0)
    #     return neglog1mthrust
    # bin_edges = np.linspace(2.5, 4, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_neglog1mthrust,
    #     bin_edges=bin_edges,
    #     x_label=r'-log10(1 - thrust)',
    #     title=r'Control Plot: -log10(1 - thrust)',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_thrust_neglog1mthrust.png")


    # ################################################
    # # pt of rho
    # ################################################
    def get_rho_pt(dl):
        events_pirho = ak.from_parquet("/afs/cern.ch/user/c/clamore/lep_tree_ana/output-pirho/Ztautau/filtered___raw.parquet")
        p4_piplus = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == 211))
        p4_piminus = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == -211))
        p4_pi0 = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == 111)) 
        p4_piplus_filled = ak.fill_none(p4_piplus, 0)
        p4_pi0_filled = ak.fill_none(p4_pi0, 0)
        p4_rho = p4_piplus_filled + p4_pi0_filled
        
        rho_pt = p4_rho.pt
        # rho_pt_filtered = rho_pt[rho_pt < 30]
        return rho_pt

    bin_edges = np.linspace(0, 50, 35)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_rho_pt,
        bin_edges=bin_edges,
        x_label='Transverse Momentum of Charged Rho [GeV]',
        title='Control Plot: pt of Charged Rho',
        luminosity=luminosity, normalize=normalize,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_charged_rho_pt.png")


    # ################################################
    # # pt of charged pion
    # ################################################
    def get_charged_pion_pt(dl):
        events_pirho = ak.from_parquet("/afs/cern.ch/user/c/clamore/lep_tree_ana/output-pirho/Ztautau/filtered___raw.parquet")
        p4_piplus = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == 211))
        p4_piminus = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == -211))
        p4_pi0 = get_p4_from_ak_events(prefix="GenPart_vector", events=events_pirho, flag=(events_pirho['GenPart_pdgId'] == 111)) 
        p4_piplus_filled = ak.fill_none(p4_piplus, 0)
        p4_pi0_filled = ak.fill_none(p4_pi0, 0)
        p4_rho = p4_piplus_filled + p4_pi0_filled
        
        piplus_pt = p4_piplus_filled.pt
        # piplus_pt_filtered = piplus_pt[(piplus_pt < 30)]
        return piplus_pt

    bin_edges = np.linspace(0, 50, 35)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_charged_pion_pt,
        bin_edges=bin_edges,
        x_label='Transverse Momentum of Charged Pion [GeV]',
        title='Control Plot: pt of Charged Pion',
        luminosity=luminosity, normalize=normalize,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_charged_pion_pt.png")


    # ################################################
    # # pt of all particles
    # ################################################
    # def get_total_particle_pt(dl):
    #     events = dl.data.get(channel_name)
    #     flag = ak.ones_like(events['GenPart_pdgId'], dtype=bool)
    #     p4 = get_sum_p4_from_ak_events(events, flag)
    #     return p4.pt
    # bin_edges = np.linspace(0, 100, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_total_particle_pt,
    #     bin_edges=bin_edges,
    #     x_label='Transverse Momentum of All Particles [GeV]',
    #     title='Control Plot: pt of All Particles',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_total_particle_pt.png")

    # ################################################
    # # HT
    # ################################################
    # def get_ht(dl):
    #     events = dl.data.get(channel_name)
    #     flag = ak.ones_like(events['GenPart_pdgId'], dtype=bool)
    #     p4_all = get_all_p4_from_ak_events(events, flag)
    #     ht = ak.sum(p4_all.pt, axis=-1)
    #     return ht
    # bin_edges = np.linspace(10, 65, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_ht,
    #     bin_edges=bin_edges,
    #     x_label='Ht [GeV]',
    #     title='Control Plot: HT',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_ht.png")



    # ################################################
    # # Missing transverse and longitudinal momentum
    # ################################################
    # def get_missing_momentum(dl):
    #     events = dl.data.get(channel_name)
    #     missing_px = -ak.sum(events['GenPart_vector_fCoordinates_fX'], axis=-1)
    #     missing_py = -ak.sum(events['GenPart_vector_fCoordinates_fY'], axis=-1)
    #     missing_pz = -ak.sum(events['GenPart_vector_fCoordinates_fZ'], axis=-1)
    #     missing_p = np.sqrt(missing_px**2 + missing_py**2 + missing_pz**2)
    #     return missing_p
    # bin_edges = np.linspace(0, 40, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_missing_momentum,
    #     bin_edges=bin_edges,
    #     x_label='Missing Momentum [GeV]',
    #     title='Control Plot: Missing Momentum',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # # ax.set_yscale('log')
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_missing_momentum.png")

    # # missing pt
    # def get_missing_pt(dl):
    #     events = dl.data.get(channel_name)
    #     missing_px = -ak.sum(events['GenPart_vector_fCoordinates_fX'], axis=-1)
    #     missing_py = -ak.sum(events['GenPart_vector_fCoordinates_fY'], axis=-1)
    #     missing_pt = np.sqrt(missing_px**2 + missing_py**2)
    #     return missing_pt
    # bin_edges = np.linspace(0, 40, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_missing_pt,
    #     bin_edges=bin_edges,
    #     x_label='Missing Pt [GeV]',
    #     title='Control Plot: Missing Pt',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_missing_pt.png")

    # # missing pz
    # def get_missing_pz(dl):
    #     events = dl.data.get(channel_name)
    #     missing_pz = -ak.sum(events['GenPart_vector_fCoordinates_fZ'], axis=-1)
    #     return missing_pz
    # bin_edges = np.linspace(-25, 25, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_missing_pz,
    #     bin_edges=bin_edges,
    #     x_label='Missing Pz [GeV]',
    #     title='Control Plot: Missing Pz',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_missing_pz.png")
    # # missing E
    # def get_missing_energy(dl):
    #     events = dl.data.get(channel_name)
    #     missing_E = cme - ak.sum(events['GenPart_vector_fCoordinates_fT'], axis=-1)
    #     return missing_E
    # bin_edges = np.linspace(20, 90, 21)
    # fig, ax, ax_ratio = do_control_plot(
    #     dl_dict,
    #     func_get_variable=get_missing_energy,
    #     bin_edges=bin_edges,
    #     x_label='Missing Energy [GeV]',
    #     title='Control Plot: Missing Energy',
    #     luminosity=luminosity, normalize=normalize,
    # )
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/control_plot_missing_energy.png")


# def make_control_plots_leplep(dl_dict, luminosity, normalize, output_dir, channel_name="leplep"):
#     # apply cuts for leplep channel if needed
#     for dl_name, dl in dl_dict.items():
#         events = dl.data.get(channel_name)
#         pass_filter = ak.ones_like(events['evtNumber'], dtype=bool)

#         reco_pdgId = events['Part_pdgId']
#         reco_abspdgid = np.abs(reco_pdgId)
#         flag_is_mu = (reco_abspdgid == 6)
#         flag_is_el = (reco_abspdgid == 2)
#         flag_is_lepton = flag_is_mu | flag_is_el
#         num_electrons = ak.sum(flag_is_el, axis=-1)
#         num_muons = ak.sum(flag_is_mu, axis=-1)
#         num_leptons = ak.sum(flag_is_lepton, axis=-1)
#         num_particles = ak.num(events['Part_pdgId'])

#         reco_charge = events['Part_charge']
#         # total_charge = ak.sum(reco_charge, axis=-1)
#         lep_total_charge = ak.sum(reco_charge * flag_is_lepton, axis=-1)

#         # require exactly two leptons and no other particles
#         # pass_filter = (num_electrons == 2) & pass_filter
#         # pass_filter = (num_muons == 2) & pass_filter
#         pass_filter = (num_leptons == 2) & pass_filter
#         # pass_filter = (num_particles <= 4) & pass_filter
#         # pass_filter = (num_particles == 2) & pass_filter
#         pass_filter = (lep_total_charge == 0) & pass_filter

#         # # mll > 80
#         # p4_leptons = get_sum_p4_from_ak_events(events, flag_is_lepton)
#         # mll = p4_leptons.mass
#         # pass_filter = (mll > 80) & pass_filter

#         dl.data[channel_name] = events[pass_filter]


#     # MLL
#     def get_mll(dl):
#         events = dl.data.get(channel_name)

#         recpart_abspdgid = np.abs(events['Part_pdgId'])
#         flag_is_mu = (recpart_abspdgid == 6)
#         flag_is_el = (recpart_abspdgid == 2)
#         flag_is_lepton = flag_is_mu | flag_is_el
#         p4_leptons = get_sum_p4_from_ak_events(events, flag_is_lepton)

#         return p4_leptons.mass
#     bin_edges = np.linspace(0, 100, 41)
#     fig, ax, ax_ratio = do_control_plot(
#         dl_dict,
#         func_get_variable=get_mll,
#         bin_edges=bin_edges,
#         x_label='Invariant Mass of Lepton Pair [GeV]',
#         title='Control Plot: Invariant Mass of Lepton Pair',
#         luminosity=luminosity, normalize=normalize,
#     )
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/control_plot_mll.png")

#     # missing E
#     def get_missing_energy(dl):
#         events = dl.data.get(channel_name)
#         missing_E = cme - ak.sum(events['GenPart_vector_fCoordinates_fT'], axis=-1)
#         return missing_E
#     bin_edges = np.linspace(0, 90, 21)
#     fig, ax, ax_ratio = do_control_plot(
#         dl_dict,
#         func_get_variable=get_missing_energy,
#         bin_edges=bin_edges,
#         x_label='Missing Energy [GeV]',
#         title='Control Plot: Missing Energy',
#         luminosity=luminosity, normalize=normalize,
#     )
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/control_plot_missing_energy.png")

#     # Angle between two leptons
#     def get_angle_between_leptons(dl):
#         events = dl.data.get(channel_name)
#         reco_pdgId = events['Part_pdgId']
#         reco_charge = events['Part_charge']
#         reco_abs_pdgId = np.abs(reco_pdgId)
#         flag_is_mu = (reco_abs_pdgId == 6)
#         flag_is_el = (reco_abs_pdgId == 2)
#         flag_is_lepton = flag_is_mu | flag_is_el
#         flag_positive = flag_is_lepton & (reco_charge == 1)
#         flag_negative = flag_is_lepton & (reco_charge == -1)
#         p4_lepton_pos = get_p4_from_ak_events(events, flag_positive)
#         p4_lepton_neg = get_p4_from_ak_events(events, flag_negative)
#         angles = p4_lepton_pos.deltaangle(p4_lepton_neg)
#         angles = ak.to_numpy(angles)
#         nan_angles = np.isnan(angles)
#         angles[nan_angles] = np.pi
#         return angles
#     bin_edges = np.linspace(2.8, np.pi, 21)
#     fig, ax, ax_ratio = do_control_plot(
#         dl_dict,
#         func_get_variable=get_angle_between_leptons,
#         bin_edges=bin_edges,
#         x_label='Angle between two leptons [rad]',
#         title='Control Plot: Angle between Two Leptons',
#         luminosity=luminosity, normalize=normalize,
#     )
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/control_plot_angle_between_leptons.png")


#     # HT
#     def get_ht(dl):
#         events = dl.data.get(channel_name)
#         flag = ak.ones_like(events['Part_charge'], dtype=bool)
#         p4_all = get_all_p4_from_ak_events(events, flag)
#         ht = ak.sum(p4_all.pt, axis=-1)
#         return ht
#     bin_edges = np.linspace(0, 80, 21)
#     fig, ax, ax_ratio = do_control_plot(
#         dl_dict,
#         func_get_variable=get_ht,
#         bin_edges=bin_edges,
#         x_label='Ht [GeV]',
#         title='Control Plot: HT',
#         luminosity=luminosity, normalize=normalize,
#     )
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/control_plot_ht.png")

#     # particle multiplicity
#     def get_num_reco_particles(dl):
#         events = dl.data.get(channel_name)
#         num_particles = ak.num(events['Part_pdgId'])
#         return num_particles
#     bin_edges = np.arange(2, 30, 1)
#     fig, ax, ax_ratio = do_control_plot(
#         dl_dict,
#         func_get_variable=get_num_reco_particles,
#         bin_edges=bin_edges,
#         x_label='Number of Reconstructed Particles',
#         title='Control Plot: Number of Reconstructed Particles',
#         luminosity=luminosity, normalize=normalize,
#     )
#     ax.set_yscale('log')
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/control_plot_num_reco_particles.png")
    



class ControlPlotProcessor(BaseProcessor):
    def __init__(self, config, output_dir):
        """
        Processor to make control plots for data/MC comparison.
        """
        super().__init__(config)
        self.config = config
        if 'output_dir_name' in config:
            output_dir = f"{output_dir}/{config['output_dir_name']}"
        else:
            output_dir = f"{output_dir}/control_plots/"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.luminosity = config.get('luminosity', None) 
        self.normalize = (self.luminosity is None)
        self.channels = config.get('channels', ['pirho'])
        # self.normalize = True

    def run(self, dl_dict):
        if 'pipi' in self.channels:
            output_dir_pipi = f"{self.output_dir}/pipi/"
            os.makedirs(output_dir_pipi, exist_ok=True)
            make_control_plots_pipi(
                dl_dict,
                luminosity=self.luminosity,
                normalize=self.normalize,
                output_dir=output_dir_pipi,
                channel_name="pipi",
            )
        if 'leplep' in self.channels:
            output_dir_leplep = f"{self.output_dir}/leplep/"
            os.makedirs(output_dir_leplep, exist_ok=True)
            make_control_plots_leplep(
                dl_dict,
                luminosity=self.luminosity,
                normalize=self.normalize,
                output_dir=output_dir_leplep,
                channel_name="leplep",
            )
        if 'pirho' in self.channels:
            output_dir_pirho = f"{self.output_dir}/pirho/"
            os.makedirs(output_dir_pirho, exist_ok=True)
            make_control_plots_pirho(
                dl_dict,
                luminosity=self.luminosity,
                normalize=self.normalize,
                output_dir=output_dir_pirho,
                channel_name="pirho",
            )




    def finalize(self):
        pass