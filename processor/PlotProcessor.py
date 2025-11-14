import numpy as np
from BaseProcessor import BaseProcessor
import DataLoader
import matplotlib.pyplot as plt
import os
import vector

class AngularProcessor(BaseProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.output_dir = self.config.get("output_dir", "./") + "/plots/"
        os.makedirs(self.output_dir, exist_ok=True)


    def run(self, dl: DataLoader.DataLoader):
        # -----------------------------------------------
        # plot the theta of taus
        # -----------------------------------------------
        fig, ax = plt.subplots(dpi=300)
        bins = np.linspace(0, np.pi, 50)
        # theta of tau1
        theta_tau1 = dl.vectored_data['single_pion_decay/TRUTH/tau1'].theta
        ax.hist(theta_tau1, bins=bins, histtype='step', label='tau1', density=True)
        # theta of tau2
        theta_tau2 = dl.vectored_data['single_pion_decay/TRUTH/tau2'].theta
        ax.hist(theta_tau2, bins=bins, histtype='step', label='tau2', density=True)
        ax.set_xlabel('theta (rad)')
        ax.set_ylabel('Normalized Entries')
        ax.set_title('Theta Distribution of Taus')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'theta_taus.png'))

        
        # -----------------------------------------------
        # plot the costh of taus
        # -----------------------------------------------
        fig, ax = plt.subplots(dpi=300)
        bins = np.linspace(-1, 1, 50)
        # costh of tau1
        costh_tau1 = dl.vectored_data['single_pion_decay/TRUTH/tau1'].costheta
        ax.hist(costh_tau1, bins=bins, histtype='step', label='tau1', density=True)
        # costh of tau2
        costh_tau2 = dl.vectored_data['single_pion_decay/TRUTH/tau2'].costheta
        ax.hist(costh_tau2, bins=bins, histtype='step', label='tau2', density=True)
        ax.set_xlabel('cos(theta)')
        ax.set_ylabel('Normalized Entries')
        ax.set_title('Costheta Distribution of Taus')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'costheta_taus.png'))

        # -----------------------------------------------
        # plot the pT of taus
        # -----------------------------------------------
        fig, ax = plt.subplots(dpi=300)
        bins_pt = np.linspace(0, 50, 50)
        # pT of tau1
        pt_tau1 = dl.vectored_data['single_pion_decay/TRUTH/tau1'].pt
        ax.hist(pt_tau1, bins=bins_pt, histtype='step', label='tau1', density=True)
        # pT of tau2
        pt_tau2 = dl.vectored_data['single_pion_decay/TRUTH/tau2'].pt
        ax.hist(pt_tau2, bins=bins_pt, histtype='step', label='tau2', density=True)
        ax.set_xlabel('pT (GeV)')
        ax.set_ylabel('Normalized Entries')
        ax.set_title('Transverse Momentum Distribution of Taus')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'pt_taus.png'))


        # -----------------------------------------------
        # plot the angle between tau and pi for all events
        # -----------------------------------------------
        fig, ax = plt.subplots(dpi=300)
        bins = np.linspace(0, 0.5, 50)
        # angle between tau1 and vischild_tau1_p4
        angle_tau1_pi = dl.vectored_data['single_pion_decay/TRUTH/tau1'].deltaangle(dl.vectored_data['single_pion_decay/TRUTH/vischild_tau1'])
        ax.hist(angle_tau1_pi, bins=bins, histtype='step', label='tau1 vs pi1', density=True)
        # angle between tau2 and vischild_tau2_p4
        angle_tau2_pi = dl.vectored_data['single_pion_decay/TRUTH/tau2'].deltaangle(dl.vectored_data['single_pion_decay/TRUTH/vischild_tau2'])
        ax.hist(angle_tau2_pi, bins=bins, histtype='step', label='tau2 vs pi2', density=True)
        ax.set_xlabel('Angle (rad)')
        ax.set_ylabel('Normalized Entries')
        ax.set_title('Angle between tau and pion')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'angle_tau_pi.png'))


        # -----------------------------------------------
        # plot inv mass of tau1 tau2 and pi1 pi2 + nu1 nu2 and Z
        # -----------------------------------------------
        fig1, ax1 = plt.subplots(dpi=300)
        bins_mass = np.linspace(85, 95, 50)
        # inv mass of tau1 and tau2
        tau1_p4 = dl.vectored_data['single_pion_decay/TRUTH/tau1']
        tau2_p4 = dl.vectored_data['single_pion_decay/TRUTH/tau2']
        inv_mass_tau1_tau2 = (tau1_p4 + tau2_p4).mass
        ax1.hist(inv_mass_tau1_tau2, bins=bins_mass, histtype='step', label='tau1 + tau2', density=True)
        # inv mass of vischild_tau1 + vischild_tau2 + nu1 + nu2
        vischild_tau1_p4 = dl.vectored_data['single_pion_decay/TRUTH/vischild_tau1']
        vischild_tau2_p4 = dl.vectored_data['single_pion_decay/TRUTH/vischild_tau2']
        nu1_p4 = dl.vectored_data['single_pion_decay/TRUTH/nu_tau1']
        nu2_p4 = dl.vectored_data['single_pion_decay/TRUTH/nu_tau2']
        inv_mass_pi1_pi2_nu1_nu2 = (vischild_tau1_p4 + vischild_tau2_p4 + nu1_p4 + nu2_p4).mass
        ax1.hist(inv_mass_pi1_pi2_nu1_nu2, bins=bins_mass, histtype='step', label='pi1 + pi2 + nu1 + nu2', density=True)
        # Z boson mass peak at 91.2 GeV
        inv_mass_z = dl.vectored_data['single_pion_decay/TRUTH/Z'].mass
        ax1.hist(inv_mass_z, bins=bins_mass, histtype='step', label='Z boson', density=True)
        ax1.axvline(91.2, color='r', linestyle='--', label='Z boson mass (91.2 GeV)')
        ax1.set_xlabel('Invariant Mass (GeV)')
        ax1.set_ylabel('Normalized Entries')
        ax1.set_title('Invariant Mass Distributions')
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(os.path.join(self.output_dir, 'invariant_mass_distributions.png'))



        # -----------------------------------------------
        # plot cos(angle_tau1) * cos(angle_tau2) distribution
        # -----------------------------------------------
        fig, ax = plt.subplots(dpi=300)
        cos_angle_tau1 = dl.vectored_data['single_pion_decay/TRUTH/tau1'].costheta
        cos_angle_tau2 = dl.vectored_data['single_pion_decay/TRUTH/tau2'].costheta
        cos_product = np.array(cos_angle_tau1) * np.array(cos_angle_tau2)
        bins_cos_product = np.linspace(-1, 0.25, 50)
        ax.hist(cos_product, bins=bins_cos_product, histtype='step', label='cos(angle_tau1) * cos(angle_tau2)', density=True)
        ax.set_xlabel('cos(angle_tau1) * cos(angle_tau2)')
        ax.set_ylabel('Normalized Entries')
        ax.set_title('Distribution of cos(angle_tau1) * cos(angle_tau2)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'cos_angle_tau1_times_cos_angle_tau2.png'))




        # # -----------------------------------------------
        # # plot 2-D plot for cos(angle_tau1) vs cos(angle_tau2)
        # # -----------------------------------------------
        # fig2, ax2 = plt.subplots(dpi=300)
        # cos_angle_tau1 = np.array(dl.vectored_data['single_pion_decay/TRUTH/tau1'].costheta)
        # cos_angle_tau2 = np.array(dl.vectored_data['single_pion_decay/TRUTH/tau2'].costheta)

        # h2d, xedges, yedges = np.histogram2d(cos_angle_tau1, cos_angle_tau2, bins=50, range=[[-1, 1], [-1, 1]], density=True)
        # # log z scale
        # h2d = np.log1p(h2d)
        # X, Y = np.meshgrid(xedges, yedges)
        # pcm = ax2.pcolormesh(X, Y, h2d.T, cmap='viridis')
        # fig2.colorbar(pcm, ax=ax2, label='Log Normalized Entries')
        # ax2.set_xlabel('cos(angle_tau1)')
        # ax2.set_ylabel('cos(angle_tau2)')
        # ax2.set_title('2D Histogram of cos(angle_tau1) vs cos(angle_tau2)')
        # fig2.tight_layout()
        # fig2.savefig(os.path.join(self.output_dir, 'cos_angle_tau1_vs_cos_angle_tau2.png'))
        # # print correlation matrix
        # correlation_matrix = np.corrcoef(cos_angle_tau1, cos_angle_tau2)
        # print("Correlation matrix between cos(angle_tau1) and cos(angle_tau2):")
        # print(correlation_matrix)


        # # plot 2-D plot for cos(pi1) vs cos(pi2)
        # fig2, ax2 = plt.subplots(dpi=300)
        # cos_pi1 = dl.vectored_data['single_pion_decay/TRUTH/vischild_tau1'].costheta
        # cos_pi2 = dl.vectored_data['single_pion_decay/TRUTH/vischild_tau2'].costheta
        # cos_pi1 = np.array(cos_pi1)
        # cos_pi2 = np.array(cos_pi2)

        # h2d, xedges, yedges = np.histogram2d(cos_pi1, cos_pi2, bins=50, range=[[-1, 1], [-1, 1]], density=True)
        # # log z scale
        # h2d = np.log1p(h2d)
        # X, Y = np.meshgrid(xedges, yedges)
        # pcm = ax2.pcolormesh(X, Y, h2d.T, cmap='viridis')
        # fig2.colorbar(pcm, ax=ax2, label='Log Normalized Entries')
        # ax2.set_xlabel('cos(pi1)')
        # ax2.set_ylabel('cos(pi2)')
        # ax2.set_title('2D Histogram of cos(pi1) vs cos(pi2)')
        # fig2.tight_layout()
        # fig2.savefig(os.path.join(self.output_dir, 'cos_pi1_vs_cos_pi2.png'))
        # # print correlation matrix
        # correlation_matrix = np.corrcoef(cos_pi1, cos_pi2)
        # print("Correlation matrix between cos(pi1) and cos(pi2):")
        # print(correlation_matrix)

    def finalize(self):
        pass