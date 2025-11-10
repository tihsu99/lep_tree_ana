import numpy as np
import pandas as pd
from pathlib import Path
import os
from scipy.interpolate import interp1d
import DataLoader
from BaseProcessor import BaseProcessor

# Constants
m_Z = 91.1876  # Mass of Z [GeV]
# m_Z = 91
m_tau = 1.77686  # Mass of tau [GeV]
# m_tau = 1777 / 1000

tau_Z = 2.4414  # Decay Width of Z [GeV]
# tau_Z = 2.44
# thW = 0.49091185891597033  # Weinberg angle http://physics.nist.gov/cgi-bin/cuu/Value?sin2th
sinW2 = 0.22224648578577766  # From MG5 aMC
# sinW2 = 0.23
cosW2 = 1 - sinW2
sinW4 = sinW2 ** 2
cosW4 = cosW2 ** 2


def calculate_correlation(m_AB, theta, Qq, gVq, gAq, Fqq=1 / 4):
    result = {}

    Qt = -1  # Tau charge
    gVt = 1 / 2 * (-1 / 2 - 2 * (-1) * sinW2)
    gAt = 1 / 2 * (-1 / 2)

    # Variables
    cosT = np.cos(theta)
    sinT = np.sin(theta)
    beta = np.sqrt(1 - 4 * m_tau ** 2 / m_AB ** 2)

    # Common terms
    common_X = ((m_AB ** 2 - m_Z ** 2) ** 2 + (m_AB ** 4) * (tau_Z ** 2) / (m_Z ** 2))
    ReX = (m_AB ** 2) * (m_AB ** 2 - m_Z ** 2) / (sinW2 * cosW2 * common_X)
    X2 = (m_AB ** 4) / (sinW4 * cosW4 * common_X)

    # normalization term
    # Formula for A
    result['A'] = Fqq * (
            Qq ** 2 * Qt ** 2 * (2 - beta ** 2 * sinT ** 2)
            + 2 * Qq * Qt * ReX * (2 * beta * gAq * gAt * cosT + gVq * gVt * (2 - beta ** 2 * sinT ** 2))
            + X2 * (
                    (gVq ** 2 + gAq ** 2) *
                    (
                            2 * gVt ** 2 + 2 * beta ** 2 * gAt ** 2 -
                            beta ** 2 * (gVt ** 2 + gAt ** 2) * sinT ** 2
                    )
                    + 8 * beta * gVq * gVt * gAq * gAt * cosT
            )
    )

    # Polarization of tau
    result['Bk'] = -2 * Fqq * (
            Qq * Qt * ReX * (beta * gAt * gVq * (1 + cosT ** 2) + 2 * gAq * gVt * cosT)
            + X2 * (
                    2 * gAq * gVq * (beta ** 2 * gAt ** 2 + gVt ** 2) * cosT
                    + beta * gAt * gVt * (gVq ** 2 + gAq ** 2) * (1 + cosT ** 2)
            )
    )

    result['Br'] = -2 * Fqq * sinT * np.sqrt(1 - beta ** 2) * (
            Qq * Qt * ReX * (beta * gAt * gVq * cosT + 2 * gAq * gVt)
            + X2 * gVt * (
                    beta * gAt * (gVq ** 2 + gAq ** 2) * cosT + 2 * gAq * gVq * gVt
            )
    )

    result['Bn'] = 0

    # Spin Correlation
    result['Cnn'] = -Fqq * beta ** 2 * np.sin(theta) ** 2 * (
            Qq ** 2 * Qt ** 2 + 2 * Qq * Qt * ReX * gVq * gVt
            - X2 * (gVq ** 2 + gAq ** 2) * (gAt ** 2 - gVt ** 2)
    )

    result['Crr'] = -Fqq * np.sin(theta) ** 2 * (
            (beta ** 2 - 2) * Qq ** 2 * Qt ** 2
            + 2 * Qq * Qt * ReX * gVq * gVt * (beta ** 2 - 2)
            + X2 * ((beta ** 2 * (gAt ** 2 + gVt ** 2) - 2 * gVt ** 2) * (gVq ** 2 + gAq ** 2))
    )

    result['Ckk'] = Fqq * (
            Qq ** 2 * Qt ** 2 * ((beta ** 2 - 2) * np.sin(theta) ** 2 + 2)
            + 2 * Qq * Qt * ReX * (
                    2 * beta * gAq * gAt * np.cos(theta)
                    + gVq * gVt * ((beta ** 2 - 2) * np.sin(theta) ** 2 + 2)
            )
            + X2 * (
                    8 * beta * gVq * gVt * gAq * gAt * np.cos(theta)
                    + (gVq ** 2 + gAq ** 2) * (
                            2 * gVt ** 2 * np.cos(theta) ** 2
                            - beta ** 2 * (gAt ** 2 - gVt ** 2) * np.sin(theta) ** 2
                            + 2 * beta ** 2 * gAt ** 2
                    )
            )
    )

    result['Crk'] = 2 * Fqq * np.sin(theta) * np.sqrt(1 - beta ** 2) * (
            Qq ** 2 * Qt ** 2 * np.cos(theta)
            + Qq * Qt * ReX * (
                    beta * gAq * gAt + 2 * gVq * gVt * np.cos(theta)
            )
            + X2 * (
                    2 * beta * gAq * gAt * gVq * gVt
                    + gVt ** 2 * (gVq ** 2 + gAq ** 2) * np.cos(theta)
            )
    )
    result['Ckr'] = result['Crk']

    result['Cnr'] = 0
    result['Crn'] = 0
    result['Cnk'] = 0
    result['Ckn'] = 0

    return result


class Kinematic:
    def __init__(self):
        current_path = Path(__file__).resolve().parent
        self.pdf_table = current_path / 'pdf_mtt_13TeV.csv'
        self.pdf_df = pd.read_csv(self.pdf_table)

        self.L_uu = interp1d(
            self.pdf_df['x (m_tau_tau)'], self.pdf_df['y (L)'], kind='cubic', fill_value="extrapolate"
        )
        self.L_ddss = interp1d(
            self.pdf_df['x (m_tau_tau)'], self.pdf_df['y (L_dd/ss)'], kind='cubic', fill_value="extrapolate"
        )

        pass

    def calculate(self, m_AB, theta):
        gVe = 1 / 2 * (-1 / 2 + 2 * sinW2)
        gAe = -1 / 4
        result_ee_1 = calculate_correlation(m_AB, theta, -1, gVe, gAe)
        result_ee_2 = calculate_correlation(m_AB, np.pi - theta, -1, gVe, gAe)

        normalize = result_ee_1['A'] + result_ee_2['A']
        B = {
            key: (result_ee_1[f'B{key}'] + result_ee_2[f'B{key}']) / normalize

            for key in ['n', 'r', 'k']
        }

        C = {
            key: (result_ee_1[f'C{key}'] + result_ee_2[f'C{key}']) / normalize
            for key in ['nn', 'rr', 'kk', 'rk', 'kr', 'nr', 'rn', 'nk', 'kn']
        }

        return B, C


class KinematicsMethodProcessor(BaseProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.output_dir = self.config.get("output_dir", "./")
        os.makedirs(self.output_dir, exist_ok=True)

        self.kinematic_method = Kinematic()

    def run(self, dl: DataLoader.DataLoader):
        inv_mass_tautau = (dl.vectored_data['TRUTH/tau1'] + dl.vectored_data['TRUTH/tau2']).mass
        theta_tautau = dl.vectored_data['TRUTH/tau1'].theta
        results = [self.kinematic_method.calculate(mass, theta) for mass, theta in zip(inv_mass_tautau, theta_tautau)]

        B_matrix, C_matrix = zip(*results)
        df_B = pd.DataFrame(B_matrix)
        df_C = pd.DataFrame(C_matrix)

        # calculate average of B and C matrix
        avg_B = df_B.mean().to_dict()
        avg_C = df_C.mean().to_dict()
        for key, value in avg_B.items():
            print(f'Average B_{key}: {value}')
        for key, value in avg_C.items():
            print(f'Average C_{key}: {value}')
        

        # # Store results into the dataloader
        # dl.derived_data['KinematicsMethod/B'] = [res[0] for res in results]

        # store results into pandas DataFrame
        pd.DataFrame(df_B).to_csv(os.path.join(self.output_dir, 'KinematicsMethod_B.csv'), index=False)
        pd.DataFrame(df_C).to_csv(os.path.join(self.output_dir, 'KinematicsMethod_C.csv'), index=False)


    def finalize(self):
        pass

if __name__ == "__main__":
    from evaluation import compute_full_density_matrix

    k = Kinematic()
    results = k.calculate(90, 0.8)
    print(results)

    a = []
    for theta in np.linspace(0.6 * np.pi / 2, 1.4 * np.pi / 2, 100):
        # print(theta)
        results = k.calculate(91, theta)
        a += [results[0]['k']]

    print(a)
    print(np.mean(a))

    # Set parameters
    # mass_peak = 91.2  # Peak of mass distribution
    # sigma_mass = 10  # Standard deviation for mass
    #
    # theta_peak = np.pi / 2  # Peak of theta distribution
    # sigma_theta = 100  # Standard deviation for theta
    #
    # # Generate synthetic data
    # num_samples = 10000
    # mass_data = np.random.normal(mass_peak, sigma_mass, num_samples)
    # theta_data = np.random.normal(theta_peak, sigma_theta, num_samples)

    mass_data = np.linspace(80, 100, 100)
    theta_data = np.linspace(0.6 * np.pi / 2, 1.4 * np.pi / 2, 100)

    k = Kinematic()
    result = [results[0]['k'] for results in [
        k.calculate(mass, theta) for mass in mass_data for theta in theta_data
        # if (80 < mass < 100) and (0.6 * np.pi / 2 < theta < 1.4 * np.pi / 2)
    ]]

    print(np.mean(result))
    print((100 - 80) * (0.8 * np.pi / 2))

    pass