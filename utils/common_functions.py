import awkward as ak
import matplotlib.pyplot as plt
import vector

cme = 91.25 # GeV
m_tau = 1.77686 # GeV

def print_and_write_to_file(text, file_path, mode='a'):
    print(text)
    with open(file_path, mode) as f:
        f.write(text + '\n')

def get_color_iterator(n):
    return iter(plt.cm.tab20.colors * (n // 20 + 1))

def get_p4_from_ak_events(events, flag, prefix='Part_fourMomentum'):
    px = ak.firsts(events[f'{prefix}_fCoordinates_fX'][flag][...,::-1]).to_numpy()
    py = ak.firsts(events[f'{prefix}_fCoordinates_fY'][flag][...,::-1]).to_numpy()
    pz = ak.firsts(events[f'{prefix}_fCoordinates_fZ'][flag][...,::-1]).to_numpy()
    E =  ak.firsts(events[f'{prefix}_fCoordinates_fT'][flag][...,::-1]).to_numpy()
    p4 = vector.zip({
        "px": px,
        "py": py,
        "pz": pz,
        "E": E,
    })
    return p4

def get_all_p4_from_ak_events(events, flag, prefix='Part_fourMomentum'):
    px = events[f'{prefix}_fCoordinates_fX'][flag] # .to_numpy()
    py = events[f'{prefix}_fCoordinates_fY'][flag] # .to_numpy()
    pz = events[f'{prefix}_fCoordinates_fZ'][flag] # .to_numpy()
    E =  events[f'{prefix}_fCoordinates_fT'][flag] # .to_numpy()
    p4 = vector.zip({
        "px": px,
        "py": py,
        "pz": pz,
        "E": E,
    })
    return p4

def get_sum_p4_from_ak_events(events, flag, prefix='Part_fourMomentum'):
    px = ak.sum(events[f'{prefix}_fCoordinates_fX'][flag], axis=-1).to_numpy()
    py = ak.sum(events[f'{prefix}_fCoordinates_fY'][flag], axis=-1).to_numpy()
    pz = ak.sum(events[f'{prefix}_fCoordinates_fZ'][flag], axis=-1).to_numpy()
    E =  ak.sum(events[f'{prefix}_fCoordinates_fT'][flag], axis=-1).to_numpy()
    p4 = vector.zip({
        "px": px,
        "py": py,
        "pz": pz,
        "E": E,
    })
    return p4