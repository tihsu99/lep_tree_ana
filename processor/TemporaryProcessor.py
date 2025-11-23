import numpy as np
from BaseProcessor import BaseProcessor
import utils.plotter as plotter
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak

cme = 91.25 # GeV
def get_color_iterator(n):
    return iter(plt.cm.tab10.colors * (n // 10 + 1))

def get_p4(events, flag, prefix='Part_fourMomentum'):
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

class TemporaryProcessor(BaseProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.output_dir = self.config.get("output_dir", "./") + "/plots/"
        os.makedirs(self.output_dir, exist_ok=True)


    def run(self, dl: DataLoader.DataLoader):
        ...



    def finalize(self):
        pass