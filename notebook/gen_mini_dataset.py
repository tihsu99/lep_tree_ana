import awkward as ak
import numpy as np
import os

from_file = '/eos/user/c/cmo/project/ZtautauLep/tree_ana/run/20260406-hadhad/Ztautau/filtered___raw.parquet'
to_file = 'dataset/filtered___raw.parquet'
fraction = 0.1

if not os.path.exists(os.path.dirname(to_file)):
    os.makedirs(os.path.dirname(to_file))

ds = ak.from_parquet(from_file)
# select = np.isin(ds['event_category'], [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44])
select = np.isin(ds['event_category'], [11, 12, 21, 22])
ds = ds[select]
ds_mini = ds[:int(len(ds) * fraction)]
ds_mini['initial_total_num_events'] = len(ds)
ak.to_parquet(ds_mini, to_file, compression='snappy')