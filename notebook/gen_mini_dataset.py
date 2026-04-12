import awkward as ak
import os

from_file = '/eos/user/c/cmo/project/ZtautauLep/tree_ana/run/20260406-hadhad/Ztautau/filtered___raw.parquet'
to_file = 'dataset/filtered___raw.parquet'
fraction = 0.01

if not os.path.exists(os.path.dirname(to_file)):
    os.makedirs(os.path.dirname(to_file))

ds = ak.from_parquet(from_file)
ds_mini = ds[:int(len(ds) * fraction)]
ds_mini['initial_total_num_events'] = len(ds)
ak.to_parquet(ds_mini, to_file, compression='snappy')