# ml_pipeline_lite

This directory holds small, nominal-only rewrites of the legacy `ml_pipeline`.

## `build_evenet_input_from_parquet.py`

This rewrite keeps only the fields needed for later EveNet work:

- filtered per-particle `Part_*` inputs needed by EveNet
- `Global` condition features needed by EveNet
- visible tau `a/b` four-vectors
- truth tau `a/b` four-vectors for `Ztautau` only
- target invisible slot `pt / eta / phi` for `Ztautau` only
- classification target index / class name aligned with the generated EveNet class order
- source slot mapping
- simple event metadata and region cuts
- recomputed truth angular observables

It is designed for large parquet inputs:

- batch-wise parquet streaming
- one worker per input parquet
- bounded-memory parquet shard outputs
- worker-side histogram filling with merged monitoring plots at the end
- `input_files` are used first, so the baseline-selected parquet is the default source
- MC normalization uses the sample-level sum of `initial_total_num_events` across all parquet files in the sample

### Example

```bash
python3 build_evenet_input_from_parquet.py   \
  --analysis-config config/analysis.yaml   \
  --output-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_baseline_v2/  \
  --batch-size 50000  \
  --rows-per-shard 100000 \
  --num-workers 4
```

Outputs:

- `shards/<sample>/*.parquet`
- `monitoring/<sample>/*.png`
- `monitoring/comparison/*.png` for data vs stacked MC control plots
- `manifest.json`

The output parquet stores:

- the filtered sequential EveNet inputs as `Part_*` jagged columns
- the global condition inputs as flat scalar columns
- `classification`
- `classification_target_index`
- `classification_target_name`

with the same class ordering used by `generated_event_info.yaml`.

If a required `Global` feature is not stored directly, `missing_p4` is used for
the nominal fallback, including `missing_pz`.


## `build evenet config`
To generate the `generated_event_info.yaml` schema, we need to provide two input yaml files:
```
python3 generate_event_info_yaml.py \
  --analysis-config config/analysis.yaml \
  --evenet-config config/evenet_schema.yaml \ 
  --output config/generated_event_info.yaml
```
## `preprocess_evenet_parquet.py`

Preprocess the lite parquet shards into shuffled train/val/test parquet files
using the standard EveNet preprocessing logic.

This stage is intentionally simple:

- read the lite shard manifest
- use all non-data shards as training input
- optionally preprocess data shards into `store-dir/data`
- event-level split inside each shard
- shuffle rows per written parquet file
- keep `event_weight` unchanged

Example:

```bash
python3 preprocess_evenet_parquet.py  \
  --manifest /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_baseline_v2/manifest.json \
  --config config/preprocess_config.yaml  \
  --store-dir /pscratch/sd/t/tihsu/database/ZtautauAnalysis/ml_baseline_v2/evenet_input  \
  --split-ratio 0.4,0.1,0.5  \
  --num-workers 4  \
  --verbose
```

Outputs:

- `train_*.parquet`
- `val/val_*.parquet`
- `test/test_*.parquet`
- `data/data_*.parquet`
- `shape_metadata.json`
- `normalization.pt`
- `preprocess_manifest.json`

## `generate_event_info_yaml.py`

Generate the `generated_event_info.yaml` schema consumed by:

- `ml_pipeline/config/preprocess_config.yaml`
- `ml_pipeline/config/train.yaml`
- `ml_pipeline/config/train_pretrain.yaml`

Example:

```bash
python3 ml_pipeline_lite/generate_event_info_yaml.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --evenet-config ml_pipeline/config/evenet_schema.yaml \
  --output ml_pipeline_lite/generated_event_info.yaml
```

This writes:

- `generated_event_info.yaml`
- `generated_event_info.summary.json`
