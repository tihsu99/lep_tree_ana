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
python3 ml_pipeline_lite/build_evenet_input_from_parquet.py \
  --analysis-config ml_pipeline/config/analysis.yaml \
  --output-dir /tmp/evenet_lite_build \
  --samples Ztautau data94 \
  --batch-size 50000 \
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
