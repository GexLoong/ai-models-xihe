# ai-models-xihe

`ai-models-xihe` is an [ai-models](https://github.com/ecmwf-lab/ai-models) plugin to run [XiHe-GlobalOceanForecasting](https://github.com/Ocean-Intelligent-Forecasting/XiHe-GlobalOceanForecasting).

## Installation

To install the package, run:

```bash
pip install ai-models-xihe
```

Until a PyPI release is published, install from this repository instead:

```bash
python -m pip install .
```

ONNX Runtime is installed separately. Use `onnxruntime` for CPU,
`onnxruntime-gpu` for NVIDIA CUDA, or the vendor's MACA ONNX Runtime wheel for
MetaX devices. The `ONNXRUNTIME` environment variable is not used by this package.
For an existing environment with all dependencies, use `pip install --no-deps .`
to retain its runtime and dependencies.


Download the assets from the upstream project. Point `AI_MODELS_ASSETS` or
`--assets` to the XiHe directory.
The following flat layout is supported:

```text
xihe/
  xihe_1to22_1day.onnx
  xihe_23to33_1day.onnx
  ...                             # one pair for each requested day, 1..10
  normalize_mean_50.npz
  normalize_std_50.npz
  mask_surface.npy
  mask_deep.npy
  input_data/
    input_surface_data/input_surface_20190101.npy
    input_deep_data/input_deep_20190101.npy
```

`--download-assets` is not implemented.


## Run

```bash
AI_MODELS_ASSETS=/path/to/models/xihe \
ai-models --lead-time 24 --date 20190101 \
  --file /path/to/models/xihe/input_data \
  --path ./xihe_output xihe
```

`--lead-time` is in **hours**, a positive multiple of 24 up to 240. XiHe predicts
the target day directly: `--lead-time 168` uses the day-7 pair once. Add
`--all-leads` to run each independent pair for days 1..7 from the same initial
condition. This is not an autoregressive rollout with the day-1 model.

To specify files directly, pass `--surface-file upper.npy --deep-file deep.npy`
and `--date YYYYMMDD`. `--path` is an output directory, default `xihe_output`. Each selected horizon
writes `xihe_YYYYMMDD_leadNday.nc`. Files are
NetCDF, not GRIB. Use `--overwrite` to replace existing output or `--output none`
to skip writing.

```bash
ai-models xihe --help-xihe
```
