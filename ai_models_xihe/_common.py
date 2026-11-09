"""Integration helpers; no changes to ai-models or the installed runtime."""

import argparse
import logging
import os
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from ai_models.model import Model
from ai_models.outputs import Output

LOG = logging.getLogger(__name__)


class NetCDFOutput(Output):
    def __init__(self, owner, path, metadata=None, **kwargs):
        self.owner = owner
        self.path = Path(path).expanduser()
        self.metadata = metadata or {}
        LOG.info("Writing ocean forecasts as NetCDF in %s", self.path)

    def check_paths(self, names):
        if not self.owner.overwrite:
            for name in names:
                path = self.path / name
                if path.exists():
                    raise FileExistsError(f"Output exists: {path}; use --overwrite to replace it")

    def write_dataset(self, dataset, name):
        self.check_paths([name])
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_netcdf(path, engine="netcdf4", unlimited_dims=["time"])
        LOG.info("Wrote %s", path)
        return path


class OceanModel(Model):
    """Use the real Model lifecycle with a model-specific NetCDF output plugin."""

    def __init__(self, **kwargs):
        mode = kwargs.get("output", "file")
        if mode not in ("file", "none", self.netcdf_output_name):
            raise ValueError("Ocean adapters support --output file, none, or their NetCDF output")
        if kwargs.get("archive_requests"):
            raise ValueError("GRIB/MARS archive requests are not supported for ocean NetCDF output")
        if mode == "file":
            kwargs["output"] = self.netcdf_output_name
        if kwargs.get("path") in (None, f"{self.model_name}.grib"):
            kwargs["path"] = f"{self.model_name}_output"
        # FileInput is lazy. Never initialize MARS retrieval for native ocean data.
        kwargs["input"] = "file"
        super().__init__(**kwargs)

    @property
    def writes_netcdf(self):
        return isinstance(self.output, NetCDFOutput)

    @staticmethod
    def add_common_arguments(parser):
        parser.add_argument("--provider", choices=["auto", "maca", "cuda", "cpu"], default="auto")
        parser.add_argument("--num-threads", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--overwrite", action="store_true", help="Replace existing forecast files")

    @cached_property
    def providers(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install an ONNX Runtime build for your device; C500 requires MACA ORT") from exc
        available = ort.get_available_providers()
        names = {"maca": "MACAExecutionProvider", "cuda": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}
        if self.provider == "auto":
            primary = next((p for p in names.values() if p in available), None)
        else:
            primary = names[self.provider]
            if primary not in available:
                raise RuntimeError(f"Requested {primary}; available providers: {available}")
        if primary is None or (self.only_gpu and primary == "CPUExecutionProvider"):
            raise RuntimeError(f"No requested GPU provider is available: {available}")
        result = [primary]
        if primary != "CPUExecutionProvider" and not self.only_gpu and "CPUExecutionProvider" in available:
            result.append("CPUExecutionProvider")
        LOG.info("ONNXRuntime providers: %s", result)
        return result

    def create_session(self, path):
        import onnxruntime as ort

        if self.num_threads < 1:
            raise ValueError("--num-threads must be positive")
        os.environ.setdefault("MACART_OP_KEEP_ONNX_PRECISION", "ON")
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = self.num_threads
        if self.only_gpu:
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        with self.timer(f"Loading {path}"):
            session = ort.InferenceSession(str(path), sess_options=options, providers=self.providers)
        LOG.info("Session providers: %s", session.get_providers())
        return session

    def find_asset(self, name, optional=False):
        root = Path(self.assets).expanduser()
        for candidate in (root / name, root / "src" / name, root / "models" / name):
            if candidate.is_file():
                return candidate
        if optional:
            return None
        raise FileNotFoundError(f"Missing {name} under asset directory {root}; see README.md")

    def download_assets(self, **kwargs):
        raise ValueError("Ocean model assets require the upstream downloads; see README.md")

    def print_requests(self, *args, **kwargs):
        raise ValueError("These ocean models use local native inputs, not MARS retrieval requests")

    def print_assets_list(self):
        for path in self.asset_files:
            print(path)

    @staticmethod
    def forecast_days(hours):
        if hours <= 0 or hours % 24:
            raise ValueError("--lead-time is in hours and must be a positive multiple of 24")
        return hours // 24

    def initial_date(self, fallback=None):
        if fallback is not None:
            return pd.Timestamp(fallback)
        if int(self.date) > 0:
            return pd.to_datetime(str(self.date), format="%Y%m%d")
        raise ValueError("This input has no time coordinate; provide --date YYYYMMDD")


def tensor_dtype(info):
    mapping = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64}
    if info.type not in mapping:
        raise ValueError(f"Unsupported model input type {info.name}: {info.type}")
    return mapping[info.type]


def check_shape(info, actual):
    if len(info.shape) != len(actual) or any(
        isinstance(expected, int) and expected > 0 and expected != got
        for expected, got in zip(info.shape, actual)
    ):
        raise ValueError(f"{info.name} expects {info.shape}, received {tuple(actual)}")


def check_wet_finite(array, invalid, label):
    """Bound temporary memory by one spatial field on the full global grid."""
    masks = np.broadcast_to(invalid, array.shape)
    for values, mask in zip(array.reshape((-1,) + array.shape[-2:]), masks.reshape((-1,) + masks.shape[-2:])):
        if np.any(~np.isfinite(values) & ~mask):
            raise ValueError(f"{label} contains nonfinite values over ocean cells")
