"""XiHe direct, day-specific two-network forecasts as an ai-models plugin."""

import argparse
import gc
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from ._common import OceanModel, check_shape, check_wet_finite, tensor_dtype

LOG = logging.getLogger(__name__)
DATA_DIR = Path(__file__).with_name("data")
DEPTHS = np.array([
    0.4940, 2.6457, 5.0782, 7.9296, 11.4050, 15.8101, 21.5988, 29.4447,
    40.3441, 55.7643, 77.8539, 92.3261, 109.7293, 130.6660, 155.8507,
    186.1256, 222.4752, 266.0403, 318.1274, 380.2130, 453.9377, 541.0889, 643.5668,
], dtype=np.float32)


class XiHe(OceanModel):
    model_name = "xihe"
    netcdf_output_name = "xihe-netcdf"
    expver = "xihe"
    hour_steps = 24
    download_files = []

    def parse_model_args(self, args):
        parser = argparse.ArgumentParser("ai-models xihe", add_help=False)
        parser.add_argument("--help-xihe", action="help")
        self.add_common_arguments(parser)
        parser.add_argument("--input-path", "--input_path", help="Native input_surface_data/input_deep_data directory")
        parser.add_argument("--surface-file", help="Raw upper-layer NPY (1,52,lat,lon)")
        parser.add_argument("--deep-file", help="Raw deeper-layer NPY (1,48,lat,lon)")
        parser.add_argument("--all-leads", action="store_true", help="Run each day-specific pair from day 1 through --lead-time")
        return parser.parse_args(args)

    def _days(self):
        days = self.forecast_days(self.lead_time)
        if days > 10:
            raise ValueError("Upstream XiHe supplies day-specific weights for 1..10 days (24..240 hours)")
        return list(range(1, days + 1)) if self.all_leads else [days]

    @property
    def asset_files(self):
        root = Path(self.assets).expanduser()
        names = [f"xihe_{group}_{day}day.onnx" for day in self._days() for group in ("1to22", "23to33")]
        names.extend(("normalize_mean_50.npz", "normalize_std_50.npz", "mask_surface.npy", "mask_deep.npy"))
        return [str(self.find_asset(name, optional=True) or root / name) for name in names]

    def print_fields(self):
        print("XiHe: raw surface/deep NPY pair, native input directory or JSON manifest in --file.")
        print("Upper raw order: zos,u,v,sst, then 12 interleaved thetao/so/uo/vo levels (52 channels).")
        print("Deep raw order: zos,u,v,sst, then 11 interleaved thetao/so/uo/vo levels (48 channels).")
        print("--lead-time is HOURS. A day-specific pair predicts that horizon directly; --all-leads runs each pair.")

    def _input_files(self):
        date = None
        surface, deep = self.surface_file, self.deep_file
        source = self.input_path or self.file
        if source and Path(source).suffix.lower() == ".json":
            path = Path(source).expanduser()
            manifest = json.loads(path.read_text(encoding="utf-8"))
            surface = surface or manifest.get("surface_file", manifest.get("surface"))
            deep = deep or manifest.get("deep_file", manifest.get("deep"))
            if not surface or not deep:
                raise ValueError("XiHe manifest requires surface_file and deep_file")
            surface = Path(surface) if Path(surface).is_absolute() else path.parent / surface
            deep = Path(deep) if Path(deep).is_absolute() else path.parent / deep
            date = manifest.get("init_date", manifest.get("date"))
        elif source and Path(source).is_dir():
            root = Path(source).expanduser()
            if int(self.date) > 0:
                date = str(self.date)
            else:
                matches = list((root / "input_surface_data").glob("input_surface_*.npy"))
                if len(matches) == 1:
                    match = re.search(r"(\d{8})\.npy$", matches[0].name)
                    date = match.group(1) if match else None
            if date is None:
                raise ValueError("Select the NPY initial date using --date YYYYMMDD")
            surface = surface or root / "input_surface_data" / f"input_surface_{date}.npy"
            deep = deep or root / "input_deep_data" / f"input_deep_{date}.npy"
        elif source and Path(source).suffix.lower() == ".npy":
            surface = surface or source
        elif source:
            raise ValueError("XiHe --file accepts a JSON manifest, native input directory, or surface NPY with --deep-file")
        if not surface or not deep:
            raise ValueError("Provide both --surface-file/--deep-file or use --file with a manifest/native input directory")
        surface, deep = Path(surface).expanduser(), Path(deep).expanduser()
        if int(self.date) > 0:
            initial = self.initial_date()
        elif date is not None:
            initial = pd.to_datetime(str(date), format="%Y%m%d") if re.fullmatch(r"\d{8}", str(date)) else pd.Timestamp(date)
        else:
            match = re.search(r"(\d{8})\.npy$", surface.name)
            initial = self.initial_date(pd.to_datetime(match.group(1), format="%Y%m%d") if match else None)
        return surface, deep, initial

    @staticmethod
    def _normalization(keys, means, stds):
        mean, std = [], []
        for key in keys:
            if key not in means or key not in stds:
                raise ValueError(f"Normalization statistics are missing variable {key}")
            a, b = np.asarray(means[key]).reshape(-1), np.asarray(stds[key]).reshape(-1)
            if a.size != 1 or b.size != 1:
                raise ValueError(f"Expected scalar normalization statistics for {key}")
            mean.append(float(a[0]) - (273.15 if key == "sst" else 0))
            std.append(float(b[0]))
        mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Normalization mean/std must be finite with positive std")
        return mean, std

    def _mask(self, name, shape):
        mask = np.load(self.find_asset(name), mmap_mode="r")
        if mask.dtype != np.bool_:
            if not np.all((mask == 0) | (mask == 1)):
                raise ValueError(f"{name} must be a boolean/0-1 invalid-cell mask")
            mask = mask.astype(bool)
        try:
            return np.broadcast_to(mask, shape)
        except ValueError as exc:
            raise ValueError(f"{name} shape {mask.shape} does not match output {shape}") from exc

    def _predict_group(self, path, group, day, config, means, stds):
        raw = np.load(path, mmap_mode="r")
        raw_keys, keys = config[f"variables_{group}"], config[f"input_{group}"]
        if raw.ndim != 4 or raw.shape[0] != 1 or raw.shape[1] != len(raw_keys):
            raise ValueError(f"{path}: expected (1,{len(raw_keys)},lat,lon), got {raw.shape}")
        mean, std = self._normalization(keys, means, stds)
        data = np.empty((1, len(keys)) + raw.shape[-2:], dtype=np.float32)
        for index, key in enumerate(keys):
            # Upstream input NPYs can be FP16. Cast before the ufunc: NumPy 1.x
            # may otherwise do the subtraction in FP16 despite a FP32 out array.
            source = np.asarray(raw[:, raw_keys.index(key)], dtype=np.float32)
            invalid = ~np.isfinite(source) | (source < -30000)
            np.subtract(source, mean[index], out=data[:, index])
            np.divide(data[:, index], std[index], out=data[:, index])
            np.copyto(data[:, index], 0, where=invalid)
        session = self.create_session(self.find_asset(f"xihe_{group}_{day}day.onnx"))
        infos = session.get_inputs()
        if len(infos) != 1:
            raise ValueError("Each XiHe ONNX network must have one input")
        check_shape(infos[0], data.shape)
        prediction = session.run(None, {infos[0].name: np.asarray(data, dtype=tensor_dtype(infos[0]))})[0]
        del data, session
        gc.collect()
        outputs = config[f"output_{group}"]
        expected = (1, len(outputs)) + raw.shape[-2:]
        if prediction.shape != expected:
            raise ValueError(f"XiHe {group} output {prediction.shape}, expected {expected}")
        prediction = np.asarray(prediction, dtype=np.float32)
        out_mean, out_std = self._normalization(outputs, means, stds)
        prediction *= out_std[None, :, None, None]
        prediction += out_mean[None, :, None, None]
        invalid = self._mask("mask_surface.npy" if group == "1to22" else "mask_deep.npy", prediction.shape)
        check_wet_finite(prediction, invalid, f"XiHe {group}")
        np.copyto(prediction, np.nan, where=invalid)
        return prediction

    def _dataset(self, outputs, initial, day):
        merged = np.concatenate(outputs, axis=1)
        if merged.shape[1] != 94:
            raise ValueError(f"Merged XiHe output must have 94 channels, got {merged.shape}")
        latitude_file = self.find_asset("mercator_lat.npy", optional=True) or DATA_DIR / "mercator_lat.npy"
        longitude_file = self.find_asset("mercator_lon.npy", optional=True) or DATA_DIR / "mercator_lon.npy"
        lat, lon = np.load(latitude_file), np.load(longitude_file)
        if len(lat) != merged.shape[-2] or len(lon) != merged.shape[-1]:
            raise ValueError("Latitude/longitude assets do not match the XiHe output grid")
        valid = initial + pd.Timedelta(days=day)
        coords = {"time": [valid.to_datetime64()], "depth": DEPTHS, "latitude": lat.astype(np.float32), "longitude": lon.astype(np.float32)}
        ds = xr.Dataset(
            {name: (("time", "depth", "latitude", "longitude"), merged[:, start::4])
             for name, start in (("thetao", 2), ("so", 3), ("uo", 4), ("vo", 5))},
            coords=coords,
        )
        for name, index in (("zos", 0), ("sst", 1)):
            ds[name] = (("time", "latitude", "longitude"), merged[:, index])
        for name, unit in (("thetao", "degrees_C"), ("so", "1e-3"), ("uo", "m s-1"), ("vo", "m s-1"), ("zos", "m"), ("sst", "degrees_C")):
            ds[name].attrs["units"] = unit
        ds.attrs.update(model="XiHe", forecast_reference_time=initial.isoformat(), lead_time_hours=day * 24, forecast_method="direct day-specific model")
        return ds

    def run(self):
        days = self._days()
        # Fail before reading global arrays if a horizon's weights are absent.
        for day in days:
            for group in ("1to22", "23to33"):
                self.find_asset(f"xihe_{group}_{day}day.onnx")
        mean_path, std_path = self.find_asset("normalize_mean_50.npz"), self.find_asset("normalize_std_50.npz")
        for name in ("mask_surface.npy", "mask_deep.npy"):
            self.find_asset(name)
        config_path = self.find_asset("data.yaml", optional=True) or DATA_DIR / "data.yaml"
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["data"]
        with np.load(mean_path) as archive:
            means = {k: archive[k] for k in archive.files}
        with np.load(std_path) as archive:
            stds = {k: archive[k] for k in archive.files}
        surface, deep, initial = self._input_files()
        names = [f"xihe_{(initial + pd.Timedelta(days=day)).strftime('%Y%m%d')}_lead{day}day.nc" for day in days]
        if self.writes_netcdf:
            self.output.check_paths(names)
        for day, name in zip(days, names):
            LOG.info("XiHe direct forecast +%dh using two day-%d networks", day * 24, day)
            with self.timer(f"XiHe +{day * 24}h"):
                outputs = [self._predict_group(surface, "1to22", day, config, means, stds), self._predict_group(deep, "23to33", day, config, means, stds)]
                if outputs[0].shape[-2:] != outputs[1].shape[-2:]:
                    raise ValueError("Upper and deeper input grids differ")
                if self.writes_netcdf:
                    ds = self._dataset(outputs, initial, day)
                    del outputs
                    try:
                        self.output.write_dataset(ds, name)
                    finally:
                        ds.close()
                        del ds
