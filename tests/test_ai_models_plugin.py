"""CLI integration test with tiny day-specific ONNXs, no real weights required.

Run: python -m unittest discover -s tests -p test_ai_models_plugin.py
Requires onnx for fixture generation.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import xarray as xr
import yaml

from ai_models_xihe.model import DATA_DIR


class PluginTest(unittest.TestCase):
    def test_framework_thread_option_is_preserved(self):
        code = (
            "import sys; from ai_models_xihe.model import XiHe; "
            "from ai_models.__main__ import main; "
            "XiHe.print_assets_list=lambda self: print(self.num_threads); "
            "sys.argv=['ai-models','--num-threads','8','--assets-list','xihe']; main()"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "8")

    def test_direct_horizons_reordering_normalization_and_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            h, w = 3, 4
            config = yaml.safe_load((DATA_DIR / "data.yaml").read_text())["data"]
            keys = set(config["variables_1to22"] + config["variables_23to33"])
            mean, std = np.float32(7.123), np.float32(3.1)
            np.savez(root / "normalize_mean_50.npz", **{key: [float(mean) + 273.15 if key == "sst" else mean] for key in keys})
            np.savez(root / "normalize_std_50.npz", **{key: [std] for key in keys})
            np.save(root / "mercator_lat.npy", np.arange(h))
            np.save(root / "mercator_lon.npy", np.arange(w))
            expected = {}
            for group, filename, maskfile in (("1to22", "upper.npy", "mask_surface.npy"), ("23to33", "deep.npy", "mask_deep.npy")):
                raw_keys, inputs, outputs = (config[f"{kind}_{group}"] for kind in ("variables", "input", "output"))
                # Real upstream inputs are float16; arithmetic must promote to FP32.
                raw = np.broadcast_to(np.arange(len(raw_keys), dtype=np.float16)[None, :, None, None] + 20, (1, len(raw_keys), h, w)).copy()
                raw[:, :, 0, 0] = np.nan
                raw[:, raw_keys.index(outputs[0]), 0, 1] = -32767
                np.save(root / filename, raw)
                mask = np.zeros((1, len(outputs), h, w), dtype=bool)
                mask[:, :, 0, 0] = True
                np.save(root / maskfile, mask)
                expected[group] = np.stack([raw[:, raw_keys.index(key)] for key in outputs], axis=1).astype(np.float32)
                expected[group][:, 0, 0, 1] = mean
                indices = np.array([inputs.index(key) for key in outputs], dtype=np.int64)
                for day in (1, 2):
                    graph = helper.make_graph([
                        helper.make_node("Gather", ["input", "indices"], ["selected"], axis=1),
                        helper.make_node("Add", ["selected", "day"], ["output"])], "direct-day-fixture",
                        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, len(inputs), h, w])],
                        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, len(outputs), h, w])],
                        [numpy_helper.from_array(indices, "indices"), numpy_helper.from_array(np.array(day, dtype=np.float32), "day")])
                    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
                    onnx.checker.check_model(model)
                    onnx.save(model, root / f"xihe_{group}_{day}day.onnx")
            (root / "input.json").write_text(json.dumps({"surface_file": "upper.npy", "deep_file": "deep.npy", "init_date": "20190101"}))

            def run(path, *extra):
                result = subprocess.run([sys.executable, "-m", "ai_models", "--assets", str(root), "--lead-time", "48", "--file", str(root / "input.json"), "--path", str(path), "xihe", "--provider", "cpu", *extra], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            run(root / "target")
            self.assertEqual(len(list((root / "target").glob("*.nc"))), 1)
            run(root / "all", "--all-leads")
            self.assertEqual(len(list((root / "all").glob("*.nc"))), 2)
            for day in (1, 2):
                merged = np.concatenate([expected[g] for g in ("1to22", "23to33")], axis=1) + std * day
                with xr.open_dataset(root / "all" / f"xihe_2019010{day + 1}_lead{day}day.nc") as ds:
                    self.assertEqual(set(ds.data_vars), {"thetao", "so", "uo", "vo", "zos", "sst"})
                    self.assertEqual(ds.sizes["depth"], 23)
                    self.assertEqual(ds.attrs["lead_time_hours"], day * 24)
                    self.assertEqual(str(ds.time.values[0])[:10], f"2019-01-0{day + 1}")
                    for name, offset in (("thetao", 2), ("so", 3), ("uo", 4), ("vo", 5)):
                        np.testing.assert_allclose(ds[name].values, merged[:, offset::4], atol=3e-5, equal_nan=True)
                    np.testing.assert_allclose(ds.zos.values, merged[:, 0], atol=3e-5, equal_nan=True)
                    np.testing.assert_allclose(ds.sst.values, merged[:, 1], atol=3e-5, equal_nan=True)
            with xr.open_dataset(root / "target/xihe_20190103_lead2day.nc") as a, xr.open_dataset(root / "all/xihe_20190103_lead2day.nc") as b:
                xr.testing.assert_equal(a, b)


if __name__ == "__main__":
    unittest.main()
