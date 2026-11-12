import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from eidos import EIDOSRegressor
from eidos._io import canonical_bytes
from eidos.protocol.strict import five_role_split_by_env


class EstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(91)
        cls.X = pd.DataFrame(rng.uniform(-2, 2, (100, 2)), columns=["temperature", "flow"])
        cls.y = 10 + 3 * cls.X.temperature.to_numpy() + rng.normal(0, 0.06, len(cls.X))
        cls.env = np.repeat(np.arange(5), 20)
        cls.roles = five_role_split_by_env(cls.env, seed=42)
        cls.model = EIDOSRegressor().fit(cls.X, cls.y, environments=cls.env, roles=cls.roles)

    def test_holdout_performance_and_original_units(self):
        self.assertLess(self.model.metrics_["test_rmse"], 0.3)
        self.assertEqual(sum(self.model.metrics_["role_counts"].values()), len(self.X))
        self.assertEqual(self.model.predict_interval(self.X).shape, (100, 2))
        prediction = self.model.predict(self.X)
        self.assertGreater(prediction.mean(), 5)

    def test_later_responses_cannot_change_constructed_model(self):
        replaced = self.y.copy()
        later = np.isin(self.roles, ["audit", "cal", "test"])
        replaced[later] = replaced[later][::-1] + 30
        changed = EIDOSRegressor().fit(self.X, replaced, environments=self.env, roles=self.roles)
        self.assertEqual(canonical_bytes(self.model.record_), canonical_bytes(changed.record_))
        self.assertFalse(np.array_equal(self.model.predict_interval(self.X), changed.predict_interval(self.X)))

    def test_named_columns_and_array_order(self):
        expected = self.model.predict(self.X)
        np.testing.assert_array_equal(expected, self.model.predict(self.X[["flow", "temperature"]]))
        np.testing.assert_array_equal(expected, self.model.predict(self.X.to_numpy()))
        with self.assertRaisesRegex(ValueError, "Missing model features"):
            self.model.predict(self.X[["flow"]])

    def test_model_loads_in_an_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "model.json"
            self.model.save(path)
            self.X.to_csv(root / "input.csv", index=False)
            loaded = EIDOSRegressor.load(path)
            np.testing.assert_array_equal(self.model.predict(self.X), loaded.predict(self.X))
            np.testing.assert_array_equal(self.model.predict_interval(self.X), loaded.predict_interval(self.X))
            code = (
                "import numpy as np,pandas as pd; from eidos import EIDOSRegressor; "
                "m=EIDOSRegressor.load('model.json'); "
                "np.save('output.npy',m.predict(pd.read_csv('input.csv',float_precision='round_trip')))"
            )
            result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            np.testing.assert_array_equal(self.model.predict(self.X), np.load(root / "output.npy"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["record"]["parameters"][0] = "0x0.0p+0"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                EIDOSRegressor.load(path)

    def test_invalid_input_fails_early(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            EIDOSRegressor().fit(self.X.assign(flow=np.nan), self.y)
        with self.assertRaisesRegex(ValueError, "roles"):
            EIDOSRegressor().fit(self.X, self.y, roles=["disc"] * len(self.X))
        with self.assertRaisesRegex(ValueError, "80"):
            EIDOSRegressor().fit(self.X.head(20), self.y[:20])
        with self.assertRaisesRegex(ValueError, "alpha"):
            EIDOSRegressor(alpha=0)

    def test_infinite_interval_survives_serialization(self):
        model = EIDOSRegressor(alpha=1e-8).fit(self.X, self.y, environments=self.env, roles=self.roles)
        self.assertTrue(np.isinf(model.interval_radius_))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            model.save(path)
            self.assertTrue(np.isinf(EIDOSRegressor.load(path).predict_interval(self.X)).all())


if __name__ == "__main__":
    unittest.main()
