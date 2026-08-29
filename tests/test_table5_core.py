from __future__ import annotations

import unittest

from table5_experiment import build_candidate_bank, generate_case, run_table5_experiment


class Table5CoreTests(unittest.TestCase):
    def test_frozen_bank_truth_ids(self) -> None:
        bank, truth_ids = build_candidate_bank()
        self.assertEqual(len(bank), 240)
        self.assertEqual(truth_ids["oscillator"], "cand_0454")
        self.assertEqual(truth_ids["lorentz"], "cand_0446")
        self.assertEqual(truth_ids["blackbody"], "cand_0596")

    def test_generated_roles_are_nonempty(self) -> None:
        frame = generate_case("oscillator", 0.8, 20260225)
        self.assertEqual(len(frame), 600)
        self.assertEqual(set(frame["split"]), {"disc", "diag", "audit", "cal", "test"})
        self.assertTrue((frame.groupby("split").size() > 0).all())

    def test_all_displayed_records_and_five_seed_gate(self) -> None:
        table5, all_runs, gate, _ = run_table5_experiment()
        self.assertEqual(len(table5), 3)
        self.assertEqual(len(all_runs), 15)
        self.assertEqual(list(gate["gate"]), ["5/5", "5/5", "5/5"])
        expected = {
            "oscillator": (
                "cand_0454",
                r"-0.07955 + 2.209\sin(1.1x_1) + 0.9181\cos(x_2) - 0.2409x_3",
                0.08322072567853776,
            ),
            "lorentz": (
                "cand_0446",
                r"-0.04502 + 1.002\frac{x_1x_2}{1+x_3^2}",
                0.03847661601467216,
            ),
            "blackbody": (
                "cand_0596",
                r"-0.08285 + 1.004\frac{x_1^3}{\exp(\operatorname{clip}(x_2,0,5))-1+0.01} + 0.007279x_3",
                0.005397846227991727,
            ),
        }
        for row in table5.to_dict(orient="records"):
            candidate_id, formula, nrmse = expected[row["generator"]]
            self.assertEqual(row["selected_candidate_id"], candidate_id)
            self.assertEqual(row["eidos_formula"], formula)
            self.assertAlmostEqual(row["semantic_nrmse"], nrmse, places=12)


if __name__ == "__main__":
    unittest.main()
