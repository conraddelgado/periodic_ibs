"""Full-ensemble, normalized load histograms without snapshot averaging."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from .__main__ import PDF_FIELDS, validate
from .pdfs import PDF_ARRAYS, particle_pdfs
from .test_volume_filter import config


class ParticlePDFTests(unittest.TestCase):
    def test_global_bins_preserve_unequal_particle_counts_and_extreme_samples(self):
        fields = ("particle_hydro_force_x", "particle_hydro_torque_z")
        groups = [
            np.array([[-2.0, 4.0]]),
            np.array([[1.0, 3.0], [2.0, 5.0], [3.0, 6.0]]),
            np.array([[50.0, 9.0], [4.0, 7.0]]),
        ]
        records = [{"particle_load_samples": values} for values in groups]
        arrays, metadata = particle_pdfs(records, fields, 4)
        pooled = np.concatenate(groups)
        for column, field in enumerate(fields):
            prefix = f"{field}.pdf."
            self.assertEqual(set(key[len(prefix):] for key in arrays if key.startswith(prefix)), set(PDF_ARRAYS))
            edges, counts = arrays[prefix + "bin_edges"], arrays[prefix + "counts"]
            expected, _ = np.histogram(pooled[:, column], bins=edges)
            assert_array_equal(arrays[prefix + "samples"], pooled[:, column])
            assert_array_equal(counts, expected)
            self.assertEqual(counts.dtype, np.int64)
            self.assertEqual(counts.sum(), 6)
            self.assertEqual(arrays[prefix + "sample_count"].item(), 6)
            self.assertEqual(metadata[field]["finite_samples"], 6)
            assert_allclose(arrays[prefix + "density"], counts / (6 * np.diff(edges)))
            assert_allclose(np.sum(arrays[prefix + "density"] * np.diff(edges)), 1)
            self.assertLessEqual(edges[0], pooled[:, column].min())
            self.assertGreaterEqual(edges[-1], pooled[:, column].max())
        # A histogram of the three snapshot means would have only three samples.
        self.assertEqual(arrays[fields[0] + ".pdf.counts"].sum(), sum(len(group) for group in groups))

    def test_constant_components_are_explicit_point_masses_and_missing_samples_are_counted(self):
        fields = ("particle_force_z", "particle_torque_z")
        records = [{"particle_load_samples": np.array([[0.0, 7.0], [0.0, 7.0], [np.nan, np.inf]])}]
        arrays, metadata = particle_pdfs(records, fields, 100)
        for field, value in zip(fields, (0.0, 7.0)):
            prefix = f"{field}.pdf."
            self.assertTrue(metadata[field]["point_mass"])
            self.assertEqual(metadata[field]["constant_value"], value)
            self.assertEqual(metadata[field]["excluded_nonfinite_samples"], 1)
            assert_array_equal(arrays[prefix + "counts"], [2])
            self.assertEqual(arrays[prefix + "sample_count"].item(), 2)
            assert_allclose(np.sum(arrays[prefix + "density"] * np.diff(arrays[prefix + "bin_edges"])), 1)
        empty, metadata = particle_pdfs([], fields, 8)
        for field in fields:
            self.assertEqual(empty[field + ".pdf.sample_count"].item(), 0)
            self.assertTrue(np.all(np.isnan(empty[field + ".pdf.density"])))
            self.assertFalse(metadata[field]["point_mass"])

    def test_default_and_invalid_bin_settings(self):
        cfg = config()
        validate(cfg)
        self.assertEqual(cfg["particle_pdf_bins"], 100)
        self.assertEqual(len(PDF_FIELDS), 12)
        for value in (0, -1, True, 2.5, "100", None):
            cfg["particle_pdf_bins"] = value
            with self.assertRaisesRegex(ValueError, "particle_pdf_bins"):
                validate(cfg)
        with self.assertRaisesRegex(ValueError, "sample array"):
            particle_pdfs([{"particle_load_samples": np.zeros((2, 3))}], ("particle_force_x",), 8)


if __name__ == "__main__":
    unittest.main()
