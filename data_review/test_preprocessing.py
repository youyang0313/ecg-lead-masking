"""Focused safeguards for channel identity, invalid inputs and population scaling."""
from pathlib import Path
import tempfile
import unittest

from data_common import LEADS
from preprocess_development import RunningMoments, normalize, read_standard
import numpy as np
import wfdb


class PreprocessingSafeguards(unittest.TestCase):
    def test_population_moments_against_direct_calculation(self):
        # Distinct record offsets expose averaging per-record standard deviations
        # instead of computing a cohort population standard deviation.
        rng = np.random.RandomState(19)
        records = [rng.normal(size=(12, n)).astype(np.float32)+offset
                   for n, offset in [(1000, -3), (750, 20), (250, 7)]]
        moments = RunningMoments()
        for x in records:
            moments.update(x)
        combined = np.concatenate(records, axis=1).astype(np.float64)
        np.testing.assert_allclose(moments.mean, combined.mean(axis=1), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(moments.std(), combined.std(axis=1, ddof=0), atol=1e-12, rtol=1e-12)
        self.assertEqual(moments.count, 2000)
        z = normalize(combined.astype(np.float32), moments.mean, moments.std())
        np.testing.assert_allclose(z.mean(axis=1, dtype=np.float64), 0, atol=1e-7)
        np.testing.assert_allclose(z.std(axis=1, dtype=np.float64), 1, atol=1e-7)

    def test_zero_variance_floor_is_finite(self):
        signal = np.full((12, 1000), 2.0, dtype=np.float32)
        moments = RunningMoments()
        moments.update(signal)
        z = normalize(signal, moments.mean, np.maximum(moments.std(), 1e-6))
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue((z == 0).all())

    def test_reader_uses_names_and_rejects_invalid_header_or_samples(self):
        canonical = (np.arange(12)[:, None]+np.arange(1000)[None, :]/1000).astype(np.float32)
        order = [7, 0, 11, 1, 2, 8, 3, 9, 4, 10, 5, 6]
        with tempfile.TemporaryDirectory(prefix="ntudl_preprocessing_") as directory:
            wfdb.wrsamp("fixture", fs=100, units=["mV"]*12,
                        sig_name=[LEADS[c].upper() for c in order],
                        p_signal=canonical[order].T.astype(np.float64),
                        fmt=["16"]*12, adc_gain=[1000.0]*12, baseline=[0]*12,
                        write_dir=directory)
            stem = Path(directory)/"fixture"
            signal, _ = read_standard(stem)
            np.testing.assert_array_equal(signal, canonical)
            header = stem.with_suffix(".hea")
            original_header = header.read_text(encoding="ascii")
            header.write_text(original_header.replace(" 100 ", " 200 ", 1), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "sampling"):
                read_standard(stem)
            header.write_text(original_header.replace("/mV", "/V"), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "units"):
                read_standard(stem)
            header.write_text(original_header.replace(" V2\n", " I\n", 1), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_standard(stem)
            header.write_text(original_header, encoding="ascii")
            data = stem.with_suffix(".dat")
            raw = bytearray(data.read_bytes())
            raw[:2] = b"\x00\x80"  # WFDB format-16 missing-sample sentinel.
            data.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                read_standard(stem)


if __name__ == "__main__":
    unittest.main(verbosity=2)
