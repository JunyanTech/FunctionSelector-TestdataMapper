"""Unit tests for the :mod:`data_loader` module."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Make the project root importable when the tests are run with
# ``python -m unittest discover`` from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data_loader import DataLoader
from exceptions import DataValidationError


class DataLoaderTest(unittest.TestCase):
    """Test the loading and validation of the CSV input files."""

    def setUp(self) -> None:
        """Create a temporary directory with a valid minimal dataset."""
        self._temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._temp_dir.name)
        self._write_valid_files()

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        self._temp_dir.cleanup()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _write_valid_files(self) -> None:
        """Write train1-4.csv, ideal.csv and test.csv with valid content."""
        x_values = [0.0, 1.0, 2.0]
        for index in range(1, 5):
            frame = pd.DataFrame({"x": x_values, "y": [float(index)] * 3})
            frame.to_csv(self.data_dir / f"train{index}.csv", index=False)

        ideal = pd.DataFrame({"x": x_values})
        for number in range(1, 51):
            ideal[f"y{number}"] = [float(number)] * 3
        ideal.to_csv(self.data_dir / "ideal.csv", index=False)

        test = pd.DataFrame({"x": [0.5, 1.5], "y": [10.0, 11.0]})
        test.to_csv(self.data_dir / "test.csv", index=False)

    def _loader(self) -> DataLoader:
        """Return a DataLoader pointing at the temporary dataset."""
        return DataLoader(
            training_files=[
                self.data_dir / f"train{i}.csv" for i in range(1, 5)
            ],
            ideal_file=self.data_dir / "ideal.csv",
            test_file=self.data_dir / "test.csv",
        )

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------
    def test_load_training_data_merges_four_files(self) -> None:
        """The four training files are merged into five columns."""
        frame = self._loader().load_training_data()
        self.assertEqual(list(frame.columns), ["x", "y1", "y2", "y3", "y4"])
        self.assertEqual(len(frame), 3)
        # y-columns keep their per-file values after the merge.
        self.assertEqual(list(frame["y1"]), [1.0, 1.0, 1.0])
        self.assertEqual(list(frame["y4"]), [4.0, 4.0, 4.0])

    def test_load_training_data_sorts_by_x(self) -> None:
        """The merged training frame is sorted by x."""
        frame = self._loader().load_training_data()
        self.assertTrue((frame["x"].diff().dropna() > 0).all())

    def test_load_ideal_functions_has_51_columns(self) -> None:
        """The ideal frame has x plus fifty y-columns."""
        frame = self._loader().load_ideal_functions()
        self.assertEqual(len(frame.columns), 51)
        self.assertEqual(frame.columns[0], "x")
        self.assertEqual(frame.columns[-1], "y50")

    def test_load_test_data_has_two_columns(self) -> None:
        """The test frame has exactly x and y."""
        frame = self._loader().load_test_data()
        self.assertEqual(list(frame.columns), ["x", "y"])
        self.assertEqual(len(frame), 2)

    def test_missing_file_raises_file_not_found(self) -> None:
        """A missing input file raises FileNotFoundError."""
        loader = DataLoader(
            training_files=[self.data_dir / "nope1.csv"] * 4,
            ideal_file=self.data_dir / "ideal.csv",
            test_file=self.data_dir / "test.csv",
        )
        with self.assertRaises(FileNotFoundError):
            loader.load_training_data()

    def test_wrong_column_names_raise_data_validation_error(self) -> None:
        """A file without a y-column raises DataValidationError."""
        broken = pd.DataFrame({"x": [0.0], "z": [1.0]})
        broken.to_csv(self.data_dir / "bad.csv", index=False)
        loader = DataLoader(
            training_files=[self.data_dir / "bad.csv"] * 4,
            ideal_file=self.data_dir / "ideal.csv",
            test_file=self.data_dir / "test.csv",
        )
        with self.assertRaises(DataValidationError):
            loader.load_training_data()

    def test_wrong_ideal_column_count_raises(self) -> None:
        """An ideal file without fifty y-columns raises."""
        small = pd.DataFrame({"x": [0.0], "y1": [1.0], "y2": [2.0]})
        small.to_csv(self.data_dir / "small_ideal.csv", index=False)
        loader = DataLoader(
            training_files=[self.data_dir / "train1.csv"] * 4,
            ideal_file=self.data_dir / "small_ideal.csv",
            test_file=self.data_dir / "test.csv",
        )
        with self.assertRaises(DataValidationError):
            loader.load_ideal_functions()

    def test_non_numeric_values_raise(self) -> None:
        """Non-numeric values raise DataValidationError."""
        broken = pd.DataFrame({"x": [0.0, 1.0], "y": ["a", "b"]})
        broken.to_csv(self.data_dir / "text.csv", index=False)
        loader = DataLoader(
            training_files=[self.data_dir / "text.csv"] * 4,
            ideal_file=self.data_dir / "ideal.csv",
            test_file=self.data_dir / "test.csv",
        )
        with self.assertRaises(DataValidationError):
            loader.load_training_data()

    def test_wrong_number_of_training_files_raises(self) -> None:
        """Passing a wrong number of training files raises."""
        loader = DataLoader(
            training_files=[
                self.data_dir / "train1.csv",
                self.data_dir / "train2.csv",
            ],
            ideal_file=self.data_dir / "ideal.csv",
            test_file=self.data_dir / "test.csv",
        )
        with self.assertRaises(DataValidationError):
            loader.load_training_data()

    def test_combined_single_file_loads_four_columns(self) -> None:
        """A single combined file (x, y1..y4) is loaded as four sets."""
        combined = pd.DataFrame(
            {"x": [2.0, 0.0, 1.0], "y1": [1.0] * 3, "y2": [2.0] * 3,
             "y3": [3.0] * 3, "y4": [4.0] * 3}
        )
        combined.to_csv(self.data_dir / "combined.csv", index=False)
        loader = DataLoader(
            training_files=[self.data_dir / "combined.csv"],
            ideal_file=self.data_dir / "ideal.csv",
            test_file=self.data_dir / "test.csv",
        )
        frame = loader.load_training_data()
        self.assertEqual(list(frame.columns), ["x", "y1", "y2", "y3", "y4"])
        self.assertEqual(len(frame), 3)
        self.assertTrue((frame["x"].diff().dropna() > 0).all())
        self.assertEqual(list(frame["y1"]), [1.0, 1.0, 1.0])
        self.assertEqual(list(frame["y4"]), [4.0, 4.0, 4.0])

    def test_combined_file_with_wrong_column_count_raises(self) -> None:
        """A combined training file must contain exactly four y-columns."""
        broken = pd.DataFrame({"x": [0.0], "y1": [1.0], "y2": [2.0]})
        broken.to_csv(self.data_dir / "combined_bad.csv", index=False)
        loader = DataLoader(
            training_files=[self.data_dir / "combined_bad.csv"],
            ideal_file=self.data_dir / "ideal.csv",
            test_file=self.data_dir / "test.csv",
        )
        with self.assertRaises(DataValidationError):
            loader.load_training_data()


if __name__ == "__main__":
    unittest.main()
