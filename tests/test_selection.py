"""Unit tests for the :mod:`selection` module."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from exceptions import SelectionError
from selection import IdealFunctionSelector


class IdealFunctionSelectorTest(unittest.TestCase):
    """Test the least-squares selection of the four ideal functions."""

    def _frames(
        self,
        training_values: dict[str, list[float]],
        ideal_values: dict[int, list[float]],
        x_values: list[float],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build training and ideal frames from plain dictionaries."""
        training = pd.DataFrame({"x": x_values, **training_values})
        ideal = pd.DataFrame({"x": x_values})
        for number, values in ideal_values.items():
            ideal[f"y{number}"] = values
        return training, ideal

    def test_perfect_fit_selects_exact_functions(self) -> None:
        """Exact matches are recovered with a least-squares error of zero."""
        x = [0.0, 1.0, 2.0, 3.0]
        training = {
            "y1": [0.0, 1.0, 4.0, 9.0],   # x**2
            "y2": [1.0, 2.0, 4.0, 8.0],   # 2**x
            "y3": [0.0, 1.0, 2.0, 3.0],   # x
            "y4": [5.0, 5.0, 5.0, 5.0],   # constant 5
        }
        ideal = {
            3: [0.0, 1.0, 4.0, 9.0],      # exact match for y1
            7: [1.0, 2.0, 4.0, 8.0],      # exact match for y2
            12: [0.0, 1.0, 2.0, 3.0],     # exact match for y3
            25: [5.0, 5.0, 5.0, 5.0],     # exact match for y4
        }
        # Fill the remaining 46 ideal functions with clearly different
        # values so that the exact matches are unambiguous.
        for number in range(1, 51):
            if number not in ideal:
                ideal[number] = [100.0 + number] * 4

        training_frame, ideal_frame = self._frames(training, ideal, x)
        selector = IdealFunctionSelector()
        results = selector.run(training_frame, ideal_frame)

        self.assertEqual(len(results), 4)
        expected = {1: 3, 2: 7, 3: 12, 4: 25}
        for result in results:
            self.assertEqual(
                result.ideal_func_no,
                expected[result.training_func_no],
                f"Training function {result.training_func_no} was mapped "
                f"to ideal {result.ideal_func_no} instead of "
                f"{expected[result.training_func_no]}.",
            )
            self.assertAlmostEqual(result.sse, 0.0, places=6)
            self.assertAlmostEqual(result.max_deviation, 0.0, places=6)
            self.assertAlmostEqual(result.threshold, 0.0, places=6)

    def test_max_deviation_and_threshold_computation(self) -> None:
        """max_deviation and threshold are computed from the deviations."""
        x = [0.0, 1.0, 2.0]
        training = {"y1": [0.0, 1.0, 2.0], "y2": [0.0, 1.0, 2.0],
                    "y3": [0.0, 1.0, 2.0], "y4": [0.0, 1.0, 2.0]}
        # ideal y5 = y_train + [0, 0, 2] -> max deviation 2, SSE 4.
        ideal = {5: [0.0, 1.0, 4.0]}
        for number in range(1, 51):
            if number != 5:
                ideal[number] = [50.0] * 3

        training_frame, ideal_frame = self._frames(training, ideal, x)
        results = IdealFunctionSelector().run(training_frame, ideal_frame)

        result = next(r for r in results if r.training_func_no == 1)
        self.assertEqual(result.ideal_func_no, 5)
        self.assertAlmostEqual(result.sse, 4.0, places=6)
        self.assertAlmostEqual(result.max_deviation, 2.0, places=6)
        self.assertAlmostEqual(result.threshold, np.sqrt(2.0) * 2.0, places=6)

    def test_missing_columns_raise_selection_error(self) -> None:
        """A frame without the expected columns raises SelectionError."""
        training = pd.DataFrame({"x": [0.0], "y1": [1.0]})
        ideal = pd.DataFrame({"x": [0.0]})
        with self.assertRaises(SelectionError):
            IdealFunctionSelector().run(training, ideal)

    def test_disjoint_x_values_raise_selection_error(self) -> None:
        """Disjoint x-grids raise SelectionError."""
        training = pd.DataFrame(
            {"x": [0.0, 1.0], "y1": [1.0, 2.0], "y2": [1.0, 2.0],
             "y3": [1.0, 2.0], "y4": [1.0, 2.0]}
        )
        ideal = pd.DataFrame({"x": [5.0, 6.0]})
        for number in range(1, 51):
            ideal[f"y{number}"] = [1.0, 2.0]

        with self.assertRaises(SelectionError):
            IdealFunctionSelector().run(training, ideal)

    def test_custom_mapping_tolerance_is_used(self) -> None:
        """The mapping tolerance is configurable."""
        x = [0.0, 1.0, 2.0]
        training = {"y1": [0.0, 1.0, 2.0], "y2": [0.0, 1.0, 2.0],
                    "y3": [0.0, 1.0, 2.0], "y4": [0.0, 1.0, 2.0]}
        ideal = {5: [0.0, 1.0, 2.0]}
        for number in range(1, 51):
            if number != 5:
                ideal[number] = [50.0] * 3

        training_frame, ideal_frame = self._frames(training, ideal, x)
        results = IdealFunctionSelector(mapping_tolerance=3.0).run(
            training_frame, ideal_frame
        )
        result = next(r for r in results if r.training_func_no == 1)
        self.assertAlmostEqual(result.threshold, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
