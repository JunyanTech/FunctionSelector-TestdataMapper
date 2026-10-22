"""Unit tests for the :mod:`mapping` module."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from exceptions import MappingError
from mapping import TestDataMapper, summarise_mapping
from selection import SelectionResult


def _selection(
    ideal_func_no: int,
    threshold: float,
    training_func_no: int = 1,
    sse: float = 0.0,
    max_deviation: float = 1.0,
) -> SelectionResult:
    """Create a selection result with explicit threshold."""
    return SelectionResult(
        training_func_no=training_func_no,
        ideal_func_no=ideal_func_no,
        sse=sse,
        max_deviation=max_deviation,
        threshold=threshold,
    )


class TestDataMapperTest(unittest.TestCase):
    """Test the mapping of test points with the sqrt(2) criterion."""

    def setUp(self) -> None:
        """Prepare an ideal frame with two relevant ideal functions."""
        # y5 follows x; y7 follows 2*x.
        self.ideal_frame = pd.DataFrame(
            {"x": [0.0, 1.0, 2.0], "y5": [0.0, 1.0, 2.0], "y7": [0.0, 2.0, 4.0]}
        )
        self.selections = [
            _selection(ideal_func_no=5, threshold=1.0),
            _selection(ideal_func_no=7, threshold=0.5),
        ]
        self.mapper = TestDataMapper(self.selections)

    def test_point_is_assigned_to_min_deviation_ideal(self) -> None:
        """A test point is mapped to the ideal function with the smallest
        deviation if the criterion is fulfilled."""
        test = pd.DataFrame({"x": [1.0], "y": [1.1]})
        assignments, unmapped = self.mapper.run(test, self.ideal_frame)

        self.assertEqual(unmapped, 0)
        self.assertEqual(len(assignments), 1)
        assignment = assignments[0]
        self.assertEqual(assignment.ideal_func_no, 5)   # deviation 0.1
        self.assertAlmostEqual(assignment.delta_y, 0.1, places=6)

    def test_point_at_exact_threshold_is_assigned(self) -> None:
        """A deviation exactly equal to the threshold is still accepted."""
        test = pd.DataFrame({"x": [1.0], "y": [2.0]})
        assignments, unmapped = self.mapper.run(test, self.ideal_frame)

        # y7 at x=1 is 2.0 -> deviation 0, ideal 7 is chosen.
        self.assertEqual(unmapped, 0)
        self.assertEqual(assignments[0].ideal_func_no, 7)
        self.assertAlmostEqual(assignments[0].delta_y, 0.0, places=6)

    def test_point_exceeding_all_thresholds_is_unmapped(self) -> None:
        """A point that violates the criterion for every ideal function
        is left unmapped."""
        test = pd.DataFrame({"x": [1.0], "y": [3.5]})
        assignments, unmapped = self.mapper.run(test, self.ideal_frame)

        # deviations: y5 -> 2.5 (>1.0), y7 -> 1.5 (>0.5)
        self.assertEqual(assignments, [])
        self.assertEqual(unmapped, 1)

    def test_interpolation_for_off_grid_x(self) -> None:
        """Test points whose x is not on the ideal grid are handled via
        linear interpolation."""
        ideal = pd.DataFrame(
            {"x": [0.0, 2.0, 4.0], "y5": [0.0, 2.0, 4.0]}
        )
        mapper = TestDataMapper([_selection(ideal_func_no=5, threshold=1.0)])
        test = pd.DataFrame({"x": [1.0], "y": [1.05]})
        assignments, unmapped = mapper.run(test, ideal)

        self.assertEqual(unmapped, 0)
        self.assertEqual(assignments[0].ideal_func_no, 5)
        self.assertAlmostEqual(assignments[0].delta_y, 0.05, places=6)

    def test_restricted_x_range_skips_outside_points(self) -> None:
        """With the optional x-range restriction, points outside the
        training range are not mapped."""
        mapper = TestDataMapper(
            self.selections, restrict_to_training_x_range=True
        )
        test = pd.DataFrame({"x": [100.0], "y": [100.0]})
        assignments, unmapped = mapper.run(test, self.ideal_frame)

        self.assertEqual(assignments, [])
        self.assertEqual(unmapped, 1)

    def test_empty_selections_raise_mapping_error(self) -> None:
        """A mapper without selections raises MappingError."""
        with self.assertRaises(MappingError):
            TestDataMapper([])

    def test_missing_test_columns_raise_mapping_error(self) -> None:
        """A test frame without x/y raises MappingError."""
        test = pd.DataFrame({"z": [1.0]})
        with self.assertRaises(MappingError):
            self.mapper.run(test, self.ideal_frame)

    def test_missing_ideal_column_raises_mapping_error(self) -> None:
        """A missing ideal column raises MappingError."""
        ideal = pd.DataFrame({"x": [0.0, 1.0]})  # no y5 / y7
        test = pd.DataFrame({"x": [1.0], "y": [1.0]})
        with self.assertRaises(MappingError):
            self.mapper.run(test, ideal)

    def test_summarise_mapping_statistics(self) -> None:
        """The summary reports counts, rate and deviation figures."""
        test = pd.DataFrame({"x": [1.0, 1.0, 2.0], "y": [1.1, 1.05, 4.1]})
        assignments, unmapped = self.mapper.run(test, self.ideal_frame)
    
        summary = summarise_mapping(assignments, unmapped)
        self.assertEqual(summary["mapped_points"], len(assignments))
        self.assertEqual(
            summary["mapped_points"] + summary["unmapped_points"], 3
        )
        self.assertAlmostEqual(
            summary["mapping_rate"], len(assignments) / 3, places=6
        )
        self.assertGreaterEqual(summary["mean_abs_deviation"], 0.0)
        self.assertGreaterEqual(summary["max_abs_deviation"], summary["mean_abs_deviation"])
        self.assertEqual(
            sum(summary["per_ideal_func"].values()), len(assignments)
        )
    
    def test_summarise_mapping_empty(self) -> None:
        """An empty mapping produces zeroed statistics and no distribution."""
        summary = summarise_mapping([], 5)
        self.assertEqual(summary["mapped_points"], 0)
        self.assertEqual(summary["unmapped_points"], 5)
        self.assertEqual(summary["mapping_rate"], 0.0)
        self.assertEqual(summary["mean_abs_deviation"], 0.0)
        self.assertEqual(summary["max_abs_deviation"], 0.0)
        self.assertEqual(summary["per_ideal_func"], {})


if __name__ == "__main__":
    unittest.main()
