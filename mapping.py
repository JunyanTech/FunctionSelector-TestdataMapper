"""Mapping of the test data to the four chosen ideal functions.

The module implements :class:`TestDataMapper`. 
Every x-y pair of the test dataset is examined 
against the four chosen ideal functions: the test point 
is assigned to the ideal function with the smallest absolute deviation, 
provided that this deviation does not exceed the largest deviation 
of the corresponding training dataset multiplied by the
factor ``sqrt(2)`` (the mapping criterion).

Test points that do not fulfil the criterion for any ideal function are
left unmapped and are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from base import AnalysisStep
from exceptions import MappingError
from selection import SelectionResult


@dataclass(frozen=True)
class TestAssignment:
    """One mapped test point with its deviation.

    Attributes
    ----------
    x:
        x-value of the test point.
    y:
        y-value of the test point.
    delta_y:
        Absolute deviation between the test y-value and the value of the
        assigned ideal function at ``x``.
    ideal_func_no:
        Number of the ideal function the test point was assigned to
        (1 to 50).
    """

    x: float
    y: float
    delta_y: float
    ideal_func_no: int

    def as_db_record(self) -> dict:
        """Return the assignment as a dictionary for the database table."""
        return {
            "x": self.x,
            "y": self.y,
            "delta_y": self.delta_y,
            "ideal_func_no": self.ideal_func_no,
        }


class TestDataMapper(AnalysisStep):
    """Map test points to the chosen ideal functions.

    Parameters
    ----------
    selections:
        The four selection results produced by
        :class:`~selection.IdealFunctionSelector`.
    restrict_to_training_x_range:
        If ``True``, test points whose x-value lies outside the x-range
        of the training data are never mapped.  The assignment itself
        defines the criterion only through the deviation factor, so the
        default is ``False``.
    """

    name: str = "test-data mapping"

    def __init__(
        self,
        selections: Sequence[SelectionResult],
        restrict_to_training_x_range: bool = False,
    ) -> None:
        if not selections:
            raise MappingError(
                "TestDataMapper requires at least one selection result."
            )
        self.selections = list(selections)
        self.restrict_to_training_x_range = restrict_to_training_x_range
        self._ideal_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def run(
        self, test_frame: pd.DataFrame, ideal_frame: pd.DataFrame
    ) -> tuple[list[TestAssignment], int]:
        """Map every test point and return the assignments.

        Parameters
        ----------
        test_frame:
            Frame with columns ``x, y``.
        ideal_frame:
            Frame with columns ``x, y1, ..., y50``.

        Returns
        -------
        tuple[list[TestAssignment], int]
            The list of assigned test points and the number of test
            points that could not be mapped.

        Raises
        ------
        MappingError
            If a needed ideal-function column is missing or a test
            point cannot be evaluated.
        """
        for column in ("x", "y"):
            if column not in test_frame.columns:
                raise MappingError(f"Test frame is missing the '{column}' column.")

        self._prepare_ideal_curves(ideal_frame)
        x_grid = ideal_frame["x"].to_numpy(dtype=float)

        assignments: list[TestAssignment] = []
        unmapped_count = 0

        for row in test_frame.itertuples(index=False):
            x_value = float(row.x)
            y_value = float(row.y)

            # Optional safeguard: ignore test points outside the x-range
            # covered by the training data.
            if self.restrict_to_training_x_range:
                x_min = float(np.min(x_grid))
                x_max = float(np.max(x_grid))
                if not (x_min <= x_value <= x_max):
                    unmapped_count += 1
                    continue

            best_selection: SelectionResult | None = None
            best_deviation: float | None = None

            for selection in self.selections:
                ideal_value = self._ideal_value_at(
                    x_value, selection.ideal_func_no
                )
                deviation = abs(y_value - ideal_value)
                if best_deviation is None or deviation < best_deviation:
                    best_deviation = deviation
                    best_selection = selection

            # Apply the assignment criterion: the smallest deviation must
            # not exceed the allowed threshold of the chosen ideal
            # function.
            if (
                best_selection is not None
                and best_deviation is not None
                and best_deviation <= best_selection.threshold
            ):
                assignments.append(
                    TestAssignment(
                        x=x_value,
                        y=y_value,
                        delta_y=best_deviation,
                        ideal_func_no=best_selection.ideal_func_no,
                    )
                )
            else:
                unmapped_count += 1

        return assignments, unmapped_count

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _prepare_ideal_curves(self, ideal_frame: pd.DataFrame) -> None:
        """Pre-compute the (x, y) arrays of every chosen ideal function.

        Parameters
        ----------
        ideal_frame:
            Frame with columns ``x, y1, ..., y50``.

        Raises
        ------
        MappingError
            If the frame does not contain the needed column or is empty.
        """
        if ideal_frame.empty or "x" not in ideal_frame.columns:
            raise MappingError("Ideal frame is empty or missing the 'x' column.")

        x_grid = ideal_frame["x"].to_numpy(dtype=float)
        if not np.all(np.diff(x_grid) > 0):
            # Interpolation requires a strictly increasing x-grid; sort
            # the frame if it is not already ordered.
            ideal_frame = ideal_frame.sort_values("x")
            x_grid = ideal_frame["x"].to_numpy(dtype=float)

        for selection in self.selections:
            column = f"y{selection.ideal_func_no}"
            if column not in ideal_frame.columns:
                raise MappingError(
                    f"Ideal frame is missing column '{column}' needed for "
                    f"ideal function {selection.ideal_func_no}."
                )
            values = ideal_frame[column].to_numpy(dtype=float)
            self._ideal_curves[selection.ideal_func_no] = (x_grid, values)

    def _ideal_value_at(self, x_value: float, func_no: int) -> float:
        """Return the ideal-function value at ``x_value``.

        The value is taken from the grid if ``x_value`` matches an exact
        grid point; otherwise linear interpolation on the ideal curve is
        used (with clamping at the grid boundaries).

        Parameters
        ----------
        x_value:
            The x-coordinate of the test point.
        func_no:
            Number of the ideal function (1 to 50).

        Returns
        -------
        float
            The interpolated ideal value.

        Raises
        ------
        MappingError
            If the ideal curve was not prepared.
        """
        curve = self._ideal_curves.get(func_no)
        if curve is None:
            raise MappingError(
                f"Ideal curve for function {func_no} was not prepared."
            )
        grid_x, grid_y = curve
        return float(np.interp(x_value, grid_x, grid_y))


def summarise_mapping(
    assignments: Sequence[TestAssignment], unmapped_count: int
) -> dict:
    """Compute compact statistics over the mapping results.

    The function complements the per-point database table with aggregate
    figures that are useful for the results analysis of the assignment,
    namely the mapping rate and the mean and maximum absolute deviation.

    Parameters
    ----------
    assignments:
        The list of :class:`TestAssignment` objects produced by
        :meth:`TestDataMapper.run`.
    unmapped_count:
        The number of test points that could not be mapped.

    Returns
    -------
    dict
        Dictionary with the keys ``mapped_points``, ``unmapped_points``,
        ``mapping_rate``, ``mean_abs_deviation``, ``max_abs_deviation``
        and ``per_ideal_func``.
    """
    mapped = len(assignments)
    total = mapped + unmapped_count
    deviations = [assignment.delta_y for assignment in assignments]
    per_ideal_func: dict[int, int] = {}
    for assignment in assignments:
        per_ideal_func[assignment.ideal_func_no] = (
            per_ideal_func.get(assignment.ideal_func_no, 0) + 1
        )
    return {
        "mapped_points": mapped,
        "unmapped_points": unmapped_count,
        "mapping_rate": (mapped / total) if total else 0.0,
        "mean_abs_deviation": float(np.mean(deviations)) if deviations else 0.0,
        "max_abs_deviation": float(np.max(deviations)) if deviations else 0.0,
        "per_ideal_func": per_ideal_func,
    }
    