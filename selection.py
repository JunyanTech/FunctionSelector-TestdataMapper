"""Least-squares selection of the four best ideal functions.

The module implements :class:`IdealFunctionSelector`: given 
the four training datasets and the fifty ideal functions, choose 
for every training function the ideal function that minimises 
the sum of the squared y-deviations (least-squares criterion).

For each chosen pair the module also computes

* the largest deviation between the training function and its ideal
  function (``max_deviation``), and
* the mapping threshold ``sqrt(2) * max_deviation`` that is later used
  by :class:`~mapping.TestDataMapper`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from base import AnalysisStep
from exceptions import SelectionError


@dataclass(frozen=True)
class SelectionResult:
    """Result of the least-squares selection for one training function.

    Attributes
    ----------
    training_func_no:
        Number of the training function (1 to 4).
    ideal_func_no:
        Number of the chosen ideal function (1 to 50).
    sse:
        Sum of squared y-deviations between the training function and
        its chosen ideal function.
    max_deviation:
        Largest absolute y-deviation between the training function and
        its chosen ideal function.
    threshold:
        Maximum allowed deviation for a test point mapped to this ideal
        function, i.e. ``sqrt(2) * max_deviation``.
    """

    training_func_no: int
    ideal_func_no: int
    sse: float
    max_deviation: float
    threshold: float

    def as_db_record(self) -> dict:
        """Return the result as a dictionary for the database table."""
        return {
            "training_func_no": self.training_func_no,
            "ideal_func_no": self.ideal_func_no,
            "sse": self.sse,
            "max_deviation": self.max_deviation,
            "threshold": self.threshold,
        }


class IdealFunctionSelector(AnalysisStep):
    """Select the best-fitting ideal function for each training function.

    The selector computes, for every training function ``i`` and every
    ideal function ``j``, the sum of squared deviations on the common
    x-values and keeps the ideal function with the smallest error.

    Parameters
    ----------
    mapping_tolerance:
        Factor applied to the largest training deviation to obtain the
        mapping threshold (``sqrt(2)``).
    """

    name: str = "ideal-function selection"

    def __init__(self, mapping_tolerance: float = np.sqrt(2.0)) -> None:
        self.mapping_tolerance = float(mapping_tolerance)

    def run(
        self, training_frame: pd.DataFrame, ideal_frame: pd.DataFrame
    ) -> list[SelectionResult]:
        """Execute the least-squares selection.

        Parameters
        ----------
        training_frame:
            Frame with columns ``x, y1, y2, y3, y4``.
        ideal_frame:
            Frame with columns ``x, y1, ..., y50``.

        Returns
        -------
        list[SelectionResult]
            One result per training function (four entries).

        Raises
        ------
        SelectionError
            If the two frames do not share any x-values or contain no
            usable columns.
        """
        training_columns = [f"y{i}" for i in range(1, 5)]
        ideal_columns = [f"y{j}" for j in range(1, 51)]

        missing_training = [c for c in training_columns if c not in training_frame]
        missing_ideal = [c for c in ideal_columns if c not in ideal_frame]
        if missing_training or missing_ideal:
            raise SelectionError(
                "Training/ideal frame is missing expected columns: "
                f"training missing {missing_training}, "
                f"ideal missing {missing_ideal}."
            )

        # Align both frames on the common x-values (inner join).
        # The ideal columns are renamed with an ``ideal_`` prefix because
        # the training and the ideal frames both contain y1..y4.
        ideal_renamed = ideal_frame[["x"] + ideal_columns].rename(
            columns={column: f"ideal_{column}" for column in ideal_columns}
        )
        merged = pd.merge(
            training_frame[["x"] + training_columns],
            ideal_renamed,
            on="x",
            how="inner",
        )
        if merged.empty:
            raise SelectionError(
                "The training data and the ideal functions do not share "
                "any x-values; selection is impossible."
            )

        results: list[SelectionResult] = []
        for index, training_column in enumerate(training_columns, start=1):
            training_values = merged[training_column].to_numpy(dtype=float)
            squared_errors: list[float] = []
            for ideal_column in ideal_columns:
                ideal_values = merged[f"ideal_{ideal_column}"].to_numpy(
                    dtype=float
                )
                squared_error = float(
                    np.sum((training_values - ideal_values) ** 2)
                )
                squared_errors.append(squared_error)

            best_index = int(np.argmin(squared_errors))
            best_ideal_column = ideal_columns[best_index]
            best_ideal_values = merged[f"ideal_{best_ideal_column}"].to_numpy(
                dtype=float
            )
            max_deviation = float(np.max(np.abs(training_values - best_ideal_values)))

            results.append(
                SelectionResult(
                    training_func_no=index,
                    ideal_func_no=best_index + 1,
                    sse=squared_errors[best_index],
                    max_deviation=max_deviation,
                    threshold=self.mapping_tolerance * max_deviation,
                )
            )
        return results
