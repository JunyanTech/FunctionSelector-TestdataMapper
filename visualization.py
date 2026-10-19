"""Bokeh-based visualization of all datasets and the mapping result.

The module provides :class:`Visualizer`, which creates one interactive
HTML document with two tabs:

* *Training data & chosen ideal functions* -- the four training data
  sets as scatter plots together with the four chosen ideal functions
  as lines.
* *Test data mapping & deviations* -- every mapped test point coloured
  by its assigned ideal function, with a vertical error bar that shows
  the deviation ``delta_y`` between the test point and the ideal
  function; test points that could not be mapped are drawn in grey.

The deviation representation was chosen as vertical error bars because
they directly visualize the mapping criterion (the absolute y-deviation 
between a test point and its ideal function).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.models import ColumnDataSource, HoverTool, TabPanel, Tabs
from bokeh.palettes import Category10
from bokeh.plotting import figure

from exceptions import VisualizationError
from mapping import TestAssignment
from selection import SelectionResult

#: Colour of the test points that could not be mapped to any ideal
#: function.
UNMAPPED_COLOR = "#7f7f7f"


class Visualizer:
    """Create and save the Bokeh visualization.

    Parameters
    ----------
    output_dir:
        Directory into which the HTML file is written.
    show_plot:
        If ``True`` the generated document is opened in the default
        browser after saving.
    """

    def __init__(self, output_dir: Path, show_plot: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.show_plot = show_plot

    def plot_all(
        self,
        training_frame: pd.DataFrame,
        ideal_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        selections: Sequence[SelectionResult],
        assignments: Sequence[TestAssignment],
        unmapped_count: int,
    ) -> Path:
        """Create the full visualization and save it as HTML.

        Parameters
        ----------
        training_frame:
            Frame with columns ``x, y1, y2, y3, y4``.
        ideal_frame:
            Frame with columns ``x, y1, ..., y50``.
        test_frame:
            Frame with columns ``x, y``.
        selections:
            The four chosen ideal functions.
        assignments:
            The mapped test points.
        unmapped_count:
            Number of test points that could not be mapped.

        Returns
        -------
        Path
            Path of the saved HTML file.

        Raises
        ------
        VisualizationError
            If the plot cannot be created or saved.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            tab_training = self._build_training_tab(
                training_frame, ideal_frame, selections
            )
            tab_mapping = self._build_mapping_tab(
                ideal_frame, test_frame, assignments, unmapped_count
            )
            tabs = Tabs(tabs=[tab_training, tab_mapping])

            output_path = self.output_dir / "visualization.html"
            output_file(output_path, title="Visualization - FunctionSelector-TestdataMapper")
            save(tabs)
            if self.show_plot:
                show(tabs)
            return output_path
        except Exception as error:  # noqa: BLE001 - wrapped for the caller
            raise VisualizationError(
                f"Could not create the visualization: {error}"
            ) from error

    # ------------------------------------------------------------------
    # tab 1: training data and chosen ideal functions
    # ------------------------------------------------------------------
    def _build_training_tab(
        self,
        training_frame: pd.DataFrame,
        ideal_frame: pd.DataFrame,
        selections: Sequence[SelectionResult],
    ) -> TabPanel:
        """Build the tab that shows training data and chosen ideals."""
        plot = figure(
            title="Training data and the four chosen ideal functions",
            x_axis_label="x",
            y_axis_label="y",
            width=1000,
            height=600,
            tools="pan,box_zoom,wheel_zoom,reset,save,hover",
        )

        for index in range(1, 5):
            column_name = f"y{index}"
            if column_name not in training_frame:
                continue
            source = ColumnDataSource(
                {
                    "x": training_frame["x"].to_numpy(dtype=float),
                    "y": training_frame[column_name].to_numpy(dtype=float),
                }
            )
            plot.scatter(
                "x",
                "y",
                source=source,
                size=5,
                color=Category10[10][(index - 1) % 10],
                legend_label=f"Training data {index}",
                alpha=0.8,
            )

        for selection in selections:
            column_name = f"y{selection.ideal_func_no}"
            if column_name not in ideal_frame:
                continue
            plot.line(
                ideal_frame["x"].to_numpy(dtype=float),
                ideal_frame[column_name].to_numpy(dtype=float),
                line_width=2.5,
                line_dash="dashed",
                color=Category10[10][(selection.ideal_func_no - 1) % 10],
                legend_label=f"Ideal function {selection.ideal_func_no}",
            )

        plot.legend.location = "top_left"
        plot.legend.click_policy = "hide"
        return TabPanel(child=plot, title="Training & ideal functions")

    # ------------------------------------------------------------------
    # tab 2: test data mapping and deviations
    # ------------------------------------------------------------------
    def _build_mapping_tab(
        self,
        ideal_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        assignments: Sequence[TestAssignment],
        unmapped_count: int,
    ) -> TabPanel:
        """Build the tab that shows the test-data mapping result."""
        plot = figure(
            title="Test data mapped to the chosen ideal functions "
                  "(deviation = error bar)",
            x_axis_label="x",
            y_axis_label="y",
            width=1000,
            height=600,
            tools="pan,box_zoom,wheel_zoom,reset,save,hover",
        )

        ideal_lookup = self._ideal_lookup(ideal_frame, assignments)

        # Group the assigned test points by their ideal function.
        grouped: dict[int, dict[str, np.ndarray]] = {}
        for assignment in assignments:
            group = grouped.setdefault(
                assignment.ideal_func_no,
                {"x": [], "y": [], "ideal_y": [], "delta_y": []},
            )
            group["x"].append(assignment.x)
            group["y"].append(assignment.y)
            ideal_row = ideal_lookup.get(assignment.ideal_func_no, {})
            group["ideal_y"].append(ideal_row.get(assignment.x, assignment.y))
            group["delta_y"].append(assignment.delta_y)

        # Error bars: vertical segment between the ideal value and 
        # the test value at the same x.
        for func_no, group in grouped.items():
            color = Category10[10][(func_no - 1) % 10]
            source = ColumnDataSource(
                {
                    "x": group["x"],
                    "y": group["y"],
                    "ideal_y": group["ideal_y"],
                    "delta_y": group["delta_y"],
                }
            )
            plot.segment(
                "x", "ideal_y", "x", "y", source=source, color=color, line_width=1.0
            )
            plot.scatter(
                "x",
                "y",
                source=source,
                size=7,
                color=color,
                legend_label=f"Test -> Ideal {func_no}",
            )

        # Draw the chosen ideal functions as reference lines.
        selected_funcs = {a.ideal_func_no for a in assignments}
        for func_no in sorted(selected_funcs):
            column_name = f"y{func_no}"
            if column_name not in ideal_frame:
                continue
            plot.line(
                ideal_frame["x"].to_numpy(dtype=float),
                ideal_frame[column_name].to_numpy(dtype=float),
                line_width=2.0,
                line_dash="dashed",
                color=Category10[10][(func_no - 1) % 10],
                legend_label=f"Ideal function {func_no}",
            )

        # Unmapped test points.
        assigned_indices = {(a.x, a.y) for a in assignments}
        unmapped_x: list[float] = []
        unmapped_y: list[float] = []
        for row in test_frame.itertuples(index=False):
            if (float(row.x), float(row.y)) not in assigned_indices:
                unmapped_x.append(float(row.x))
                unmapped_y.append(float(row.y))
        if unmapped_x:
            plot.scatter(
                unmapped_x,
                unmapped_y,
                size=6,
                color=UNMAPPED_COLOR,
                marker="x",
                alpha=0.7,
                legend_label=f"Unmapped ({unmapped_count})",
            )

        # Hover tooltip that shows the deviation.
        hover = HoverTool(
            tooltips=[
                ("x", "@x"),
                ("y", "@y"),
                ("delta_y", "@delta_y"),
                ("ideal value", "@ideal_y"),
            ]
        )
        plot.add_tools(hover)
        plot.legend.location = "top_left"
        plot.legend.click_policy = "hide"
        return TabPanel(child=plot, title="Test data mapping")

    @staticmethod
    def _ideal_lookup(
        ideal_frame: pd.DataFrame, assignments: Sequence[TestAssignment]
    ) -> dict[int, dict[float, float]]:
        """Build ``{func_no: {x: ideal_y}}`` lookup for the chosen ideals.

        Parameters
        ----------
        ideal_frame:
            Frame with columns ``x, y1, ..., y50``.
        assignments:
            The mapped test points (only their function numbers are
            needed).

        Returns
        -------
        dict[int, dict[float, float]]
            Mapping from ideal-function number to x -> ideal y-value.
        """
        lookup: dict[int, dict[float, float]] = {}
        for func_no in {a.ideal_func_no for a in assignments}:
            column_name = f"y{func_no}"
            if column_name not in ideal_frame:
                continue
            lookup[func_no] = {
                float(x): float(y)
                for x, y in zip(
                    ideal_frame["x"].to_numpy(dtype=float),
                    ideal_frame[column_name].to_numpy(dtype=float),
                )
            }
        return lookup
