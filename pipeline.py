"""End-to-end pipeline of the program.

The module provides :class:`AssignmentPipeline`, which orchestrates the
complete workflow in a correct order:

1. load and validate the four training data sets, the fifty ideal
   functions and the test data set from the CSV files,
2. create the SQLite database and load the training data and the ideal
   functions into it,
3. select the four best ideal functions by the least-squares criterion,
4. map the test data to the chosen ideal functions (criterion
   ``sqrt(2) * max_deviation``) and store the results,
5. create the Bokeh visualization,
6. report a compact summary of every step.
"""

from __future__ import annotations

from config import Config
from data_loader import DataLoader
from database import SQLiteDatabaseManager
from mapping import TestAssignment, TestDataMapper
from selection import IdealFunctionSelector, SelectionResult
from visualization import Visualizer


class AssignmentPipeline:
    """Coordinate all components of the program.

    Parameters
    ----------
    config:
        The central configuration object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.summary: dict = {}

    def run(self) -> dict:
        """Execute the complete workflow.

        Returns
        -------
        dict
            Summary of the run containing the loaded row counts, the
            selection results, the mapping statistics and the paths of
            the created artefacts.
        """
        self.config.ensure_directories()

        # 1. load and validate the input data.
        loader = DataLoader(
            self.config.training_files,
            self.config.ideal_file,
            self.config.test_file,
        )
        training_frame = loader.load_training_data()
        ideal_frame = loader.load_ideal_functions()
        test_frame = loader.load_test_data()

        # 2. create the database and load the training / ideal data.
        database = SQLiteDatabaseManager(self.config.db_path)
        database.connect()
        database.create_schema()
        training_rows = database.save_training_data(training_frame)
        ideal_rows = database.save_ideal_functions(ideal_frame)

        # 3. select the four best ideal functions (least squares).
        selector = IdealFunctionSelector(self.config.mapping_tolerance)
        selections: list[SelectionResult] = selector.run(
            training_frame, ideal_frame
        )
        database.save_selection_results(
            [selection.as_db_record() for selection in selections]
        )

        # 4. map the test data and store the results.
        mapper = TestDataMapper(
            selections,
            restrict_to_training_x_range=self.config.restrict_to_training_x_range,
        )
        assignments: list[TestAssignment]
        assignments, unmapped = mapper.run(test_frame, ideal_frame)
        mapped_rows = database.save_test_results(
            [assignment.as_db_record() for assignment in assignments]
        )

        # 5. visualize everything.
        visualizer = Visualizer(self.config.output_dir, self.config.show_plot)
        plot_path = visualizer.plot_all(
            training_frame,
            ideal_frame,
            test_frame,
            selections,
            assignments,
            unmapped,
        )
        database.close()

        self.summary = {
            "training_rows": training_rows,
            "ideal_rows": ideal_rows,
            "test_points": len(test_frame),
            "selections": [
                {
                    "training_func_no": s.training_func_no,
                    "ideal_func_no": s.ideal_func_no,
                    "sse": s.sse,
                    "max_deviation": s.max_deviation,
                    "threshold": s.threshold,
                }
                for s in selections
            ],
            "mapped_points": mapped_rows,
            "unmapped_points": unmapped,
            "database_path": str(self.config.db_path),
            "plot_path": str(plot_path),
        }
        return self.summary
