"""Unit tests for the :mod:`database` module."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from database import (
    IDEAL_TABLE,
    SELECTION_TABLE,
    TEST_RESULTS_TABLE,
    TRAINING_TABLE,
    SQLiteDatabaseManager,
)
from exceptions import DatabaseOperationError


class SQLiteDatabaseManagerTest(unittest.TestCase):
    """Test schema creation and the data loading/storing operations."""

    def setUp(self) -> None:
        """Create a fresh database in a temporary directory."""
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "test.db"
        self.manager = SQLiteDatabaseManager(self.db_path)
        self.manager.connect()
        self.manager.create_schema()

    def tearDown(self) -> None:
        """Close the manager and remove the temporary directory."""
        self.manager.close()
        self._temp_dir.cleanup()

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------
    def test_schema_contains_all_tables(self) -> None:
        """All four tables exist after create_schema()."""
        tables = {
            row["name"]
            for row in self.manager.read_table("sqlite_master")
            if row["type"] == "table"
        }
        self.assertTrue({TRAINING_TABLE, IDEAL_TABLE, TEST_RESULTS_TABLE,
                         SELECTION_TABLE}.issubset(tables))

    def test_save_and_read_training_data(self) -> None:
        """Training data are stored and read back correctly."""
        frame = pd.DataFrame(
            {"x": [0.0, 1.0], "y1": [1.0, 2.0], "y2": [3.0, 4.0],
             "y3": [5.0, 6.0], "y4": [7.0, 8.0]}
        )
        count = self.manager.save_training_data(frame)
        self.assertEqual(count, 2)

        rows = self.manager.read_table(TRAINING_TABLE)
        self.assertEqual(len(rows), 2)
        first = rows[0]
        self.assertAlmostEqual(first["x"], 0.0)
        self.assertAlmostEqual(first["y1"], 1.0)
        self.assertAlmostEqual(first["y4"], 7.0)

    def test_save_ideal_functions_round_trip(self) -> None:
        """Ideal functions are stored and read back correctly."""
        frame = pd.DataFrame({"x": [0.0, 1.0]})
        for number in range(1, 51):
            frame[f"y{number}"] = [float(number), float(number)]
        count = self.manager.save_ideal_functions(frame)
        self.assertEqual(count, 2)

        rows = self.manager.read_table(IDEAL_TABLE)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 51)  # x plus 50 y-columns
        self.assertAlmostEqual(rows[0]["y50"], 50.0)

    def test_save_test_results_row_by_row(self) -> None:
        """Test results are inserted and can be read back."""
        records = [
            {"x": 0.5, "y": 10.0, "delta_y": 0.05, "ideal_func_no": 7},
            {"x": 1.5, "y": 11.0, "delta_y": 0.10, "ideal_func_no": 3},
        ]
        count = self.manager.save_test_results(records)
        self.assertEqual(count, 2)

        rows = self.manager.read_table(TEST_RESULTS_TABLE)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["delta_y"], 0.05)
        self.assertEqual(rows[0]["ideal_func_no"], 7)

    def test_save_selection_results(self) -> None:
        """Selection summaries are stored correctly."""
        records = [
            {"training_func_no": 1, "ideal_func_no": 12, "sse": 0.1,
             "max_deviation": 0.4, "threshold": 0.565685424949238},
        ]
        count = self.manager.save_selection_results(records)
        self.assertEqual(count, 1)
        rows = self.manager.read_table(SELECTION_TABLE)
        self.assertAlmostEqual(rows[0]["threshold"], 0.565685424949238)

    def test_create_schema_drops_old_data(self) -> None:
        """Re-running create_schema() clears previously inserted data."""
        frame = pd.DataFrame(
            {"x": [0.0], "y1": [1.0], "y2": [1.0], "y3": [1.0], "y4": [1.0]}
        )
        self.manager.save_training_data(frame)
        self.assertEqual(len(self.manager.read_table(TRAINING_TABLE)), 1)

        self.manager.create_schema(drop_existing=True)
        self.assertEqual(len(self.manager.read_table(TRAINING_TABLE)), 0)

    def test_read_before_connect_raises(self) -> None:
        """Reading a table before connect() raises DatabaseOperationError."""
        manager = SQLiteDatabaseManager(self.db_path)
        with self.assertRaises(DatabaseOperationError):
            manager.read_table(TRAINING_TABLE)


if __name__ == "__main__":
    unittest.main()
