"""SQLite database handling based on SQLAlchemy.

The module creates the SQLite database file, defines three required tables
plus one auxiliary table and loads/stores the data:

* ``training_data``     -- x plus the four training functions (5 columns)
* ``ideal_functions``   -- x plus the fifty ideal functions (51 columns)
* ``test_results``      -- x, y, delta-y and the number of the assigned
  ideal function (4 columns)
* ``selection_results`` -- auxiliary table that stores the four chosen
  ideal functions together with their least-squares error, the largest
  training deviation and the resulting mapping threshold

The class hierarchy demonstrates inheritance: :class:`DatabaseManager`
is the abstract base class defining the contract, and
:class:`SQLiteDatabaseManager` implements it for SQLite using the
SQLAlchemy Core API.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sqlalchemy import Column, Float, Integer, MetaData, Table, create_engine, text
from sqlalchemy.engine import Engine

from exceptions import DatabaseOperationError

#: Names of the tables managed by the database layer.
TRAINING_TABLE = "training_data"
IDEAL_TABLE = "ideal_functions"
TEST_RESULTS_TABLE = "test_results"
SELECTION_TABLE = "selection_results"

#: Number of ideal functions in the provided dataset.
NUM_IDEAL_FUNCTIONS = 50

#: Number of training functions in the provided dataset.
NUM_TRAINING_FUNCTIONS = 4


class DatabaseManager(abc.ABC):
    """Abstract contract of a database backend.

    Subclasses must implement every abstract method below.  The public
    workflow is: :meth:`connect`, :meth:`create_schema`,
    :meth:`save_training_data`, :meth:`save_ideal_functions`,
    :meth:`save_test_results`, :meth:`save_selection_results`.
    """

    @abc.abstractmethod
    def connect(self) -> None:
        """Open the connection to the database file."""

    @abc.abstractmethod
    def create_schema(self, drop_existing: bool = True) -> None:
        """Create all tables, optionally dropping existing ones first."""

    @abc.abstractmethod
    def save_training_data(self, frame: pd.DataFrame) -> int:
        """Insert the merged training data; return the row count."""

    @abc.abstractmethod
    def save_ideal_functions(self, frame: pd.DataFrame) -> int:
        """Insert the fifty ideal functions; return the row count."""

    @abc.abstractmethod
    def save_test_results(self, records: Sequence[dict]) -> int:
        """Insert the mapped test results; return the row count."""

    @abc.abstractmethod
    def save_selection_results(self, records: Sequence[dict]) -> int:
        """Insert the selection summary; return the row count."""

    @abc.abstractmethod
    def read_table(self, name: str) -> list[dict]:
        """Read every row of *name* as a list of dictionaries."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close the underlying connection."""


class SQLiteDatabaseManager(DatabaseManager):
    """SQLite implementation of :class:`DatabaseManager` using SQLAlchemy.

    Parameters
    ----------
    db_path:
        File system path of the SQLite database file.  The parent
        directory must exist; it is created by :meth:`connect` if it
        does not.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._engine: Engine | None = None
        self._metadata = MetaData()
        self._tables: dict[str, Table] = {}

    # ------------------------------------------------------------------
    # connection handling
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Create the engine and the database file.

        Raises
        ------
        DatabaseOperationError
            If the SQLite engine cannot be created.
        """
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._engine = create_engine(
                f"sqlite:///{self.db_path}", echo=False, future=True
            )
            # Touch the file by opening a connection.
            with self._engine.connect():
                pass
        except Exception as error:  # noqa: BLE001 - wrapped for the caller
            raise DatabaseOperationError(
                f"Could not create the SQLite database at {self.db_path}: {error}"
            ) from error

    def create_schema(self, drop_existing: bool = True) -> None:
        """Create the database schema.

        Parameters
        ----------
        drop_existing:
            If ``True`` (default) all tables managed by this class are
            dropped first, so that every program run starts with a fresh
            database.

        Raises
        ------
        DatabaseOperationError
            If the schema cannot be created.
        """
        if self._engine is None:
            raise DatabaseOperationError(
                "create_schema() called before connect()."
            )
        try:
            # Reset the metadata so that a repeated call on the same
            # manager instance can re-define the tables.
            self._metadata = MetaData()
            self._tables = {}
            self._define_tables()
            if drop_existing:
                self._metadata.drop_all(self._engine)
            self._metadata.create_all(self._engine)
        except Exception as error:  # noqa: BLE001 - wrapped for the caller
            raise DatabaseOperationError(
                f"Could not create the database schema: {error}"
            ) from error

    # ------------------------------------------------------------------
    # data loading
    # ------------------------------------------------------------------
    def save_training_data(self, frame: pd.DataFrame) -> int:
        """Insert the merged training data into ``training_data``.

        Parameters
        ----------
        frame:
            Frame with columns ``x, y1, y2, y3, y4``.

        Returns
        -------
        int
            Number of inserted rows.
        """
        return self._save_frame(frame, TRAINING_TABLE)

    def save_ideal_functions(self, frame: pd.DataFrame) -> int:
        """Insert the fifty ideal functions into ``ideal_functions``.

        Parameters
        ----------
        frame:
            Frame with columns ``x, y1, ..., y50``.

        Returns
        -------
        int
            Number of inserted rows.
        """
        return self._save_frame(frame, IDEAL_TABLE)

    def save_test_results(self, records: Sequence[dict]) -> int:
        """Insert the mapped test results row by row.

        The test data are read line by line and 
        the results are stored afterwards, therefore
        the records handed in here are inserted individually.

        Parameters
        ----------
        records:
            Sequence of dictionaries with keys ``x``, ``y``,
            ``delta_y`` and ``ideal_func_no``.

        Returns
        -------
        int
            Number of inserted rows.
        """
        return self._insert_records(records, TEST_RESULTS_TABLE)

    def save_selection_results(self, records: Sequence[dict]) -> int:
        """Insert the selection summary into ``selection_results``.

        Parameters
        ----------
        records:
            Sequence of dictionaries with keys ``training_func_no``,
            ``ideal_func_no``, ``sse``, ``max_deviation`` and
            ``threshold``.

        Returns
        -------
        int
            Number of inserted rows.
        """
        return self._insert_records(records, SELECTION_TABLE)

    # ------------------------------------------------------------------
    # inspection helpers
    # ------------------------------------------------------------------
    def read_table(self, name: str) -> list[dict]:
        """Read every row of *name* as a list of dictionaries.

        Parameters
        ----------
        name:
            Name of the table to read.

        Returns
        -------
        list[dict]
            The rows with column names as keys.
        """
        if self._engine is None:
            raise DatabaseOperationError("read_table() called before connect().")
        try:
            with self._engine.connect() as connection:
                rows = connection.execute(
                    text(f"SELECT * FROM {name}")  # noqa: S608 - name is trusted
                ).mappings().all()
            return [dict(row) for row in rows]
        except Exception as error:  # noqa: BLE001 - wrapped for the caller
            raise DatabaseOperationError(
                f"Could not read table '{name}': {error}"
            ) from error

    def close(self) -> None:
        """Dispose the engine and free the database file handle."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _define_tables(self) -> None:
        """Define the four tables on the shared metadata object."""
        self._tables = {
            TRAINING_TABLE: Table(
                TRAINING_TABLE,
                self._metadata,
                Column("x", Float, primary_key=True),
                *[
                    Column(f"y{index}", Float)
                    for index in range(1, NUM_TRAINING_FUNCTIONS + 1)
                ],
            ),
            IDEAL_TABLE: Table(
                IDEAL_TABLE,
                self._metadata,
                Column("x", Float, primary_key=True),
                *[
                    Column(f"y{index}", Float)
                    for index in range(1, NUM_IDEAL_FUNCTIONS + 1)
                ],
            ),
            TEST_RESULTS_TABLE: Table(
                TEST_RESULTS_TABLE,
                self._metadata,
                # Exactly four columns: 
                # x, y, delta-y and the ideal function number.
                Column("x", Float),
                Column("y", Float),
                Column("delta_y", Float),
                Column("ideal_func_no", Integer),
            ),
            SELECTION_TABLE: Table(
                SELECTION_TABLE,
                self._metadata,
                Column("training_func_no", Integer, primary_key=True),
                Column("ideal_func_no", Integer),
                Column("sse", Float),
                Column("max_deviation", Float),
                Column("threshold", Float),
            ),
        }

    def _save_frame(self, frame: pd.DataFrame, table_name: str) -> int:
        """Bulk-insert a pandas frame into an existing table.

        Parameters
        ----------
        frame:
            The frame to insert; its column names must match the table
            columns.
        table_name:
            Name of the target table.

        Returns
        -------
        int
            Number of inserted rows.

        Raises
        ------
        DatabaseOperationError
            If the insert fails.
        """
        if self._engine is None:
            raise DatabaseOperationError("save called before connect().")
        try:
            with self._engine.begin() as connection:
                frame.to_sql(
                    table_name,
                    con=connection,
                    if_exists="append",
                    index=False,
                )
            return len(frame)
        except Exception as error:  # noqa: BLE001 - wrapped for the caller
            raise DatabaseOperationError(
                f"Could not insert data into '{table_name}': {error}"
            ) from error

    def _insert_records(self, records: Iterable[dict], table_name: str) -> int:
        """Insert a sequence of record dictionaries row by row.

        Parameters
        ----------
        records:
            Iterable of dictionaries mapping column names to values.
        table_name:
            Name of the target table.

        Returns
        -------
        int
            Number of inserted rows.

        Raises
        ------
        DatabaseOperationError
            If the insert fails.
        """
        if self._engine is None:
            raise DatabaseOperationError("save called before connect().")
        table = self._tables.get(table_name)
        if table is None:
            raise DatabaseOperationError(
                f"Unknown table '{table_name}'; call create_schema() first."
            )
        record_list = list(records)
        try:
            with self._engine.begin() as connection:
                for record in record_list:
                    connection.execute(table.insert().values(**record))
            return len(record_list)
        except Exception as error:  # noqa: BLE001 - wrapped for the caller
            raise DatabaseOperationError(
                f"Could not insert records into '{table_name}': {error}"
            ) from error
