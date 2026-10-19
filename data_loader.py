"""Loading and validation of the input CSV data sets.

The module provides the :class:`DataLoader` class that reads the four
training datasets, the fifty ideal functions and the test dataset from
CSV files and returns them as :class:`pandas.DataFrame` objects with a
well-defined column layout:

* training data   -> columns ``x, y1, y2, y3, y4``
* ideal functions -> columns ``x, y1, ..., y50``
* test data       -> columns ``x, y``

Every loaded frame is validated and sorted by its x-values so that the
downstream components can rely on a clean, deterministic input.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from exceptions import DataValidationError

#: Column names that are accepted for the x-axis of an input CSV file.
X_COLUMN_NAMES: tuple[str, ...] = ("x", "X", "x-value", "x_value")

#: Column names that are accepted for the y-axis of an input CSV file.
Y_COLUMN_NAMES: tuple[str, ...] = ("y", "Y", "y-value", "y_value")


class DataLoader:
    """Loads and validates all CSV input files.

    Parameters
    ----------
    training_files:
        Either the four training CSV files (one x-y pair per line each),
        or a single CSV file with the columns ``x, y1, y2, y3, y4``.
    ideal_file:
        CSV file containing the fifty ideal functions (one x column and
        fifty y columns).
    test_file:
        CSV file containing the test data (one x-y pair per line).
    """

    def __init__(
        self,
        training_files: list[Path],
        ideal_file: Path,
        test_file: Path,
    ) -> None:
        self.training_files = [Path(f) for f in training_files]
        self.ideal_file = Path(ideal_file)
        self.test_file = Path(test_file)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def load_training_data(self) -> pd.DataFrame:
        """Load and merge the training datasets.

        Two input layouts are accepted. Either four separate files are
        given, each containing one x-y pair per line, or a single file
        with the columns ``x, y1, y2, y3, y4`` is given.
        In both cases a frame with the columns
        ``x, y1, y2, y3, y4`` is returned.

        Returns
        ------
        pandas.DataFrame
            The merged training data, sorted by ``x``.

        Raises
        ------
        FileNotFoundError
            If a training file does not exist.
        DataValidationError
            If the number of training files is neither one nor four, a
            file has an unexpected structure, a combined file does not
            contain exactly four y-columns, or the x-values of the four
            files do not overlap.
        """
        if len(self.training_files) not in (1, 4):
            raise DataValidationError(
                "Expected either 4 training files or 1 combined file "
                f"(columns x, y1..y4), got {len(self.training_files)}."
            )

        if len(self.training_files) == 4:
            frames: list[pd.DataFrame] = []
            for index, path in enumerate(self.training_files, start=1):
                frame = self._read_xy_csv(path, single_column=True)
                frame = frame.rename(columns={frame.columns[1]: f"y{index}"})
                frames.append(frame)

            merged = frames[0]
            for frame in frames[1:]:
                merged = pd.merge(merged, frame, on="x", how="inner")

            if merged.empty:
                raise DataValidationError(
                    "The x-values of the four training datasets do not "
                    "overlap; the merged training table is empty."
                )
            return merged.sort_values("x").reset_index(drop=True)

        # Single combined file with the columns x, y1, y2, y3, y4.
        frame = self._read_xy_csv(self.training_files[0], single_column=False)
        y_columns = [column for column in frame.columns if column != "x"]
        if len(y_columns) != 4:
            raise DataValidationError(
                "A combined training file must contain exactly four "
                f"y-columns; found {len(y_columns)} in "
                f"{self.training_files[0]}."
            )
        for index, column in enumerate(y_columns, start=1):
            frame = frame.rename(columns={column: f"y{index}"})
        return frame.sort_values("x").reset_index(drop=True)

    def load_ideal_functions(self) -> pd.DataFrame:
        """Load the fifty ideal functions.

        Returns
        -------
        pandas.DataFrame
            Frame with columns ``x, y1, ..., y50``, sorted by ``x``.

        Raises
        ------
        FileNotFoundError
            If the ideal-function file does not exist.
        DataValidationError
            If the file does not contain an x-column and fifty y-columns.
        """
        frame = self._read_xy_csv(self.ideal_file, single_column=False)
        y_columns = [col for col in frame.columns if col != "x"]
        if len(y_columns) != 50:
            raise DataValidationError(
                f"Expected 50 ideal functions but found {len(y_columns)} "
                "y-columns in the ideal-function file."
            )
        return frame.sort_values("x").reset_index(drop=True)

    def load_test_data(self) -> pd.DataFrame:
        """Load the test data set.

        Returns
        -------
        pandas.DataFrame
            Frame with columns ``x, y``, sorted by ``x``.

        Raises
        ------
        FileNotFoundError
            If the test-data file does not exist.
        DataValidationError
            If the file does not have exactly one x- and one y-column.
        """
        frame = self._read_xy_csv(self.test_file, single_column=True)
        return frame.sort_values("x").reset_index(drop=True)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _read_xy_csv(path: Path, single_column: bool) -> pd.DataFrame:
        """Read one CSV file and validate its column structure.

        Parameters
        ----------
        path:
            Path of the CSV file to read.
        single_column:
            If ``True`` exactly one y-column is expected, otherwise any
            number of y-columns is accepted (used for the ideal file).

        Returns
        -------
        pandas.DataFrame
            The raw frame with a normalised ``x`` column name.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        DataValidationError
            If the structure of the file is invalid.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        # csv.Sniffer is used only to detect the delimiter; the actual
        # parsing is done by pandas for speed and robustness.
        try:
            with open(path, newline="", encoding="utf-8-sig") as handle:
                sample = handle.read(4096)
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t ").delimiter
        except (csv.Error, UnicodeDecodeError, OSError):
            delimiter = ","

        frame = pd.read_csv(path, delimiter=delimiter)

        if frame.shape[1] < 2:
            raise DataValidationError(
                f"File {path} does not contain at least two columns (x, y)."
            )

        # Normalise the x-column name.
        x_column = DataLoader._find_column(frame, X_COLUMN_NAMES, "x", path)
        frame = frame.rename(columns={x_column: "x"})

        if single_column:
            y_column = DataLoader._find_column(frame, Y_COLUMN_NAMES, "y", path)
            frame = frame[[frame.columns[0], y_column]]
        else:
            # Keep the x column and every other column as y-columns.
            frame = frame[[frame.columns[0]] + [c for c in frame.columns[1:]]]

        # Strip any whitespace or invalid characters from the y-column
        # names so that they can be used as SQL column names later.
        cleaned = {c: DataLoader._clean_column_name(c) for c in frame.columns}
        frame = frame.rename(columns=cleaned)

        # Convert to numeric and reject missing / non-numeric values.
        for column in frame.columns:
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.isna().any():
                raise DataValidationError(
                    f"File {path} contains non-numeric or missing values "
                    f"in column '{column}'."
                )
            frame[column] = converted

        return frame

    @staticmethod
    def _find_column(
        frame: pd.DataFrame, candidates: tuple[str, ...], fallback: str, path: Path
    ) -> str:
        """Return the first column of *frame* matching *candidates*.

        Parameters
        ----------
        frame:
            The raw frame read from the CSV file.
        candidates:
            Accepted column-name variants.
        fallback:
            Name used for the error message.
        path:
            Source file path (used for the error message only).

        Returns
        -------
        str
            The matched column name.

        Raises
        ------
        DataValidationError
            If no column matches the accepted names.
        """
        for name in candidates:
            if name in frame.columns:
                return name
        raise DataValidationError(
            f"File {path} is missing the '{fallback}' column; "
            f"expected one of {list(candidates)}."
        )

    @staticmethod
    def _clean_column_name(name: str) -> str:
        """Sanitise a column name for use as a SQL identifier.

        Parameters
        ----------
        name:
            Raw column name from the CSV header.

        Returns
        -------
        str
            Lower-cased name with every character that is not a letter,
            digit or underscore replaced by an underscore.
        """
        cleaned = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in str(name).strip()
        )
        return cleaned or "column"
