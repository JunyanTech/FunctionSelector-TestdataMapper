"""Configuration for the program.

This module centralises every path and tuning parameter of the program.
A single :class:`Config` object is created in :mod:`main` and passed to
all components, so that changing a file name or the mapping tolerance
never requires a code change in the business logic.

Default layout
--------------
The program expects the following files inside the data directory
(overridable through the command line, see :mod:`main`)::

    data/
        train1.csv   # first  training data set  (x, y)
        train2.csv   # second training data set  (x, y)
        train3.csv   # third  training data set  (x, y)
        train4.csv   # fourth training data set  (x, y)
        ideal.csv    # the 50 ideal functions    (x, y1..y50)
        test.csv     # the test data             (x, y)

    Alternatively, a combined training file with the columns
    ``x, y1, y2, y3, y4`` is accepted in place of the four per-function
    files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Central configuration object of the program.

    Attributes
    ----------
    data_dir:
        Directory that contains the input CSV files.
    db_path:
        Path of the SQLite database file that is created by the program.
    output_dir:
        Directory into which the Bokeh HTML visualization is written.
    training_files:
        Explicit list of the training CSV files: either the four
        per-function files or a single combined file with the columns
        ``x, y1..y4`` (resolved lazily if the default names are used).
    ideal_file:
        CSV file with the 50 ideal functions.  If ``None``, the file
        ``ideal.csv`` inside :attr:`data_dir` is used.
    test_file:
        CSV file with the test data.  If ``None``, the file ``test.csv``
        inside :attr:`data_dir` is used.
    mapping_tolerance:
        Factor that multiplies the largest training deviation to obtain
        the allowed maximum deviation for a test point (sqrt(2)).
    restrict_to_training_x_range:
        If ``True``, test points whose x-value lies outside the training
        x-range are never mapped.  Defaults to ``False`` because it is
        defined that the mapping criterion purely through the
        deviation factor; the flag is provided as an optional, documented
        safeguard.
    show_plot:
        If ``True``, the generated Bokeh plot is opened in the default
        browser after it has been saved.
    """

    data_dir: Path = Path("data")
    db_path: Path = Path("output/SQLite_database.db")
    output_dir: Path = Path("output")
    training_files: list[Path] = field(default_factory=list)
    ideal_file: Path | None = None
    test_file: Path | None = None
    mapping_tolerance: float = math.sqrt(2.0)
    restrict_to_training_x_range: bool = False
    show_plot: bool = True

    #: Default file names of the four training data sets.
    DEFAULT_TRAINING_NAMES: tuple[str, ...] = (
        "train1.csv",
        "train2.csv",
        "train3.csv",
        "train4.csv",
    )

    def __post_init__(self) -> None:
        """Resolve relative paths and default file names."""
        # Normalise all paths to absolute paths relative to the current
        # working directory so that every component works independently
        # of the folder from which it is imported.
        self.data_dir = Path(self.data_dir).resolve()
        self.db_path = Path(self.db_path).resolve()
        self.output_dir = Path(self.output_dir).resolve()

        # The ideal and test files default to the data directory so that
        # overriding --data-dir alone switches the whole input data set.
        self.ideal_file = (
            self.data_dir / "ideal.csv"
            if self.ideal_file is None
            else Path(self.ideal_file).resolve()
        )
        self.test_file = (
            self.data_dir / "test.csv"
            if self.test_file is None
            else Path(self.test_file).resolve()
        )

        if not self.training_files:
            self.training_files = [
                self.data_dir / name for name in self.DEFAULT_TRAINING_NAMES
            ]
        else:
            self.training_files = [Path(p).resolve() for p in self.training_files]

    def ensure_directories(self) -> None:
        """Create the output directory if it does not exist yet.

        Raises
        ------
        OSError
            If the directory cannot be created (e.g. missing write
            permission).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
