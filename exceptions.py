"""User-defined exceptions for the project.

This module defines the custom exception hierarchy used across the whole
program.  All user-defined exceptions inherit from a common base class so
that a caller can catch every project-specific error with a single
``except AssignmentError`` clause, while standard Python exceptions
(e.g. ``FileNotFoundError``, ``ValueError``) are still raised where they
are the most accurate description of the problem.
"""


class AssignmentError(Exception):
    """Base class for all user-defined exceptions of this project.

    Every custom exception raised inside the program derives from this
    class.  It allows callers to handle all project-specific failures in
    one place without catching unrelated ``Exception`` subclasses.
    """


class DataValidationError(AssignmentError):
    """Raised when an input CSV file does not match the expected schema.

    Typical triggers are missing ``x``/``y`` columns, non-numeric
    values, empty files or files that cannot be parsed as two-column
    x-y data.
    """


class DatabaseOperationError(AssignmentError):
    """Raised when a SQLite/SQLAlchemy operation fails.

    This exception wraps the underlying SQLAlchemy exception so that
    callers do not have to depend on SQLAlchemy internals.
    """


class SelectionError(AssignmentError):
    """Raised when the least-squares selection of ideal functions fails.

    Typical triggers are an empty x-overlap between the training data
    and the ideal functions, or training/ideal tables without data.
    """


class MappingError(AssignmentError):
    """Raised when the mapping of test data to ideal functions fails.

    Typical triggers are an empty ideal-function lookup or an
    inconsistent selection result handed to the mapper.
    """


class VisualizationError(AssignmentError):
    """Raised when a Bokeh plot cannot be created or saved."""
