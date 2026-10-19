"""Shared abstract base classes for the analysis steps.

The module defines :class:`AnalysisStep`, an abstract base class that
captures the common contract of every analysis stage of the program
(currently the least-squares selection and the test-data mapping).
Concrete steps inherit from it and only implement :meth:`run`.
"""

from __future__ import annotations

import abc


class AnalysisStep(abc.ABC):
    """Abstract base class of a single analysis stage.

    A step is a reusable, self-contained unit that transforms one or
    more input data frames into a well-defined result.  Subclasses
    document their exact inputs and outputs in :meth:`run`.
    """

    #: Short human readable name used in log output.
    name: str = "analysis step"

    @abc.abstractmethod
    def run(self) -> object:
        """Execute the analysis step and return its result.

        The concrete signature (parameters and return type) is defined
        by every subclass in its own docstring.
        """

    def describe(self) -> str:
        """Return a short description of the step for log messages."""
        return self.name
