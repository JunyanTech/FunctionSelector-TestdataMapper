"""Command-line entry point of the program.

Usage::

    python main.py [--data-dir DIR] [--db-path PATH] [--output-dir DIR]
                   [--ideal-file PATH] [--test-file PATH]
                   [--training-files F1 F2 F3 F4 | FILE] [--no-show]
                   [--quiet]

If no explicit file paths are given, the program expects the files
``train1.csv`` ... ``train4.csv``, ``ideal.csv`` and ``test.csv`` in
the data directory (see :mod:`config`).  Alternatively, a single
combined training file with the columns ``x, y1..y4`` 
can be passed via ``--training-files FILE``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import Config
from exceptions import AssignmentError
from pipeline import AssignmentPipeline


def parse_arguments() -> argparse.Namespace:
    """Parse and return the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Select the four best ideal functions,"
            "map the test data and store everything in SQLite."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the CSV input files (default: data).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("output/SQLite_database.db"),
        help="Path of the SQLite database file (default: output/SQLite_database.db).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for the generated contents (default: output).",
    )
    parser.add_argument(
        "--ideal-file",
        type=Path,
        default=None,
        help="CSV file with the fifty ideal functions (default: <data-dir>/ideal.csv).",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="CSV file with the test data (default: <data-dir>/test.csv).",
    )
    parser.add_argument(
        "--training-files",
        type=Path,
        nargs="+",
        default=None,
        metavar="FILE",
        help=(
            "Either the four training CSV files in order, or a single "
            "combined file with the columns x, y1..y4"
            "If omitted, train1.csv ... train4.csv inside "
            "--data-dir are used."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the visualization in the browser after saving.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human readable summary output.",
    )
    return parser.parse_args()


def build_config(arguments: argparse.Namespace) -> Config:
    """Create the :class:`Config` from parsed arguments."""
    training_files = (
        list(arguments.training_files)
        if arguments.training_files is not None
        else []
    )
    if training_files and len(training_files) not in (1, 4):
        raise argparse.ArgumentError(
            arguments.training_files,
            "expected either 4 training files or 1 combined file "
            "(columns x, y1..y4).",
        )
    return Config(
        data_dir=arguments.data_dir,
        db_path=arguments.db_path,
        output_dir=arguments.output_dir,
        training_files=training_files,
        ideal_file=arguments.ideal_file,
        test_file=arguments.test_file,
        show_plot=not arguments.no_show,
    )


def print_summary(summary: dict) -> None:
    """Print a compact, human readable summary of the run."""
    print("=" * 60)
    print("Selection of Four Ideal Functions and Mapping of Test Data")
    print("=" * 60)
    print(f"Loaded training rows        : {summary['training_rows']}")
    print(f"Loaded ideal-function rows  : {summary['ideal_rows']}")
    print(f"Test points evaluated       : {summary['test_points']}")
    print("-" * 60)
    print("Chosen ideal functions (least squares):")
    for selection in summary["selections"]:
        print(
            f"  training {selection['training_func_no']} -> ideal "
            f"{selection['ideal_func_no']:>2} | SSE {selection['sse']:.4f} | "
            f"max_dev {selection['max_deviation']:.4f} | "
            f"threshold {selection['threshold']:.4f}"
        )
    print("-" * 60)
    print(f"Mapped test points       : {summary['mapped_points']}")
    print(f"Unmapped test points     : {summary['unmapped_points']}")
    mapping_summary = summary.get("mapping_summary")
    if mapping_summary is not None:
        print(
            f"Mean absolute deviation   : "
            f"{mapping_summary['mean_abs_deviation']:.4f}"
        )
        print(
            f"Max absolute deviation    : "
            f"{mapping_summary['max_abs_deviation']:.4f}"
        )
        print(
            "Mapped by ideal function : "
            f"{mapping_summary['per_ideal_func']}"
        )
    print(f"Database                 : {summary['database_path']}")
    print(f"visualization            : {summary['plot_path']}")
    print("=" * 60)


def main() -> int:
    """Run the pipeline and return the process exit code."""
    arguments = parse_arguments()
    config = build_config(arguments)

    try:
        pipeline = AssignmentPipeline(config)
        summary = pipeline.run()
        if not arguments.quiet:
            print_summary(summary)
        return 0
    except AssignmentError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except FileNotFoundError as error:
        print(f"ERROR: input file missing - {error}", file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - top-level safety net
        print(f"UNEXPECTED ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
