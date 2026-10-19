# FunctionSelector-TestdataMapper

A Python program that selects the four best ideal functions 
out of fifty that were provided, and maps test data against 
the sqrt(2) x max-deviation criterion, with SQLite 
persistence and interactive Bokeh visualization.

* it loads four training datasets, one test dataset and fifty ideal
  functions from CSV files,
* it creates a SQLite database with SQLAlchemy and stores the training
  data (`x, y1..y4`), the fifty ideal functions (`x, y1..y50`) and the
  mapping results (`x, y, delta_y, ideal_func_no`),
* it selects the **four best ideal functions** with the
  **least-squares criterion** (minimum sum of squared y-deviations),
* it maps every test point to one of the four selected ideal functions 
  if its deviation does not exceed `sqrt(2) × max_deviation` of the
  corresponding training function,
* it visualizes training data, test data, selected ideal functions and 
  the deviations with **Bokeh** (interactive HTML),
* it is object-oriented, inheritance containing, standard/user-defined
  exception handling and is a full unit-test suite.

## Project Structure

```
FunctionSelector-TestdataMapper/
├── main.py                 # command-line entry point
├── config.py               # central configuration (paths, tolerance)
├── exceptions.py           # custom exception hierarch
├── base.py                 # abstract base class AnalysisStep
├── data_loader.py          # CSV loading and validation
├── selection.py            # least-squares selection of ideal functions
├── mapping.py              # test-point mapping (sqrt(2) criterion)
├── database.py             # SQLite storage via SQLAlchemy
├── visualization.py        # Bokeh visualization (HTML)
├── pipeline.py             # end-to-end orchestration
├── generate_sample_data.py # OPTIONAL sample-data generator
├── verify_html.py          # check of the generated HTML
└── tests/                  # unit tests (unittest)
```

## Installation

```bash
python -m pip install -r requirements.txt
```

Requires Python 3.10 or newer.

## Usage BEFORE official data available

The project contains an optional sample-data-generator that creates 
plausible placeholder data. 
Run the generator first if the official data are NOT yet available.

```bash
python generate_sample_data.py
python main.py
```

The synthetic datasets are designed so that four of the fifty ideal
functions are the exact generating functions of the four training data
sets (with added noise); the least-squares selection therefore recovers
exactly those four functions, which makes the demonstration verifiable.

## Usage AFTER official data available

Replace ALL sample datasets with the official datasets in the 
`data/` directory using these names:

```
data/train1.csv   # x, y
data/train2.csv   # x, y
data/train3.csv   # x, y
data/train4.csv   # x, y
data/ideal.csv    # x, y1 .. y50
data/test.csv     # x, y
```

Run the complete pipeline:

```bash
python main.py
```

Alternatively, a combined training file with the columns
`x, y1, y2, y3, y4` (a single `train.csv`) is also accepted 
in place of the four train*.csv files, but then run as follows:

```bash
python main.py --training-files data/train.csv
```

The program prints a summary, writes a SQLite database
(`output/SQLite_database.db`) and saves a interactive visualization
(`output/visualization.html`).

## Running the unit tests

```bash
python -m unittest discover -s tests -v
```

## Notes on the design decisions

* **Selection**: for every training function the ideal function with the
  smallest sum of squared deviations is chosen (one choice per training
  function according to the mapping criterion).
* **Mapping**: a test point is assigned to the ideal function with the
  smallest absolute deviation, provided that this deviation does not
  exceed `sqrt(2) × max_deviation` of the corresponding training
  function; otherwise the point remains unmapped and is reported.
* **Ideal value lookup**: the ideal value is read from the grid; if a
  test x-value is not on the grid, linear interpolation (with boundary
  clamping) is used.
* **Database**: SQLAlchemy Core is used to create the SQLite schema;
  the test results are inserted row by row as demanded.
* **visualization**: the deviation of every mapped test point is shown
  as a vertical error bar between the test value and the ideal value.
