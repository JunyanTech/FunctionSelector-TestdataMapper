"""Generate synthetic example data for local demonstration.

.. note::

   This module is an **optional helper** for local testing.  
   When the official datasets are available, 
   place them in the ``data/`` directory 
   and the program will use them unchanged.
   The script only creates plausible placeholder data so that the
   complete pipeline (database, selection, mapping, visualization) can
   be exercised before the official data are available.

The synthetic dataset is constructed so that four of the fifty ideal
functions are the exact generating functions of the four training datasets 
(with added noise), which makes the least-squares selection
recover exactly those four functions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_ideal_functions() -> list[callable]:
    """Return fifty analytic functions used as the ideal functions.

    The list contains ten linear, ten quadratic, ten cubic, ten
    sine-based and ten exponential functions -- a heterogeneous pool
    from which the least-squares selection has to pick the best fits.

    Returns
    -------
    list[callable]
        Fifty functions mapping a float x to a float y.
    """
    functions: list[callable] = []

    # 10 linear functions: f(x) = i*x + 5
    functions.extend(lambda x, i=i: i * x + 5.0 for i in range(10))
    # 10 quadratic functions: f(x) = i*x**2 + 1
    functions.extend(lambda x, i=i: i * x**2 + 1.0 for i in range(10))
    # 10 cubic functions: f(x) = 0.1*i*x**3 + 2
    functions.extend(lambda x, i=i: 0.1 * i * x**3 + 2.0 for i in range(10))
    # 10 sine-based functions: f(x) = sin((i+1)*x/5) + 2*i
    functions.extend(
        lambda x, i=i: np.sin((i + 1) * x / 5.0) + 2.0 * i for i in range(10)
    )
    # 10 exponential functions: f(x) = i*exp(x/20) + 1
    functions.extend(lambda x, i=i: i * np.exp(x / 20.0) + 1.0 for i in range(10))

    if len(functions) != 50:
        raise AssertionError(
            f"Expected 50 ideal functions, got {len(functions)}."
        )
    return functions


def generate(data_dir: Path, seed: int = 42) -> None:
    """Create the example CSV files inside *data_dir*.

    Parameters
    ----------
    data_dir:
        Directory into which ``train1.csv`` ... ``train4.csv``,
        ``ideal.csv`` and ``test.csv`` are written.
    seed:
        Random seed for reproducible noise.

    Raises
    ------
    OSError
        If the data directory cannot be created.
    """
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Shared x-grid (the assignment datasets share one x-axis).
    x = np.round(np.arange(0.0, 20.0, 0.2), 4)  # 100 points

    ideal_functions = make_ideal_functions()
    ideal_frame = pd.DataFrame({"x": x})
    for number, function in enumerate(ideal_functions, start=1):
        ideal_frame[f"y{number}"] = np.round(
            np.array([function(xi) for xi in x]), 6
        )
    ideal_frame.to_csv(data_dir / "ideal.csv", index=False)

    # The four generating functions are ideal functions no. 2, 12, 22, 32.
    training_base = [1, 11, 21, 31]
    for index, base_index in enumerate(training_base, start=1):
        base_values = np.array(
            [ideal_functions[base_index](xi) for xi in x]
        )
        noisy = base_values + rng.normal(0.0, 0.4, size=len(x))
        training_frame = pd.DataFrame(
            {"x": x, "y": np.round(noisy, 6)}
        )
        training_frame.to_csv(data_dir / f"train{index}.csv", index=False)

    # Test data: points of the four generating functions with small
    # noise (to be mapped) plus clearly deviating points (to be
    # rejected by the sqrt(2) criterion).
    test_x = np.round(rng.uniform(0.0, 19.0, size=90), 4)
    test_y: list[float] = []
    for i, xi in enumerate(test_x):
        if i < 80:
            # 20 points per generating function with small noise.
            base_index = training_base[i // 20]
            value = ideal_functions[base_index](float(xi))
            test_y.append(value + rng.normal(0.0, 0.3))
        else:
            # 10 strongly deviating points that clearly violate the
            # sqrt(2) criterion and therefore must remain unmapped.
            value = ideal_functions[training_base[i % 4]](float(xi))
            test_y.append(value + rng.normal(0.0, 15.0))
    test_frame = pd.DataFrame(
        {"x": np.round(test_x, 4), "y": np.round(test_y, 6)}
    )
    test_frame.to_csv(data_dir / "test.csv", index=False)

    print(f"Example data written to: {data_dir.resolve()}")


if __name__ == "__main__":
    generate(Path("data"))
