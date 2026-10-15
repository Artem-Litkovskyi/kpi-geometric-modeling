# AI Agent Guidelines

This document outlines the architectural patterns and mathematical precision rules for this project. 

## 1. Core Stack
* **Language:** Python 3.13
* **Libraries:** `numpy` (arrays), `scipy` (solvers).

## 2. Type Hinting & Precision
* **Scalars:** Use Python's built-in `float` for scalar type hints, not `np.float64`.
* **Arrays:** Allow flexible sequence inputs `Sequence[float] | np.ndarray`. Normalize them internally using `np.asarray(points, dtype=np.float64)`.
* **Scalar OR Array:** Use `np.typing.ArrayLike` when the function expects either a one value or an array of them. Use `np.isscalar(val)` for scalar check. Use `scalar = float(arr.item())` to get the scalar value.

## 3. Documentation
* **Format:** Strictly use **NumPy-style** docstrings. Do not duplicate types in the docstring (type hints are the single source of truth).
* **Content:** All public functions require comprehensive docstrings detailing mathematical constraints and expected array shapes (e.g., `(N, 3)`). If the function is simple and its arguments are self-explanatory, the `Prameters` and `Returns` sections can be omitted. If the function is private the docstring can be absent.

## 4. Post-Editing Actions
* **Linting:** Use `ruff check --fix .` command to automatically fix trivial rule breaks.

## 5. Standard Code Template Example

```python
from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


class SomeClass:
    """Class docstring."""
    
    def __magic_methods__(self):
        pass

    def a_method(self):
        """Then normal methods, in order of importance."""

    def _private_method(self):
        pass


def process_coordinates(x: np.typing.ArrayLike) -> float | np.ndarray:
    """Do something.
    
    A description with more details.
    
    Parameters
    ----------
    x
        Coordinate (coordinates).
    """
    x_arr = np.asarray(x)
    
    if np.isscalar(x):
        return float(x_arr.item())
    return x_arr
```