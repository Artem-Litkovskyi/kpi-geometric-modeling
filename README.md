# Geometric Modeling

This repository contains a Python implementation of studied at the **Geometric Modeling** university course.

## 🚀 Getting Started

1. Install the library from GitHub:

   ```bash
   python -m pip install https://github.com/Artem-Litkovskyi/kpi-geometric-modeling.git
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   ```

   On Linux/macOS:

   ```bash
   source .venv/bin/activate
   ```

   On Windows:

   ```powershell
   .venv\Scripts\activate
   ```

### Normal Setup

3. Install the required dependencies:

    ``` bash
    python -m pip install -e .
    python -m pip install --group dev
    ```

### Development Setup

3. Manually install dependencies listed in `pyproject.toml`. Do NOT install `tmm-kit`.
4. Install `tmm-kit` separately from GitHub in the same directory as this project:

    ``` bash
    cd ../
    git clone https://github.com/Artem-Litkovskyi/tmm-kit.git
    ```

5. Follow the "Development Setup" guide from `tmm-kit/README.md`
6. Return into the directory of this project and install `tmm-kit` as an editable dependency:

    ```bash
    cd ../kpi-geometric-modeling
    pip install -e ../tmm-kit
    ```
