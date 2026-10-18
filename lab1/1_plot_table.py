import pandas as pd
from matplotlib import pyplot as plt

from common.plot import save_figure


THEORETICAL_STYLE = {
    "linestyle": "--",
    "marker": "x",
    "markersize": 10,
}

MEASURED_STYLE = {
    "linestyle": "-",
    "marker": ".",
}


def main():
    """Plot response values from the bandwidth experiment table."""
    table = pd.read_csv("tables/1_bandwidth.csv")
    table = table[table["meas_step_divider"] == 5]

    fig, ax = plt.subplots(figsize=(6, 4))
    bandwidth_divider = table["bandwidth_divider"]

    ax.plot(
        bandwidth_divider,
        table["response_use_ex"],
        label=r"$\varepsilon_{use}$ Theoretical",
        **THEORETICAL_STYLE,
        color="tab:green",
    )
    ax.plot(
        bandwidth_divider,
        table["response_use_meas"],
        label=r"$\varepsilon_{use}$ Measured",
        **MEASURED_STYLE,
        color="tab:green",
    )
    ax.plot(
        bandwidth_divider,
        table["response_err_ex"],
        label=r"$\varepsilon_{err}$ Theoretical",
        **THEORETICAL_STYLE,
        color="tab:red",
    )
    ax.plot(
        bandwidth_divider,
        table["response_err_meas"],
        label=r"$\varepsilon_{err}$ Measured",
        **MEASURED_STYLE,
        color="tab:red",
    )

    ax.set_xlabel(r"Bandwidth Divider ($g = T_{err} / k$)")
    ax.set_ylabel(r"Response ($\varepsilon$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "plots", "response")


if __name__ == "__main__":
    main()
