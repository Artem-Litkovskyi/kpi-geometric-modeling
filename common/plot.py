from collections.abc import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

# noinspection unresolved-references
DEFAULT_COLORS = plt.colormaps["tab10"].colors

GRID_STYLE = {
    "alpha": 0.3,
}

ZERO_LINE_STYLE = {
    "color": "gray",
    "linestyle": "--",
    "alpha": 0.5,
}


def custom_legend(axes: Axes, styles: Sequence[dict], labels: Sequence[str]):
    """Apply custom legend to the axes."""
    lines = [Line2D([0], [0], **kwargs) for kwargs in styles]
    axes.legend(lines, labels)


def tmm_element_subplots(
    axes_number: int,
    titles: Sequence[str] | None = None,
    figsize: tuple = (10, 8),
    xlabel: str = "s",
) -> tuple[plt.Figure, np.ndarray]:
    """Create subplots for the TMM element.

    Parameters
    ----------
    axes_number
        The number of subplots.
    titles
        Optional sequence of titles.
    figsize
        The size of a figure.
    xlabel
        Common x label.

    Returns
    -------
    Figure and Axes array
    """
    if titles and len(titles) != axes_number:
        raise ValueError(
            f"titles length ({len(titles)}) does not match"
            f"the number of axes ({axes_number})"
        )

    fig, axes = plt.subplots(
        axes_number,
        1,
        sharex=True,
        figsize=figsize,
        squeeze=False,
    )
    axes = axes[:,0]

    for i, ax in enumerate(axes):
        ax.grid(True, **GRID_STYLE)

        if titles:
            ax.set_title(titles[i])

        if i == 0:
            continue

        ax.axhline(y=0, **ZERO_LINE_STYLE)

    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    return fig, axes


def plot_tmm_element(
    axes: Sequence[Axes],
    eval_ts: Sequence[float] | np.ndarray,
    eval_states: np.ndarray,
    edge_ts: Sequence[float] | np.ndarray | None = None,
    edge_states: np.ndarray | None = None,
    state_mask: Sequence[int] | None = None,
    plot_style: dict | Sequence[dict] | None = None,
) -> None:
    """Plot evaluated beam states and their node states.

    Parameters
    ----------
    axes
        Array of Axes with shape ``(A)`` created by :func:`beam_subplots`.
    eval_ts
        Evaluation coordinates with shape ``(N)``.
    eval_states
        Evaluated states array with shape ``(N, K)``.
        The rows must correspond to ``eval_ts``.
    edge_ts
        Optional coordinates of the spline element edges with shape ``(M)``.
    edge_states
        Optional array of all edge states with shape ``(2 * (M-1), K)``.
        Each pair of rows must correspond to ``node_ts``.
    state_mask
        Optional array of state variable indexes that will be plotted.
        Must match the number of axes ``(A)``.
    plot_style
        Optional plot kwargs or a sequence of kwargs per plot.
        Must match the number of axes ``(A)``.

    Raises
    ------
    ValueError
        If the axes_number of axes or corresponding array dimensions do not match.
    """
    axes = list(axes)

    eval_ts = np.asarray(eval_ts, dtype=np.float64)
    eval_states = np.asarray(eval_states, dtype=np.float64)

    if eval_ts.ndim != 1:
        raise ValueError("eval_ts must be one-dimensional")
    if eval_states.ndim != 2:
        raise ValueError("eval_states must be two-dimensional")
    if eval_ts.size != eval_states.shape[0]:
        raise ValueError("eval_ts and eval_states size mismatch")

    state_size = eval_states.shape[1]
    if state_mask is None:
        state_mask = list(range(state_size))
    elif len(state_mask) != len(axes):
        raise ValueError("state_mask length must match the number of axes")
    if len(state_mask) > state_size:
        raise ValueError("state_mask length must not exceed the state size")

    if edge_ts is not None and edge_states is not None:
        edge_ts = np.asarray(edge_ts, dtype=np.float64)
        edge_states = np.asarray(edge_states, dtype=np.float64)

        if edge_ts.ndim != 1:
            raise ValueError("edge_ts must be one-dimensional")
        if edge_states.ndim != 2:
            raise ValueError("edge_states must be two-dimensional")
        if 2 * (edge_ts.size - 1) != edge_states.shape[0]:
            raise ValueError("edge_ts and edge_states size mismatch")
        if edge_states.shape[1] != state_size:
            raise ValueError("edge_states and eval_states state size mismatch")

    if len(state_mask) != len(axes):
        raise ValueError("state_mask length must match the number of axes")

    if plot_style is None:
        plot_styles = [{"color": clr} for clr in DEFAULT_COLORS[:len(axes)]]
    elif isinstance(plot_style, Sequence):
        if len(plot_style) != len(axes):
            raise ValueError("plot_style length must match the number of axes")
        plot_styles = plot_style
    else:
        plot_styles = [plot_style] * len(axes)

    for axis_i, (state_i, ax) in enumerate(zip(state_mask, axes)):
        ax.plot(
            eval_ts,
            eval_states[:, state_i],
            **plot_styles[axis_i],
        )

    if edge_ts is None or edge_states is None:
        return

    edge_ts_start_end = np.concatenate(
        (
            # Element 0 start
            edge_ts[:1],
            # El. 0 end + el. 1 to (N-1) start and end + el. N start
            np.repeat(edge_ts[1:-1], 2),
            # Element N end
            edge_ts[-1:],
        )
    )

    for axis_i, (state_i, ax) in enumerate(zip(state_mask, axes)):
        ax.scatter(
            edge_ts_start_end,
            edge_states[:, state_i],
            **plot_styles[axis_i],
            marker="o",
        )
