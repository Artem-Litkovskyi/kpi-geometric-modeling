import numpy as np
from matplotlib import pyplot as plt
from tmm_kit.beam_splines.constants import BEAM_STATE_SIZE

from common.constants import BEAM_STATE_NAMES
from common.helpers import (
    statistical_smoothing_spline_adaptive,
)
from common.plot import (
    custom_legend,
    plot_tmm_element,
    tmm_element_subplots,
)

# Given
T = [2, 4, 5]
SUPPORTS = [-1, 4, 2]

# Solve
bandwidths = [0.5, 1.0]

results = []
for bandwidth in bandwidths:
    spline = statistical_smoothing_spline_adaptive(T, SUPPORTS, bandwidth)
    results.append(spline.solve())

# Evaluate
eval_t = np.linspace(T[0], T[-1], 256)

eval_states_list = []
edge_states_list = []

for result in results:
    eval_states_list.append(result.evaluate(eval_t))
    edge_states_list.append(result.get_all_edge_states())

# Show
fig, ax = tmm_element_subplots(
    BEAM_STATE_SIZE,
    BEAM_STATE_NAMES,
    xlabel="t",
)

beam_labels = [f"Bandwidth: {g}" for g in bandwidths]
beam_styles = [{"color": "tab:blue"}, {"color": "tab:orange"}]

supports_label = "Supports"
supports_style = {
    "marker": "o",
    "color": "k",
}

for eval_states, edge_states, style in zip(
        eval_states_list, edge_states_list, beam_styles):
    plot_tmm_element(
        ax,
        eval_t,
        eval_states,
        T,
        edge_states,
        plot_style=style,
    )

ax[0].scatter(T, SUPPORTS, **supports_style)

custom_legend(
    ax[-1],
    [supports_style, *beam_styles],
    [supports_label, *beam_labels],
)

plt.show()
