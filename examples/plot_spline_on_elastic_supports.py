import numpy as np
from matplotlib import pyplot as plt
from tmm_kit.beam_splines.boundary_conditions import pinned_ends
from tmm_kit.beam_splines.constants import BEAM_STATE_SIZE
from tmm_kit.beam_splines.elements import BeamElement
from tmm_kit.beam_splines.nodes import BeamElasticSupport
from tmm_kit.beam_splines.solvers import StatisticalBeamSplineSolver
from tmm_kit.beam_splines.utils import adaptive_stiffness

from common.constants import BEAM_STATE_NAMES
from common.plot import custom_legend, plot_tmm_element, tmm_element_subplots

# Given
T = [2, 4, 5]
SUPPORTS = [-1, 4, 2]

# Solve
bandwidths = [0.5, 1.0]

results = []
for bandwidth in bandwidths:
    spline = StatisticalBeamSplineSolver(
        t=T,
        element_class=BeamElement,
        element_data={},
        node_class=BeamElasticSupport,
        node_data={
            "displacement": SUPPORTS[1:-1],
            "stiffness": adaptive_stiffness(np.diff(T)[1:], bandwidth),
        },
        boundary_conditions=pinned_ends(SUPPORTS[0], SUPPORTS[-1]),
    )

    results.append(spline.solve())

# Evaluate
eval_t = np.linspace(T[0], T[-1], 128)

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

for eval_states, edge_states, bandwidth, style in zip(
        eval_states_list, edge_states_list, bandwidths, beam_styles):
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
