import numpy as np
from matplotlib import pyplot as plt
from tmm_kit.beam_splines.boundary_conditions import free_ends
from tmm_kit.beam_splines.constants import BEAM_STATE_SIZE
from tmm_kit.beam_splines.elements import BeamElementOnElasticFoundation
from tmm_kit.beam_splines.nodes import BeamDummyNode
from tmm_kit.beam_splines.solvers import StatisticalBeamSplineSolver

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
        element_class=BeamElementOnElasticFoundation,
        element_data={
            "start_foundation": SUPPORTS[:-1],
            "end_foundation": SUPPORTS[1:],
            "bandwidth": np.repeat(bandwidth, len(T) - 1),
        },
        node_class=BeamDummyNode,
        node_data={},
        boundary_conditions=free_ends(),
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

foundation_label = "Foundation"
foundation_style = {
    "linestyle": "--",
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

ax[0].plot(T, SUPPORTS, **foundation_style, alpha=0.5)

custom_legend(
    ax[-1],
    [foundation_style, *beam_styles],
    [foundation_label, *beam_labels],
)

plt.show()
