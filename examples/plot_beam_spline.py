import numpy as np
from matplotlib import pyplot as plt
from tmm_kit.beam_splines.boundary_conditions import pinned_ends
from tmm_kit.beam_splines.constants import BEAM_STATE_SIZE
from tmm_kit.beam_splines.elements import BeamElement
from tmm_kit.beam_splines.nodes import BeamPinnedSupport
from tmm_kit.beam_splines.solvers import StatisticalBeamSplineSolver

from common.constants import BEAM_STATE_NAMES
from common.plot import plot_tmm_element, tmm_element_subplots

# Given
T = [2, 4, 5]
SUPPORTS = [-1, 4, 2]

# Solve
spline = StatisticalBeamSplineSolver(
    t=T,
    element_class=BeamElement,
    element_data={},
    node_class=BeamPinnedSupport,
    node_data={"displacement": SUPPORTS[1:-1]},
    boundary_conditions=pinned_ends(SUPPORTS[0], SUPPORTS[-1]),
)

result = spline.solve()

# Evaluate
eval_t = np.linspace(T[0], T[-1], 128)
eval_states = result.evaluate(eval_t)
edge_states = result.get_all_edge_states()

# Show
fig, ax = tmm_element_subplots(
    BEAM_STATE_SIZE,
    BEAM_STATE_NAMES,
    xlabel="t",
)

plot_tmm_element(
    ax,
    eval_t,
    eval_states,
    T,
    edge_states,
)
plt.show()
