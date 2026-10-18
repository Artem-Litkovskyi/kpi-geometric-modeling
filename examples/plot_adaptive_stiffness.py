import numpy as np
from matplotlib import pyplot as plt
from tmm_kit.beam_splines.boundary_conditions import elastic_ends
from tmm_kit.beam_splines.elements import BeamElement
from tmm_kit.beam_splines.nodes import BeamElasticSupport
from tmm_kit.beam_splines.solvers import StatisticalBeamSplineSolver
from tmm_kit.beam_splines.utils import adaptive_stiffness

from common.constants import BEAM_STATE_NAMES
from common.helpers import statistical_smoothing_spline_adaptive
from common.plot import (
    custom_legend,
    plot_tmm_element,
    tmm_element_subplots,
    save_figure,
)
from lab1.base import cos_signal_period

delta_t_0 = 0.3
delta_t_k = 0.05

data_ts = [0.0]
for i in range(69):
    data_ts.append(data_ts[-1] + delta_t_0 + delta_t_k * i)
data_ts = np.asarray(data_ts, dtype=np.float64)

amp = 0.04
xi = 4.472

data_ws = amp * np.sin(data_ts / xi)

# Solve
bandwidth = cos_signal_period(xi) / 10.0

avg_l = np.mean(np.diff(data_ts))
constant_stiffness = adaptive_stiffness(avg_l, bandwidth)

spline_const = StatisticalBeamSplineSolver(
    t=data_ts,
    element_class=BeamElement,
    element_data={},
    node_class=BeamElasticSupport,
    node_data={
        "displacement": data_ws[1:-1],
        "stiffness": np.repeat(constant_stiffness, len(data_ts) - 2),
    },
    boundary_conditions=elastic_ends(
        data_ws[0],
        data_ws[-1],
        constant_stiffness,
        constant_stiffness,
    ),
)

spline_adaptive = statistical_smoothing_spline_adaptive(data_ts, data_ws, bandwidth)

result_const = spline_const.solve()
result_adaptive = spline_adaptive.solve()

# Evaluate
eval_ts = np.linspace(data_ts[0], data_ts[-1], 256)
eval_states_const = result_const.evaluate(eval_ts)
eval_states_adaptive = result_adaptive.evaluate(eval_ts)

# Show
fig, ax = tmm_element_subplots(
    1,
    [BEAM_STATE_NAMES[0]],
    figsize=(8, 3),
    xlabel="t",
)

data_style = {"linestyle": "None", "marker": "x", "color": "k"}
const_style = {"color": "tab:red"}
adaptive_style = {"color": "tab:green"}

ax[0].scatter(data_ts, data_ws, **data_style)

plot_tmm_element(
    ax,
    eval_ts,
    eval_states_const,
    state_mask=[0],
    plot_style=const_style,
)

plot_tmm_element(
    ax,
    eval_ts,
    eval_states_adaptive,
    state_mask=[0],
    plot_style=adaptive_style,
)

custom_legend(
    ax[-1],
    [data_style, const_style, adaptive_style],
    [
        r"Measured ($\Delta_i = \Delta_{i-1} + 0.3 + 0.05 \cdot i$)",
        r"Constant Stiffness ($D = \frac{\Delta_{avg}}{g^4}$)",
        "Adaptive ($g = T$)",
    ],
    loc="center right"
)

plt.show()
save_figure(fig, "path", "0_const_vs_adaptive")
