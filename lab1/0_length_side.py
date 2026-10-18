import numpy as np
from tmm_kit.beam_splines.boundary_conditions import elastic_ends
from tmm_kit.beam_splines.elements import BeamElement
from tmm_kit.beam_splines.nodes import BeamElasticSupport
from tmm_kit.beam_splines.solvers import StatisticalBeamSplineSolver
from tmm_kit.beam_splines.utils import adaptive_stiffness

from common.constants import BEAM_STATE_NAMES
from common.plot import (
    custom_legend,
    plot_tmm_element,
    save_figure,
    tmm_element_subplots,
)
from lab1.base import cos_signal, cos_signal_moment, cos_signal_period, uneven_points

# Given
amp_use = 1.0
xi_use = np.sqrt(10)
period_use = cos_signal_period(xi_use)

meas_step_avg = period_use / 5.0
meas_step_min = 0.5 * meas_step_avg
meas_step_max = 1.5 * meas_step_avg

periods_n_use = 12

eval_points_per_period_use = 100


# Precalculate
max_t = periods_n_use * period_use
t_meas = uneven_points(meas_step_min, meas_step_max, max_t)

eval_points_n = int(periods_n_use * eval_points_per_period_use + 1)
t_eval = np.linspace(t_meas[0], t_meas[-1], eval_points_n)


# Reference
y_use_ref = cos_signal(t_eval, amp_use, xi_use)
moment_use_ref = cos_signal_moment(t_eval, amp_use, xi_use)


# Measured
y_meas = cos_signal(t_meas, amp_use, xi_use)


# Smooth
bandwidth = period_use / 5.0

element_lengths = np.diff(t_meas)
const_stiffness = adaptive_stiffness(np.mean(element_lengths), bandwidth)

spline_const = StatisticalBeamSplineSolver(
    t=t_meas,
    element_class=BeamElement,
    element_data={},
    node_class=BeamElasticSupport,
    node_data={
        "displacement": y_meas[1:-1],
        "stiffness": np.repeat(const_stiffness, len(t_meas) - 2),
    },
    boundary_conditions=elastic_ends(
        y_meas[0],
        y_meas[-1],
        const_stiffness,
        const_stiffness,
    ),
)

spline_left = StatisticalBeamSplineSolver(
    t=t_meas,
    element_class=BeamElement,
    element_data={},
    node_class=BeamElasticSupport,
    node_data={
        "displacement": y_meas[1:-1],
        "stiffness": adaptive_stiffness(element_lengths[:-1], bandwidth),
    },
    boundary_conditions=elastic_ends(
        y_meas[0],
        y_meas[-1],
        adaptive_stiffness(element_lengths[0], bandwidth),
        adaptive_stiffness(element_lengths[-1], bandwidth),
    ),
)

spline_right = StatisticalBeamSplineSolver(
    t=t_meas,
    element_class=BeamElement,
    element_data={},
    node_class=BeamElasticSupport,
    node_data={
        "displacement": y_meas[1:-1],
        "stiffness": adaptive_stiffness(element_lengths[1:], bandwidth),
    },
    boundary_conditions=elastic_ends(
        y_meas[0],
        y_meas[-1],
        adaptive_stiffness(element_lengths[0], bandwidth),
        adaptive_stiffness(element_lengths[-1], bandwidth),
    ),
)

spline_center = StatisticalBeamSplineSolver(
    t=t_meas,
    element_class=BeamElement,
    element_data={},
    node_class=BeamElasticSupport,
    node_data={
        "displacement": y_meas[1:-1],
        "stiffness": adaptive_stiffness(
            (element_lengths[:-1] + element_lengths[1:]) / 2.0,
            bandwidth
        ),
    },
    boundary_conditions=elastic_ends(
        y_meas[0],
        y_meas[-1],
        adaptive_stiffness(element_lengths[0], bandwidth),
        adaptive_stiffness(element_lengths[-1], bandwidth),
    ),
)

result_const = spline_const.solve()
result_left = spline_left.solve()
result_right = spline_right.solve()
result_center = spline_center.solve()

y_const = result_const.evaluate(t_eval)
y_left = result_left.evaluate(t_eval)
y_right = result_right.evaluate(t_eval)
y_center = result_center.evaluate(t_eval)


# Show
fig, ax = tmm_element_subplots(
    1,
    [BEAM_STATE_NAMES[0]],
    figsize=(8, 3),
    xlabel="t",
)

use_ref_style = {"linestyle": "--", "color": "k", "alpha": 0.5}
ax[0].plot(t_eval, y_use_ref, **use_ref_style)

meas_style = {"linestyle": "None", "marker": "x", "color": "k"}
ax[0].scatter(t_meas, y_meas, **meas_style)

const_style = {"linestyle": "--", "color": "tab:blue"}
plot_tmm_element(
    ax,
    t_eval,
    y_const,
    state_mask=[0],
    plot_style=const_style,
)

left_style = {"color": "tab:red"}
plot_tmm_element(
    ax,
    t_eval,
    y_left,
    state_mask=[0],
    plot_style=left_style,
)

right_style = {"color": "tab:green"}
plot_tmm_element(
    ax,
    t_eval,
    y_right,
    state_mask=[0],
    plot_style=right_style,
)

center_style = {"color": "tab:purple"}
plot_tmm_element(
    ax,
    t_eval,
    y_center,
    state_mask=[0],
    plot_style=center_style,
)

custom_legend(
    ax[0],
    [use_ref_style, meas_style, const_style, left_style, right_style, center_style],
    [
        "Reference", r"Measured ($\Delta_{avg} = \frac{T}{5}$)", "Constant Stiffness",
        "Adaptive (Left Length)", "Adaptive (Right Length)", "Adaptive (Center)",
    ],
    loc="lower left",
)

ax[0].set_xlim((8 * period_use, 12 * period_use))
ax[0].set_ylim((-1.1, 1.1))

save_figure(fig, "plots", "0_length_side")
