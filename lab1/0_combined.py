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

amp_err = 0.05
xi_err = 1.0 / np.sqrt(10)
period_err = cos_signal_period(xi_err)

meas_step_avg = period_err / 5.0
meas_step_min = 0.01 * meas_step_avg
meas_step_max = 1.99 * meas_step_avg

periods_n_use = 2

eval_points_per_period_err = 16


# Precalculate
max_t = periods_n_use * period_use
t_meas = uneven_points(meas_step_min, meas_step_max, max_t)

periods_n_err = periods_n_use * period_use / period_err
eval_points_n = int(periods_n_err * eval_points_per_period_err + 1)
t_eval = np.linspace(t_meas[0], t_meas[-1], eval_points_n)


# Reference
y_use_ref = cos_signal(t_eval, amp_use, xi_use)
moment_use_ref = cos_signal_moment(t_eval, amp_use, xi_use)

y_meas_ref = cos_signal(t_eval, amp_use, xi_use) \
             + cos_signal(t_eval, amp_err, xi_err)
moment_meas_ref = cos_signal_moment(t_eval, amp_use, xi_use) \
                  + cos_signal_moment(t_eval, amp_err, xi_err)


# Measured
y_meas = cos_signal(t_meas, amp_use, xi_use) \
         + cos_signal(t_meas, amp_err, xi_err)


# Smooth
bandwidth = period_err

element_lengths = np.diff(t_meas)
spline = StatisticalBeamSplineSolver(
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

result = spline.solve()
result_smooth = result.evaluate(t_eval)


# Show
fig, ax = tmm_element_subplots(
    2,
    BEAM_STATE_NAMES[0::2],
    figsize=(8, 6),
    xlabel="t",
)

use_ref_style = {"linestyle": "--", "color": "k", "alpha": 0.5}
ax[0].plot(t_eval, y_use_ref, **use_ref_style)
ax[1].plot(t_eval, moment_use_ref, **use_ref_style)

meas_ref_style = {"linestyle": "--", "color": "tab:red", "alpha": 0.5}
ax[0].plot(t_eval, y_meas_ref, **meas_ref_style)
ax[1].plot(t_eval, moment_meas_ref, **meas_ref_style)

meas_style = {"linestyle": "None", "marker": "+", "color": "k"}
ax[0].scatter(t_meas, y_meas, **meas_style)

smooth_style = {"color": "tab:green"}
plot_tmm_element(
    ax,
    t_eval,
    result_smooth,
    state_mask=[0, 2],
    plot_style=smooth_style,
)

custom_legend(
    ax[0],
    [use_ref_style, meas_ref_style, meas_style, smooth_style],
    [
        "$y_{use}$", "$y_{use} + y_{err}$",
        r"Measured ($\Delta_{avg} = \frac{T_{err}}{5}$)", "Smoothed ($g = T_{err}$)"
    ],
    loc="lower left",
)

save_figure(fig, "plots", "0_combined")
