import numpy as np
from tmm_kit.beam_splines.utils import smoothing_spline_response

from common import table
from common.constants import BEAM_STATE_NAMES
from common.helpers import statistical_smoothing_spline_adaptive
from common.plot import (
    custom_legend,
    plot_tmm_element,
    save_figure,
    tmm_element_subplots,
)
from lab1.base import (
    cos_signal,
    cos_signal_moment,
    cos_signal_period,
    rms,
    uneven_points,
)

PATH_PLOTS = "plots"
PATH_TABLES = "tables"

COLOR_USE = "tab:green"
COLOR_ERR = "tab:red"


def main():
    """Run experiment."""
    table.reset_table(
        PATH_TABLES,
        "1_bandwidth",
        [
            "meas_step_divider",
            "bandwidth_divider",
            "rms_use_ref",
            "rms_use_smooth",
            "response_use_meas",
            "response_use_ex",
            "response_use_relative_error",
            "rms_err_ref",
            "rms_err_smooth",
            "response_err_meas",
            "response_err_ex",
            "response_err_relative_error",
        ],
    )

    # Given
    amp_use = 1.0
    xi_use = np.sqrt(10)
    period_use = cos_signal_period(xi_use)

    amp_err = 0.05
    xi_err = 1.0 / np.sqrt(10)
    period_err = cos_signal_period(xi_err)

    periods_n_use = 12

    eval_points_per_period_err = 16

    # Run
    meas_step_dividers = [5, 10]
    bandwidth_dividers = [1, 2, 4, 8]
    for j, meas_step_divider in enumerate(meas_step_dividers):
        for i, bandwidth_divider in enumerate(bandwidth_dividers):
            bandwidth = period_err / bandwidth_divider
            run_sub_experiment(
                j + 1,
                i + 1,
                amp_use,
                xi_use,
                period_use,
                periods_n_use,
                amp_err,
                xi_err,
                period_err,
                meas_step_divider,
                bandwidth,
                bandwidth_divider,
                eval_points_per_period_err,
            )


def run_sub_experiment(
    exp_id,
    exp_sub_id,

    amp_use,
    xi_use,
    period_use,
    periods_n_use,

    amp_err,
    xi_err,
    period_err,

    meas_step_divider,

    bandwidth,
    bandwidth_divider,

    eval_points_per_period_err,
):
    """Run a sub-experiment for a given bandwidth."""
    meas_step_avg = period_err / meas_step_divider
    meas_step_min = 0.01 * meas_step_avg
    meas_step_max = 1.99 * meas_step_avg

    max_t = periods_n_use * period_use
    t_meas = uneven_points(meas_step_min, meas_step_max, max_t)

    periods_n_err = periods_n_use * period_use / period_err
    eval_points_n = int(periods_n_err * eval_points_per_period_err + 1)
    t_eval = np.linspace(t_meas[0], t_meas[-1], eval_points_n)

    # Ignore first and last periods
    t_eval_masked = t_eval[
        (period_use < t_eval) \
        & (t_eval < (periods_n_use - 1) * period_use)
    ]

    # Show only middle
    xlim = (5 * period_use, 6 * period_use)

    # Run Experiments
    fig, ax, rms_use_ref, rms_use_smooth, response_use_meas = run_sub_experiment_side(
        t_meas,
        t_eval_masked,
        amp_use,
        xi_use,
        meas_step_divider,
        bandwidth,
        bandwidth_divider,
        xlim,
        COLOR_USE,
    )
    save_figure(fig, PATH_PLOTS, f"{exp_id}_{exp_sub_id}_use")

    fig, ax, rms_err_ref, rms_err_smooth, response_err_meas =  run_sub_experiment_side(
        t_meas,
        t_eval_masked,
        amp_err,
        xi_err,
        meas_step_divider,
        bandwidth,
        bandwidth_divider,
        xlim,
        COLOR_ERR,
    )
    save_figure(fig, PATH_PLOTS, f"{exp_id}_{exp_sub_id}_err")

    # Metrics
    response_use_ex = smoothing_spline_response(xi_use, bandwidth)
    response_err_ex = smoothing_spline_response(xi_err, bandwidth)

    table.add_row_to_table(
        PATH_TABLES,
        "1_bandwidth",
        [
            meas_step_divider,
            bandwidth_divider,
            rms_use_ref,
            rms_use_smooth,
            response_use_meas,
            response_use_ex,
            abs(response_use_meas - response_use_ex) / response_use_meas,
            rms_err_ref,
            rms_err_smooth,
            response_err_meas,
            response_err_ex,
            abs(response_err_meas - response_err_ex) / response_err_meas,
        ],
    )


def run_sub_experiment_side(
    t_meas,
    t_eval_masked,

    amplitude,
    t_divider,

    meas_step_divider,

    bandwidth,
    bandwidth_divider,

    xlim,
    color,
):
    """Create a figure with subplots for y(t) and y''(t)."""
    # Reference
    y_ref = cos_signal(t_eval_masked, amplitude, t_divider)
    moment_ref = cos_signal_moment(t_eval_masked, amplitude, t_divider)

    # Measured
    y_meas = cos_signal(t_meas, amplitude, t_divider)

    # Smooth
    spline = statistical_smoothing_spline_adaptive(t_meas, y_meas, bandwidth)
    result = spline.solve()
    result_smooth = result.evaluate(t_eval_masked)

    # Calculate response
    rms_ref = rms(y_ref)
    rms_smooth = rms(result_smooth[:,0])
    response = rms_smooth / rms_ref

    # Show
    fig, ax = tmm_element_subplots(
        2,
        BEAM_STATE_NAMES[0::2],
        figsize=(5, 5),
        xlabel="t",
    )

    ref_style = {"linestyle": "--", "color": "k", "alpha": 0.5}
    ax[0].plot(t_eval_masked, y_ref, **ref_style)
    ax[1].plot(t_eval_masked, moment_ref, **ref_style)

    meas_style = {"linestyle": "None", "marker": "+", "color": "k"}
    ax[0].scatter(t_meas, y_meas, **meas_style)

    smooth_style = {"color": color}
    plot_tmm_element(
        ax,
        t_eval_masked,
        result_smooth,
        state_mask=[0, 2],
        plot_style=smooth_style,
    )

    ax[0].set_xlim(xlim)

    custom_legend(
        ax[-1],
        [ref_style, meas_style, smooth_style],
        [
            "Reference",
            rf"Measured ($\Delta_{{avg}} = \frac{{T_{{err}}}}{{{meas_step_divider}}}$)",
            rf"Smoothed ($g = \frac{{T_{{err}}}}{{{bandwidth_divider}}}$)",
        ],
        loc="lower left",
    )

    return fig, ax, rms_ref, rms_smooth, response
if __name__ == '__main__':
    main()
