import numpy as np
from tmm_kit.beam_splines.boundary_conditions import elastic_ends
from tmm_kit.beam_splines.elements import BeamElement
from tmm_kit.beam_splines.nodes import BeamElasticSupport
from tmm_kit.beam_splines.solvers import StatisticalBeamSplineSolver
from tmm_kit.beam_splines.utils import adaptive_stiffness


def statistical_smoothing_spline_adaptive(
    ts,
    ws,
    bandwidth,
):
    """Get statistical smoothing spline with adaptive stiffness."""
    element_lengths = np.diff(ts)

    return StatisticalBeamSplineSolver(
        t=ts,
        element_class=BeamElement,
        element_data={},
        node_class=BeamElasticSupport,
        node_data={
            "displacement": ws[1:-1],
            "stiffness": adaptive_stiffness(
                (element_lengths[:-1] + element_lengths[1:]) / 2.0,
                bandwidth,
            ),
        },
        boundary_conditions=elastic_ends(
            ws[0],
            ws[-1],
            adaptive_stiffness(element_lengths[0], bandwidth),
            adaptive_stiffness(element_lengths[-1], bandwidth),
        ),
    )
