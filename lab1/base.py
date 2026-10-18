import random as rnd

import numpy as np


def uneven_points(min_step: float, max_step: float, stop: float, seed: int = 123):
    """Generate unevenly spaced points."""
    step_range = max_step - min_step
    rnd.seed(seed)
    points = [0.0]
    while points[-1] < stop:
        step = min_step + step_range * rnd.random()
        points.append(points[-1] + step)
    return np.array(points, dtype=np.float64)


def cos_signal(t, amplitude, t_divider):
    """Generate an y = amplitude * cos(t / t_divider) signal."""
    return amplitude * np.cos(t / t_divider)


def cos_signal_moment(t, amplitude, t_divider):
    """Generate a second derivative of an y = amplitude * cos(t / t_divider) signal."""
    return -amplitude / t_divider**2 * np.cos(t / t_divider)


def cos_signal_period(t_divider):
    """Calculate a period for a given t_divider."""
    return 2.0 * np.pi * t_divider


def rms(values):
    """Calculate a root-mean-square for values."""
    return np.sqrt(np.mean(np.square(values)))
