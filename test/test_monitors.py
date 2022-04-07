import numpy as np
import febug.monitors


def test_monitor_unicode_graph_monotonic_decrease():
    rnorms = np.logspace(0, -12, 10)
    monitor = febug.monitors.monitor_unicode_graph()
    for i, rnorm in enumerate(rnorms):
        monitor(None, i, rnorm)


def test_monitor_unicode_graph_divergence():
    rnorms = [1e-2, 1e9, 1e2, 1e3, 1e-2]
    monitor = febug.monitors.monitor_unicode_graph()
    for i, rnorm in enumerate(rnorms):
        monitor(None, i, rnorm)

def test_monitor_unicode_graph_exponent_intervals():
    rnorms = [1e2, 5e1, 1e-2, 5e-3]
    monitor = febug.monitors.monitor_unicode_graph()
    for i, rnorm in enumerate(rnorms):
        monitor(None, i, rnorm)