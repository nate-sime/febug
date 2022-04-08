import sys
from .plot import (plot, plot_meshtags, plot_dofmap, plot_warp, plot_quiver,
                   plot_entity_indices)


def overload_dolfinx():
    sys.modules["dolfinx"] = febug.dolfinx


class FebugError(Exception):

    def __init__(self, message):
        super().__init__(message)


error_on_issue = False


def report_issue(msg):
    import dolfinx
    dolfinx.log.log(dolfinx.log.LogLevel.WARNING, msg)
    if error_on_issue:
        raise FebugError(msg)