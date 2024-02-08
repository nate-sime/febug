import sys
from .plot import (plot_mesh, plot_function, plot_meshtags, plot_dofmap,
                   plot_warp, plot_quiver,
                   plot_streamlines_evenly_spaced_2D,
                   plot_streamlines_from_source,
                   plot_entity_indices,
                   plot_mesh_quality, plot_function_dofs,
                   plot_point_cloud)


def overload_dolfinx():
    import febug.dolfinx
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
