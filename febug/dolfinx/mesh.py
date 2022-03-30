import numpy as np
import dolfinx.log

import febug

from dolfinx.mesh import *


def meshtags(mesh: dolfinx.mesh.Mesh, dim: int, indices: np.ndarray,
             values: np.ndarray):
    mts = dolfinx.mesh.meshtags(mesh, dim, indices, values)
    check_mesh_tags(mts)
    return mts


def check_mesh_tags(meshtags):
    if not np.all(meshtags.indices[:-1] < meshtags.indices[1:]):
        febug.report_issue(f"Mesh tags '{meshtags.name}'"
                           f" indices are not sorted")
    if np.unique(meshtags.indices).shape[0] != meshtags.indices.shape[0]:
        febug.report_issue(f"Mesh tags '{meshtags.name}'"
                           f" indices are not unique")
    if meshtags.indices.shape[0] != meshtags.values.shape[0]:
        febug.report_issue(f"Mesh tags '{meshtags.name}' has incorrectly "
                           f"sized data: "
                           f" indices size {meshtags.indices.shape}, "
                           f" values size {meshtags.values.shape}")
