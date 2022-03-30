import febug
febug.overload_dolfinx()
febug.error_on_issue = True

import dolfinx
from mpi4py import MPI
import numpy as np


def test_unordered_meshtags():
    febug.overload_dolfinx()
    febug.error_on_issue = True
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)

    indices = np.array([1, 2, 3, 4, 2], dtype=np.int32)
    values = np.array([1.0, 2.0, 3.0, 4.0, 2.0], dtype=np.double)

    try:
        dolfinx.mesh.meshtags(mesh, mesh.topology.dim, indices, values)
    except Exception as e:
        assert isinstance(e, febug.FebugError)
