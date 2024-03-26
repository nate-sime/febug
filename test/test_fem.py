import numpy as np
import pytest

import febug.dolfinx.fem
import dolfinx
import ufl
from mpi4py import MPI


def test_petsc_object_count_limit():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("CG", 1))
    v = ufl.TestFunction(V)

    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.ones_like(x[0]))
    F = ufl.inner(u, v) * ufl.dx
    J = ufl.derivative(F, u)

    febug.error_on_issue = True

    with pytest.raises(febug.FebugError):
        for j in range(febug.dolfinx.fem.petsc._MatrixCount._limit):
            febug.dolfinx.fem.petsc.create_matrix(dolfinx.fem.form(J))

    with pytest.raises(febug.FebugError):
        for j in range(febug.dolfinx.fem.petsc.LinearProblem._limit):
            febug.dolfinx.fem.petsc.LinearProblem(J, F)

    with pytest.raises(febug.FebugError):
        for j in range(febug.dolfinx.fem.petsc.NonlinearProblem._limit):
            febug.dolfinx.fem.petsc.NonlinearProblem(F, u, J=J)
