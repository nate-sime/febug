import pytest

import dolfinx
import ufl
from mpi4py import MPI


@pytest.mark.parametrize("u_lmbda_has_zero",
                         ((lambda x: x[0], True),
                          (lambda x: x[0]+1, False)))
def test_singularity_finder_residual(u_lmbda_has_zero):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    v = ufl.TestFunction(V)

    u_lmbda, has_zero = u_lmbda_has_zero

    u = dolfinx.fem.Function(V)
    u.interpolate(u_lmbda)

    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    def do_test(form):
        issues = febug.dolfinx.nls.search_for_potential_singularity(form)
        assert issues[0][0].name == u.name
        assert len(issues[0][1]) > 0 if has_zero else len(issues[0][1]) == 0

    do_test(F)
    do_test(dolfinx.fem.form(F))
