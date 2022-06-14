import febug
febug.overload_dolfinx()
febug.error_on_issue = False

from petsc4py import PETSc
from mpi4py import MPI
import ufl
import dolfinx.nls.petsc
import numpy as np


mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
v = ufl.TestFunction(V)

u = dolfinx.fem.Function(V)
u.name = "u"
u.interpolate(lambda x: x[0]*x[1])

# -- Form contains singularities
F = ufl.inner((1 / u) * ufl.grad(u), ufl.grad(v)) * ufl.dx

# -- Mesh tag data is invalid!
indices = np.array([1, 2, 3, 4, 2], dtype=np.int32)
values = np.array([1.0, 2.0, 3.0, 4.0, 2.0], dtype=np.double)
mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, indices, values)

facets = dolfinx.mesh.locate_entities_boundary(
    mesh, dim=1, marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0))
dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
bcs = [dolfinx.fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs, V=V)]

problem = dolfinx.fem.petsc.NonlinearProblem(F, u, bcs)
problem = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)

# -- We should reuse data structures wherever possible
J = ufl.derivative(F, u)
for j in range(11):
    linear_solver = dolfinx.fem.petsc.LinearProblem(J, F)
    linear_solver.solve()
