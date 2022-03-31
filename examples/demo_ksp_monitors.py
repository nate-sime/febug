import numpy as np

import ufl
import dolfinx
from ufl import dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc



mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])],
    [32, 32, 32], dolfinx.mesh.CellType.tetrahedron,
    dolfinx.mesh.GhostMode.shared_facet)


x = ufl.SpatialCoordinate(mesh)
f = ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])*ufl.sin(ufl.pi*x[2])

V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = dolfinx.fem.form(inner(grad(u), grad(v)) * dx)
L = dolfinx.fem.form(inner(f, v) * dx)


facets = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim-1, lambda x: np.full_like(x[0], True))
bc = dolfinx.fem.dirichletbc(
    np.array(0.0, dtype=np.double),
    dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1, facets), V)


A = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc])
A.assemble()


b = dolfinx.fem.petsc.assemble_vector(L)
dolfinx.fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.petsc.set_bc(b, [bc])


opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1.0e-10
opts["pc_type"] = "gamg"

opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_pc_type"] = "jacobi"

opts["mg_levels_esteig_ksp_type"] = "cg"
opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

solver = PETSc.KSP().create(mesh.comm)
solver.setFromOptions()

solver.setOperators(A)


uh = dolfinx.fem.Function(V)
#
# import matplotlib.pyplot as plt
# rnorms = []
# plt.xlabel("Iterations")
# plt.ylabel(r"$\Vert \vec{r} \Vert$")
# plt.grid("on")
# def monitor(ctx, it, rnorm):
#     rnorms.append(rnorm)
#     plt.semilogy(range(it+1), rnorms, "-or")
#     plt.pause(0.2)

import febug.monitors
# solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
solver.setMonitor(febug.monitors.monitor_mpl())
solver.solve(b, uh.vector)