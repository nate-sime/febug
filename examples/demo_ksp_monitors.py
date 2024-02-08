import numpy as np

import ufl
import dolfinx
import dolfinx.fem.petsc
from ufl import dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc


mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
    [16, 16, 16], cell_type=dolfinx.mesh.CellType.tetrahedron,
    ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

import febug.meshquality
febug.meshquality.hist_unicode(
    febug.meshquality.pyvista_entity_quality(
        mesh, mesh.topology.dim, quality_measure="min_angle"),
    np.linspace(0, 180, 10), title="Minimum dihedral angle")

x = ufl.SpatialCoordinate(mesh)
f = ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])*ufl.sin(ufl.pi*x[2])

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
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
opts["ksp_rtol"] = 1.0e-12
opts["pc_type"] = "jacobi"

solver = PETSc.KSP().create(mesh.comm)
solver.setFromOptions()

solver.setOperators(A)
uh = dolfinx.fem.Function(V)

import febug.monitors
solver.setMonitor(febug.monitors.monitor_mpl())
solver.setMonitor(febug.monitors.monitor_unicode_graph())
# solver.setMonitor(febug.monitors.monitor_text_petsc())
solver.solve(b, uh.vector)