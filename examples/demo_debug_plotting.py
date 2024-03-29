import numpy as np
from mpi4py import MPI
import dolfinx
import pyvista
import febug

# Create mesh and define function space
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, ((0.0, 0.0), (1.0, 1.0)), (2, 2),
    dolfinx.mesh.CellType.quadrilateral)

subplotter = pyvista.Plotter(shape=(3, 3))

# -- mesh
subplotter.subplot(0, 0)
subplotter.add_title("mesh")
febug.plot_mesh(mesh, plotter=subplotter)

# -- CG1 dofmap and ordering
V = dolfinx.fem.functionspace(mesh, ("CG", 1))
subplotter.subplot(0, 1)
subplotter.add_title("CG1 dofmap")
febug.plot_dofmap(V, plotter=subplotter)

# -- CG2 dofmap and ordering
V = dolfinx.fem.functionspace(mesh, ("CG", 2))
subplotter.subplot(0, 2)
subplotter.add_title("CG2 dofmap")
febug.plot_dofmap(V, plotter=subplotter)

# -- Entity indices
entity_types = ["vertex", "facet", "cell"]
for tdim in range(mesh.topology.dim + 1):
    subplotter.subplot(1, tdim)
    subplotter.add_title(f"{entity_types[tdim]} indices")
    febug.plot_entity_indices(mesh, tdim, plotter=subplotter)

# -- Function DoFs
eles = [("CG", 1), ("CG", 2), ("DG", 0)]
for e_i, e in enumerate(eles):
    subplotter.subplot(2, e_i)
    subplotter.add_title(f"{e[0]}{e[1]} space DoFs")
    V = dolfinx.fem.functionspace(mesh, e)
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0]*x[1])
    febug.plot_function(u, plotter=subplotter)
    febug.plot_mesh(mesh, plotter=subplotter)
    febug.plot_function_dofs(u, fmt=".2e", plotter=subplotter)

# -- show
subplotter.show()
