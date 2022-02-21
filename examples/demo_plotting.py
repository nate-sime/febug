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

# -- function
subplotter.subplot(0, 0)
subplotter.add_title("CG1")
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
u = dolfinx.fem.Function(V)
u.interpolate(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
febug.plot(u, plotter=subplotter)

# -- high order function
subplotter.subplot(0, 1)
subplotter.add_title("CG2")
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 2))
u = dolfinx.fem.Function(V)
u.interpolate(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
febug.plot(u, plotter=subplotter)

# -- dofmap and ordering
subplotter.subplot(0, 2)
subplotter.add_title("CG2 dofmap")
febug.plot_dofmap(V, plotter=subplotter)

# -- mesh
subplotter.subplot(1, 0)
subplotter.add_title("mesh")
febug.plot(mesh, plotter=subplotter)

# -- mesh tags
def create_meshtags(tdim):
    mesh.topology.create_entities(tdim)
    indices = np.arange(mesh.topology.index_map(tdim).size_local)
    values = indices
    return dolfinx.mesh.MeshTags(mesh, tdim, indices, values)

# -- facets
subplotter.subplot(1, 1)
subplotter.add_title("facet tags")
febug.plot(create_meshtags(mesh.topology.dim - 1), plotter=subplotter)

# -- cells
subplotter.subplot(1, 2)
subplotter.add_title("cell tags")
febug.plot(create_meshtags(mesh.topology.dim), plotter=subplotter)

# -- Entity indices
entity_types = ["vertex", "facet", "cell"]
for tdim in range(mesh.topology.dim + 1):
    subplotter.subplot(2, tdim)
    subplotter.add_title(f"{entity_types[tdim]} indices")
    febug.plot_entity_indices(mesh, tdim, plotter=subplotter)

# -- show
subplotter.show()
