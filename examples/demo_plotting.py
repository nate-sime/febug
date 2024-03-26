import numpy as np
from mpi4py import MPI
import dolfinx
import ufl
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

# -- function
subplotter.subplot(0, 1)
subplotter.add_title("CG1")
V = dolfinx.fem.functionspace(mesh, ("CG", 1))
u = dolfinx.fem.Function(V)
u.interpolate(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
febug.plot_function(u, plotter=subplotter)

# -- high order function
subplotter.subplot(0, 2)
subplotter.add_title("CG2")
V = dolfinx.fem.functionspace(mesh, ("CG", 2))
u = dolfinx.fem.Function(V)
u.interpolate(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
febug.plot_function(u, plotter=subplotter)

# -- vector function magnitude
subplotter.subplot(1, 0)
subplotter.add_title("magnitude")
V = dolfinx.fem.functionspace(mesh, ("CG", 1, (2,)))
u = dolfinx.fem.Function(V, dtype=np.float64)
u.interpolate(lambda x: np.stack((-x[1], x[0])))
u_scalar = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("CG", 1)))
u_scalar.interpolate(dolfinx.fem.Expression(ufl.inner(u, u),
                     u_scalar.function_space.element.interpolation_points()))
febug.plot_function(u_scalar, plotter=subplotter)

# -- vector function
subplotter.subplot(1, 1)
subplotter.add_title("quiver")
V = dolfinx.fem.functionspace(mesh, ("CG", 1, (2,)))
u = dolfinx.fem.Function(V)
u.interpolate(lambda x: np.stack((-x[1], x[0])))
febug.plot_quiver(u, plotter=subplotter)

# -- warp
subplotter.subplot(1, 2)
subplotter.add_title("warp")
V = dolfinx.fem.functionspace(mesh, ("CG", 1, (2,)))
u = dolfinx.fem.Function(V)
u.interpolate(lambda x: np.stack((-x[1], x[0])))
febug.plot_warp(u, plotter=subplotter)

# -- mesh tags
def create_meshtags(tdim):
    mesh.topology.create_entities(tdim)
    indices = np.arange(mesh.topology.index_map(tdim).size_local)
    values = indices
    return dolfinx.mesh.meshtags(mesh, tdim, indices, values)

# -- vertices
subplotter.subplot(2, 0)
subplotter.add_title("vertex tags")
febug.plot_meshtags(
    create_meshtags(mesh.topology.dim - 2), mesh, plotter=subplotter)

# -- facets
subplotter.subplot(2, 1)
subplotter.add_title("facet tags")
febug.plot_meshtags(
    create_meshtags(mesh.topology.dim - 1), mesh, plotter=subplotter)

# -- cells
subplotter.subplot(2, 2)
subplotter.add_title("cell tags")
febug.plot_meshtags(
    create_meshtags(mesh.topology.dim), mesh, plotter=subplotter)

# -- show
subplotter.show()
