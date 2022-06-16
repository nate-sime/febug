import numpy as np
from mpi4py import MPI
import dolfinx
import pyvista
import febug


mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, ((0.0, 0.0), (1.0, 1.0)), (3, 3),
    dolfinx.mesh.CellType.quadrilateral)

V = dolfinx.fem.FunctionSpace(mesh, ("CG", 2))
u = dolfinx.fem.Function(V)
u.interpolate(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))

W = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 1))
w = dolfinx.fem.Function(W)
w.interpolate(lambda x: np.stack((-x[1], x[0])))

default_args = dict(transparent_background=True, window_size=(400, 400))


def newplt():
    plotter = pyvista.Plotter(off_screen=True)
    plotter.camera.zoom(1.5)
    return plotter


if mesh.comm.size == 1:
    febug.plot_function(u, plotter=newplt()).screenshot("function.png", **default_args)
    febug.plot_quiver(w, plotter=newplt()).screenshot("quiver.png", **default_args)
    febug.plot_warp(w, plotter=newplt()).screenshot("warp.png", **default_args)


def fn(fname):
    if mesh.comm.size == 1:
        return fname
    splt = fname.split(".")
    return f"{splt[0]}_ghosts_np{mesh.comm.rank}_sz{mesh.comm.size}.{splt[1]}"


febug.plot_mesh(mesh, plotter=newplt()).screenshot(
    fn("mesh.png"), **default_args)
for tdim in range(mesh.topology.dim+1):
    mesh.topology.create_entities(tdim)
    febug.plot_entity_indices(mesh, tdim, plotter=newplt()).screenshot(
        fn(f"enitity_indices_{tdim}.png"), **default_args)
febug.plot_dofmap(V, plotter=newplt()).screenshot(
    fn("dofmap.png"), **default_args)
