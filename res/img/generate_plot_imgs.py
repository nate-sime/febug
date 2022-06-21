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


# Serial only plots
if mesh.comm.size == 1:
    febug.plot_function(u, plotter=newplt()).screenshot(
        "function.png", **default_args)
    febug.plot_quiver(w, plotter=newplt()).screenshot(
        "quiver.png", **default_args)
    febug.plot_warp(w, plotter=newplt()).screenshot(
        "warp.png", **default_args)


# Generate meaningful filename
def fn(fname):
    if mesh.comm.size == 1:
        return fname
    splt = fname.split(".")
    return f"{splt[0]}_p{mesh.comm.rank}_s{mesh.comm.size}.{splt[1]}"


# -- Serial and parallel plots
febug.plot_mesh(mesh, plotter=newplt()).screenshot(
    fn("mesh.png"), **default_args)

for tdim in range(mesh.topology.dim+1):
    mesh.topology.create_entities(tdim)
    febug.plot_entity_indices(mesh, tdim, plotter=newplt()).screenshot(
        fn(f"entity_indices_{tdim}.png"), **default_args)

mesh_tri = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, ((0.0, 0.0), (1.0, 1.0)), (3, 3),
    dolfinx.mesh.CellType.triangle)
mesh_interval = dolfinx.mesh.create_unit_interval(
    MPI.COMM_WORLD, 3)
configs = {mesh: [("CG", 1), ("CG", 2), ("DG", 0), ("DPC", 1)],
           mesh_tri: [("Bubble", 3), ("CR", 1)],
           mesh_interval: []}
for mesh_, es in configs.items():
    for e in es:
        V = dolfinx.fem.FunctionSpace(mesh_, e)

        febug.plot_dofmap(V, plotter=newplt()).screenshot(
            fn(f"dofmap_{e[0]}{e[1]}.png"), **default_args)

        u = dolfinx.fem.Function(V)
        if u.ufl_shape:
            u.interpolate(lambda x: np.stack([x[j] for j in range(u.ufl_shape[0])]))
        else:
            u.interpolate(lambda x: x[0]*x[1])
        plotter = newplt()
        try:
            febug.plot_function(u, plotter=plotter)
        except RuntimeError:
            pass
        febug.plot_mesh(mesh_, plotter=plotter)
        febug.plot_function_dofs(u, plotter=plotter).screenshot(
            fn(f"function_dofmap_{e[0]}{e[1]}.png"), **default_args)



mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, ((0.0, 0.0), (1.0, 1.0)), (3, 3),
    dolfinx.mesh.CellType.quadrilateral)