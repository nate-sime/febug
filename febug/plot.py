import functools
import typing

import numpy as np
import pyvista

import dolfinx
import dolfinx.plot


@functools.singledispatch
def _to_pyvista_grid(mesh: dolfinx.mesh.Mesh, tdim: int,
                     entities=None):
    return pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(
        mesh, tdim, entities))


@_to_pyvista_grid.register
def _(V: dolfinx.fem.FunctionSpace):
    return pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(V))


@functools.singledispatch
def plot(mesh: dolfinx.cpp.mesh.Mesh, plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()
    grid = _to_pyvista_grid(mesh, mesh.topology.dim)
    plotter.add_mesh(grid, style="wireframe", line_width=2, color="black")

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


@plot.register(dolfinx.cpp.mesh.MeshTags_int32)
@plot.register(dolfinx.cpp.mesh.MeshTags_int64)
def _(meshtags: typing.Union[dolfinx.cpp.mesh.MeshTags_int32,
                             dolfinx.cpp.mesh.MeshTags_int64],
      plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()
    mesh = meshtags.mesh

    edges = _to_pyvista_grid(mesh, meshtags.dim, meshtags.indices)
    edges.cell_arrays["Marker"] = meshtags.values
    edges.set_active_scalars("Marker")
    plotter.add_mesh(edges, show_scalar_bar=True)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


@plot.register
def _(u: dolfinx.fem.function.Function, plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()
    mesh = u.function_space.mesh

    grid = _to_pyvista_grid(u.function_space)

    dof_values = u.x.array
    if np.iscomplexobj(dof_values):
        dof_values = dof_values.real

    grid.point_data[u.name] = dof_values
    grid.set_active_scalars(u.name)

    plotter.add_mesh(grid, scalars=u.name, show_scalar_bar=True)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_warp(u: dolfinx.fem.Function, plotter: pyvista.Plotter=None,
              factor: float=1.0):
    assert len(u.ufl_shape) <= 1

    if plotter is None:
        plotter = pyvista.Plotter()
    mesh = u.function_space.mesh

    grid = _to_pyvista_grid(mesh, mesh.topology.dim)

    dof_values = u.x.array
    if np.iscomplexobj(dof_values):
        dof_values = dof_values.real

    print(dof_values.shape)
    quit()
    # if dof_values.shape
    # vertex_magnitudes = np.linalg.norm(dof_values, axis=dof_values.shape)

    # Add 3rd dimension padding for 2d plots
    if dof_values.shape[1] == 2:
        dof_values = np.hstack(
            (dof_values, np.zeros((dof_values.shape[0], 1))))

    grid.point_data[u.name] = vertex_magnitudes
    grid["vectors"] = dof_values
    grid.set_active_scalars(u.name)

    warped = grid.warp_by_vector("vectors", factor=factor)
    plotter.add_mesh(warped, scalars=u.name, show_scalar_bar=True)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_quiver(u: dolfinx.fem.Function, plotter: pyvista.Plotter=None,
                factor: float=1.0):
    assert len(u.ufl_shape) == 1
    assert u.ufl_shape[0] in (1, 2)

    if plotter is None:
        plotter = pyvista.Plotter()
    mesh = u.function_space.mesh

    grid = _to_pyvista_grid(mesh, mesh.topology.dim)

    vertex_values = u.compute_point_values()
    if np.iscomplexobj(vertex_values):
        vertex_values = vertex_values.real

    vertex_magnitudes = np.linalg.norm(vertex_values, axis=1)

    # Add 3rd dimension padding for 2d plots
    if vertex_values.shape[1] == 2:
        vertex_values = np.hstack(
            (vertex_values, np.zeros((vertex_values.shape[0], 1))))

    grid.point_arrays[u.name] = vertex_magnitudes
    grid["vectors"] = vertex_values
    grid.set_active_scalars(u.name)

    geom = pyvista.Arrow()
    glyphs = grid.glyph(orient="vectors", scale=u.name, factor=factor,
                        geom=geom)
    plotter.add_mesh(glyphs)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_dofmap(V: dolfinx.fem.FunctionSpace, plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()
    mesh = V.mesh
    mesh_grid = _to_pyvista_grid(mesh, mesh.topology.dim)
    dof_grid = _to_pyvista_grid(V)

    x = V.tabulate_dof_coordinates()
    x_polydata = pyvista.PolyData(x)
    x_polydata["labels"] = [f"{i}" for i in np.arange(x.shape[0])]

    plotter.add_mesh(mesh_grid, style="wireframe", line_width=2, color="black")
    plotter.add_point_labels(x_polydata, "labels", point_size=20, font_size=36)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_entity_indices(mesh: dolfinx.mesh.Mesh, tdim: int,
                        plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()
    mesh_grid = _to_pyvista_grid(mesh, mesh.topology.dim)

    entities = np.arange(mesh.topology.index_map(tdim).size_local,
                         dtype=np.int32)
    x = dolfinx.mesh.compute_midpoints(mesh, tdim, entities)
    x_polydata = pyvista.PolyData(x)
    x_polydata["labels"] = [f"{i}" for i in np.arange(x.shape[0])]

    plotter.add_mesh(mesh_grid, style="wireframe", line_width=2, color="black")
    plotter.add_point_labels(x_polydata, "labels", point_size=20, font_size=36)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter