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


def plot_mesh(mesh: dolfinx.cpp.mesh.Mesh, plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()
    grid = _to_pyvista_grid(mesh, mesh.topology.dim)
    plotter.add_mesh(grid, style="wireframe", line_width=2, color="black")

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_function(u: dolfinx.fem.function.Function,
                  plotter: pyvista.Plotter=None):
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


def plot_meshtags(meshtags: typing.Union[
    dolfinx.cpp.mesh.MeshTags_int8, dolfinx.cpp.mesh.MeshTags_int32,
    dolfinx.cpp.mesh.MeshTags_int64, dolfinx.mesh.MeshTagsMetaClass],
                  plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()
    mesh = meshtags.mesh

    if meshtags.dim > 0:
        edges = _to_pyvista_grid(mesh, meshtags.dim, meshtags.indices)
        edges.cell_data[meshtags.name] = meshtags.values
        edges.set_active_scalars(meshtags.name)
        plotter.add_mesh(edges, show_scalar_bar=True)
    else:
        x = mesh.geometry.x[meshtags.indices]
        point_cloud = pyvista.PolyData(x)
        point_cloud[meshtags.name] = meshtags.values

        plotter.add_mesh(point_cloud)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_warp(u: dolfinx.fem.Function, plotter: pyvista.Plotter=None,
              factor: float=1.0):
    assert len(u.ufl_shape) == 1
    assert u.ufl_shape[0] in (2, 3)

    if plotter is None:
        plotter = pyvista.Plotter()
    mesh = u.function_space.mesh

    grid = _to_pyvista_grid(u.function_space)

    bs = u.function_space.dofmap.index_map_bs
    plot_values = u.x.array.reshape(
        u.function_space.tabulate_dof_coordinates().shape[0],
        u.function_space.dofmap.index_map_bs)
    if bs < 3:
        plot_values = np.hstack(
            (plot_values, np.zeros((plot_values.shape[0], 3-bs))))
    grid.point_data[u.name] = plot_values

    warped = grid.warp_by_vector(vectors=u.name)
    plotter.add_mesh(warped)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_quiver(u: dolfinx.fem.Function, plotter: pyvista.Plotter=None,
                factor: float=1.0):
    assert len(u.ufl_shape) == 1
    assert u.ufl_shape[0] in (1, 2, 3)

    if plotter is None:
        plotter = pyvista.Plotter()
    mesh = u.function_space.mesh

    grid = _to_pyvista_grid(u.function_space)

    bs = u.function_space.dofmap.index_map_bs
    plot_values = u.x.array.reshape(
        u.function_space.tabulate_dof_coordinates().shape[0],
        u.function_space.dofmap.index_map_bs)
    if bs < 3:
        plot_values = np.hstack(
            (plot_values, np.zeros((plot_values.shape[0], 3-bs))))
    grid.point_data[u.name] = plot_values

    geom = pyvista.Arrow()
    glyphs = grid.glyph(orient=u.name, scale=u.name, factor=factor,
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


def plot_mesh_quality(mesh: dolfinx.mesh.Mesh, tdim: int,
                      plotter: pyvista.Plotter=None,
                      quality_measure: str = "scaled_jacobian",
                      entities=None,
                      progress_bar: bool=False):
    if plotter is None:
        plotter = pyvista.Plotter()
    if mesh.topology.index_map(tdim) is None:
        mesh.topology.create_entities(tdim)
    mesh_grid = _to_pyvista_grid(mesh, tdim, entities)

    qual = mesh_grid.compute_cell_quality(
        quality_measure=quality_measure, progress_bar=progress_bar)

    qual.set_active_scalars("CellQuality")
    plotter.add_mesh(qual, show_scalar_bar=True)

    return plotter