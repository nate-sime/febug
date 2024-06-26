import functools
import typing

import numpy as np
import numpy.typing as npt
import pyvista

import dolfinx
import dolfinx.plot
import ufl.core.expr

entity_label_args = dict(point_size=15, font_size=12, bold=False,
                         shape_color="white", text_color="black")

@functools.singledispatch
def _to_pyvista_grid(mesh: dolfinx.mesh.Mesh, tdim: int,
                     entities=None):
    mesh.topology.create_connectivity(0, tdim)
    mesh.topology.create_connectivity(tdim, mesh.topology.dim)
    return pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(
        mesh, tdim, entities))


@_to_pyvista_grid.register
def _(V: dolfinx.fem.FunctionSpace):
    return pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V))


@_to_pyvista_grid.register
def _(u: dolfinx.fem.Function):
    V = u.function_space
    mesh = V.mesh

    bs = V.dofmap.index_map_bs
    dof_values = u.x.array.reshape(
        V.tabulate_dof_coordinates().shape[0],
        V.dofmap.index_map_bs)

    if bs == 2:
        dof_values = np.hstack(
            (dof_values, np.zeros((dof_values.shape[0], 3-bs))))

    if np.iscomplexobj(dof_values):
        dof_values = dof_values.real

    if V.ufl_element().degree == 0:
        grid = _to_pyvista_grid(mesh, mesh.topology.dim)
        num_dofs_local = V.dofmap.index_map.size_local
        grid.cell_data[u.name] = dof_values[:num_dofs_local]
    else:
        grid = _to_pyvista_grid(V)
        grid.point_data[u.name] = dof_values

    grid.set_active_scalars(u.name)
    return grid


def plot_mesh(mesh: dolfinx.mesh.Mesh, tdim: int=None,
              show_owners: bool=False, plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()

    if tdim is None:
        tdim = mesh.topology.dim

    size_local = mesh.topology.index_map(tdim).size_local
    entities = np.arange(size_local, dtype=np.int32)
    ghost_entities = np.arange(
        size_local, size_local + mesh.topology.index_map(tdim).num_ghosts,
        dtype=np.int32)

    # Plot ghosts first so the z-heightmap shows local entities primarily
    for grp, color in ((ghost_entities, "pink"), (entities, "black")):
        if len(grp) == 0:
            continue
        if tdim > 0:
            grid = _to_pyvista_grid(mesh, tdim, entities=grp)
            plotter.add_mesh(grid, style="wireframe", line_width=2, color=color)
        else:
            e2g = dolfinx.mesh.entities_to_geometry(mesh, 0, grp)
            assert e2g.shape[1] == 1
            e2g = e2g.ravel()
            point_cloud = pyvista.PolyData(mesh.geometry.x[e2g])
            plotter.add_mesh(point_cloud, point_size=8, color=color)

    if len(ghost_entities) > 0 and show_owners:
        ghost_owners = mesh.topology.index_map(tdim).ghost_owners()
        x_ghost = dolfinx.mesh.compute_midpoints(mesh, tdim, ghost_entities)

        # todo: plot more meaningful data than the neighbourhood comm ghosts
        ghost_polydata = pyvista.PolyData(x_ghost)
        ghost_polydata["labels"] = ghost_owners
        plotter.add_point_labels(ghost_polydata, "labels", point_size=8,
                                 font_size=24)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def create_plottable_ufl_expression(
        expr_ufl: ufl.core.expr.Expr,
        u: dolfinx.fem.Function | dolfinx.fem.FunctionSpace):
    if isinstance(u, dolfinx.fem.FunctionSpace):
        u = dolfinx.fem.Function(u)
    expr = dolfinx.fem.Expression(
        expr_ufl, u.function_space.element.interpolation_points())
    u.interpolate(expr)
    return u


def plot_function(u: dolfinx.fem.function.Function,
                  plotter: pyvista.Plotter=None, **pv_args):
    if plotter is None:
        plotter = pyvista.Plotter()

    mesh = u.function_space.mesh

    grid = _to_pyvista_grid(u)

    if len(grid[u.name]) == 0:
        # No data on process
        return plotter

    pv_args.setdefault("show_scalar_bar", True)
    plotter.add_mesh(grid, scalars=u.name, **pv_args)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_ufl_expression(expr_ufl: ufl.core.expr.Expr,
                        V: dolfinx.fem.FunctionSpace,
                        plotter: pyvista.Plotter=None):
    fh = create_plottable_ufl_expression(expr_ufl, V)
    return plot_function(fh, plotter)


def plot_meshtags(meshtags: dolfinx.mesh.MeshTags,
                  mesh: dolfinx.mesh.Mesh,
                  plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()

    if np.issubdtype(meshtags.values.dtype, np.integer):
        unique_vals = np.unique(meshtags.values)
        annotations = dict(zip(unique_vals, map(str, unique_vals)))
    else:
        annotations = None

    if meshtags.dim > 0:
        entities = _to_pyvista_grid(mesh, meshtags.dim, meshtags.indices)
        entities.cell_data[meshtags.name] = meshtags.values
        entities.set_active_scalars(meshtags.name)
    else:
        e2g = dolfinx.mesh.entities_to_geometry(mesh, 0, meshtags.indices)
        # e2g dim 0 always shape (n, 1) so ravel
        assert e2g.shape[1] == 1
        e2g = e2g.ravel()
        x = mesh.geometry.x[e2g]
        entities = pyvista.PolyData(x)
        entities[meshtags.name] = meshtags.values

    if len(entities[meshtags.name]) == 0:
        return plotter

    plotter.add_mesh(entities, show_scalar_bar=True, annotations=annotations)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_meshtags_values(meshtags: dolfinx.mesh.MeshTags,
                         mesh: dolfinx.mesh.Mesh,
                         fmt: str | None=None,
                         plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()

    if fmt is None:
        fmt = ".2f"
        if np.issubdtype(meshtags.values.dtype, np.integer):
            fmt = "d"

    tdim = meshtags.dim
    size_local = mesh.topology.index_map(tdim).size_local

    entities = meshtags.indices
    labels = meshtags.values

    entities_local = entities[entities < size_local]
    labels_local = labels[entities < size_local]
    entities_ghost = entities[entities >= size_local]
    labels_ghost = labels[entities >= size_local]

    def plot_midpoint_label(entities, labels, color):
        x = dolfinx.mesh.compute_midpoints(mesh, tdim, entities)
        x_polydata = pyvista.PolyData(x)
        x_polydata["labels"] = [f"{i:{fmt}}" for i in labels]
        plotter.add_point_labels(x_polydata, "labels", **entity_label_args,
                                 point_color=color)

    if entities_local.size > 0:
        plot_midpoint_label(entities_local, labels_local, "grey")

    if entities_ghost.size > 0:
        plot_midpoint_label(entities_ghost, labels_ghost, "pink")

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

    grid = _to_pyvista_grid(u)
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

    grid = _to_pyvista_grid(u)

    geom = pyvista.Arrow()
    glyphs = grid.glyph(orient=u.name, scale=u.name, factor=factor,
                        geom=geom)
    plotter.add_mesh(glyphs)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_streamlines_evenly_spaced_2D(
        u: dolfinx.fem.Function,
        start_position: typing.Sequence[float],
        plotter: pyvista.Plotter=None,
        **pv_args):
    assert len(u.ufl_shape) == 1
    assert u.ufl_shape[0] in (1, 2, 3)

    if plotter is None:
        plotter = pyvista.Plotter()
    mesh = u.function_space.mesh

    grid = _to_pyvista_grid(u)

    streamlines = grid.streamlines_evenly_spaced_2D(
        vectors=u.name, start_position=start_position, **pv_args)
    plotter.add_mesh(streamlines)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_streamlines_from_source(
        u: dolfinx.fem.Function,
        source: pyvista.DataSet,
        plotter: pyvista.Plotter=None,
        **pv_args):
    assert len(u.ufl_shape) == 1
    assert u.ufl_shape[0] in (1, 2, 3)

    if plotter is None:
        plotter = pyvista.Plotter()
    mesh = u.function_space.mesh

    grid = _to_pyvista_grid(u)

    streamlines = grid.streamlines_from_source(
        source, vectors=u.name, **pv_args)
    plotter.add_mesh(streamlines)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_dofmap(V: dolfinx.fem.FunctionSpace, plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()
    mesh = V.mesh

    x = V.tabulate_dof_coordinates()
    if x.shape[0] == 0:
        return plotter

    size_local = V.dofmap.index_map.size_local
    num_ghosts = V.dofmap.index_map.num_ghosts

    if size_local > 0:
        x_local_polydata = pyvista.PolyData(x[:size_local])
        x_local_polydata["labels"] = [f"{i}" for i in np.arange(size_local)]
        plotter.add_point_labels(
            x_local_polydata, "labels", **entity_label_args,
            point_color="black")

    if num_ghosts > 0:
        x_ghost_polydata = pyvista.PolyData(x[size_local:size_local+num_ghosts])
        x_ghost_polydata["labels"] = [
            f"{i}" for i in np.arange(size_local, size_local+num_ghosts)]
        plotter.add_point_labels(
            x_ghost_polydata, "labels", **entity_label_args,
            point_color="pink")

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_function_dofs(u: dolfinx.fem.function.Function,
                       fmt: str=".2f",
                       plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()

    V = u.function_space
    mesh = V.mesh

    x = V.tabulate_dof_coordinates()
    if x.shape[0] == 0:
        return plotter

    size_local = V.dofmap.index_map.size_local
    num_ghosts = V.dofmap.index_map.num_ghosts
    bs = V.dofmap.bs
    u_arr = u.x.array.reshape((-1, bs))
    str_formatter = lambda x: "\n".join((f"{u_:{fmt}}" for u_ in x))

    if size_local > 0:
        x_local_polydata = pyvista.PolyData(x[:size_local])
        x_local_polydata["labels"] = list(
            map(str_formatter, u_arr[:size_local]))
        plotter.add_point_labels(
            x_local_polydata, "labels", **entity_label_args,
            point_color="black")

    if num_ghosts > 0:
        x_ghost_polydata = pyvista.PolyData(x[size_local:size_local+num_ghosts])
        x_ghost_polydata["labels"] = list(
            map(str_formatter, u_arr[size_local:size_local+num_ghosts]))
        plotter.add_point_labels(
            x_ghost_polydata, "labels", **entity_label_args,
            point_color="pink")

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def _plot_entity_indices_impl(mesh: dolfinx.mesh.Mesh, tdim: int,
                              local_to_label_mapping: typing.Callable[
                             [npt.NDArray[np.int32]], npt.NDArray[np.int32]],
                              plotter: pyvista.Plotter):

    plot_mesh(mesh, tdim=tdim, plotter=plotter, show_owners=False)

    size_local = mesh.topology.index_map(tdim).size_local
    num_ghosts = mesh.topology.index_map(tdim).num_ghosts
    entities = np.arange(size_local, dtype=np.int32)
    ghosts = np.arange(size_local, size_local + num_ghosts, dtype=np.int32)

    if size_local > 0:
        x = dolfinx.mesh.compute_midpoints(mesh, tdim, entities)
        x_polydata = pyvista.PolyData(x)
        labels = local_to_label_mapping(np.arange(size_local))
        x_polydata["labels"] = [f"{i}" for i in labels]
        plotter.add_point_labels(x_polydata, "labels", **entity_label_args,
                                 point_color="grey")

    if num_ghosts > 0:
        x_ghost = dolfinx.mesh.compute_midpoints(mesh, tdim, ghosts)
        x_ghost_polydata = pyvista.PolyData(x_ghost)
        ghost_labels = local_to_label_mapping(
            np.arange(size_local, size_local + num_ghosts))
        x_ghost_polydata["labels"] = [f"{i}" for i in ghost_labels]
        plotter.add_point_labels(
            x_ghost_polydata, "labels", **entity_label_args,
            point_color="pink")

    return plotter


def plot_entity_indices(mesh: dolfinx.mesh.Mesh, tdim: int,
                        plotter: pyvista.Plotter=None,
                        local_to_label_mapping: typing.Callable[
                            [npt.NDArray[np.int32]], npt.NDArray[np.int32]
                                                ] | None = None):
    if plotter is None:
        plotter = pyvista.Plotter()

    if local_to_label_mapping is None:
        def local_to_label_mapping(local_idxs: npt.NDArray[np.int32]):
            return local_idxs

    plot_mesh(mesh, tdim=tdim, plotter=plotter, show_owners=False)
    _plot_entity_indices_impl(mesh, tdim, local_to_label_mapping, plotter)

    if mesh.geometry.dim == 2:
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_entity_indices_global(mesh: dolfinx.mesh.Mesh, tdim: int,
                               plotter: pyvista.Plotter=None):
    im_tdim = mesh.topology.index_map(tdim)
    return plot_entity_indices(
        mesh, tdim, plotter=plotter,
        local_to_label_mapping=lambda local_idx: im_tdim.local_to_global(
            local_idx))


def plot_entity_indices_original(mesh: dolfinx.mesh.Mesh, tdim: int,
                                 plotter: pyvista.Plotter=None):
    if not tdim in (0, mesh.topology.dim):
        msg = ("Original indices only defined for topological dimensions 0 "
               f"and {mesh.topology.dim}")
        raise AttributeError(msg)

    def get_original_indices(
            local_idx: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        assert tdim in (0, mesh.topology.dim)
        return (
            mesh.geometry.input_global_indices[local_idx]
            if tdim == 0 else mesh.topology.original_cell_index[local_idx])

    return plot_entity_indices(
        mesh, tdim, plotter=plotter,
        local_to_label_mapping=get_original_indices)


def plot_geometry_indices(mesh: dolfinx.mesh.Mesh,
                          plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()

    plot_mesh(mesh, tdim=0, plotter=plotter, show_owners=False)

    size_local = mesh.geometry.index_map().size_local
    num_ghosts = mesh.geometry.index_map().num_ghosts
    entities = np.arange(size_local, dtype=np.int32)
    ghosts = np.arange(size_local, size_local + num_ghosts, dtype=np.int32)

    if size_local > 0:
        x = mesh.geometry.x[entities]
        x_polydata = pyvista.PolyData(x)
        labels = np.arange(size_local)
        x_polydata["labels"] = [f"{i}" for i in labels]
        plotter.add_point_labels(x_polydata, "labels", **entity_label_args,
                                 point_color="grey")

    if num_ghosts > 0:
        x_ghost = mesh.geometry.x[ghosts]
        x_ghost_polydata = pyvista.PolyData(x_ghost)
        ghost_labels = np.arange(size_local, size_local + num_ghosts)
        x_ghost_polydata["labels"] = [f"{i}" for i in ghost_labels]
        plotter.add_point_labels(
            x_ghost_polydata, "labels", **entity_label_args,
            point_color="pink")

    return plotter


def plot_point_cloud(xp: np.ndarray[float],
                     plotter: pyvista.Plotter=None):
    if plotter is None:
        plotter = pyvista.Plotter()

    pv_point_cloud = pyvista.PolyData(xp)
    plotter.add_mesh(pv_point_cloud)

    if np.all(np.isclose(xp[:,2], 0.0)):
        plotter.enable_parallel_projection()
        plotter.view_xy()

    return plotter


def plot_mesh_quality(mesh: dolfinx.mesh.Mesh, tdim: int,
                      plotter: pyvista.Plotter=None,
                      quality_measure: str="scaled_jacobian",
                      entities=None,
                      progress_bar: bool=False):
    if plotter is None:
        plotter = pyvista.Plotter()
    if mesh.topology.index_map(tdim) is None:
        mesh.topology.create_entities(tdim)
    mesh_grid = _to_pyvista_grid(mesh, tdim, entities)
    if mesh_grid.n_points < 1:
        return plotter

    qual = mesh_grid.compute_cell_quality(
        quality_measure=quality_measure, progress_bar=progress_bar)

    qual.set_active_scalars("CellQuality")
    plotter.add_mesh(qual, show_scalar_bar=True)

    return plotter
