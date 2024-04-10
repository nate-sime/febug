import numpy as np
import pytest
import pyvista
import ufl
from mpi4py import MPI
import dolfinx
import dolfinx.mesh
import dolfinx.fem
import febug
import febug.dolfinx


skip_parallel = pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1, reason="Test disabled in parallel")

meshes1D = [dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 3)]
meshes2D = [dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)]
meshes3D = [dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)]
all_meshes = meshes1D + meshes2D + meshes3D


@pytest.mark.parametrize("mesh", all_meshes)
@pytest.mark.parametrize("p", [1, 2])
def test_plot_function_cg(p, mesh):
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("CG", p)))
    febug.plot_function(u)


@pytest.mark.parametrize("mesh", all_meshes)
@pytest.mark.parametrize("p", [0, 1, 2])
def test_plot_function_dg(p, mesh):
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", p)))
    febug.plot_function(u)


@pytest.mark.parametrize("mesh", all_meshes)
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("element", ["CG", "DG"])
def test_plot_ufl_expression(p, mesh, element):
    V = dolfinx.fem.functionspace(mesh, (element, p))
    febug.plot_ufl_expression(ufl.SpatialCoordinate(mesh)[0], V)


@pytest.mark.parametrize(
    "mesh",
    [pytest.param(msh, marks=pytest.mark.xfail(
        msh.topology.dim == 1, reason="No warping 1D meshes"))
     for msh in all_meshes])
@pytest.mark.parametrize("p", [1, 2])
def test_plot_warp(p, mesh):
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("CG", p, (mesh.geometry.dim,))))
    febug.plot_warp(u)


@pytest.mark.parametrize(
    "mesh",
    [pytest.param(msh, marks=pytest.mark.xfail(
        msh.topology.dim == 1, reason="No quivering 1D meshes"))
     for msh in all_meshes])
@pytest.mark.parametrize("p", [1, 2])
def test_plot_quiver(p, mesh):
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("CG", p, (mesh.geometry.dim,))))
    febug.plot_quiver(u)


@pytest.mark.parametrize(
    "mesh",
    [pytest.param(msh, marks=pytest.mark.xfail(
        msh.topology.dim != 2, reason="Streamline function for 2D only"))
     for msh in all_meshes])
@pytest.mark.parametrize("p", [1, 2])
def test_plot_streamlines_evenly_spaced_2D(p, mesh):
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("CG", p, (mesh.geometry.dim,))))
    febug.plot_streamlines_evenly_spaced_2D(u, start_position=(0.5, 0.5, 0.0))


@pytest.mark.parametrize(
    "mesh",
    [pytest.param(msh, marks=pytest.mark.xfail(
        msh.topology.dim == 1, reason="Streamlines not valid for 1D"))
     for msh in all_meshes])
@pytest.mark.parametrize("p", [1, 2])
def test_plot_streamlines_from_source(p, mesh):
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("CG", p, (mesh.geometry.dim,))))
    pt_cloud_source = pyvista.PolyData(mesh.geometry.x)
    febug.plot_streamlines_from_source(u, pt_cloud_source, max_time=1.0)

    mesh_source = febug.plot._to_pyvista_grid(mesh, mesh.topology.dim)
    febug.plot_streamlines_from_source(u, mesh_source, max_time=1.0)


@pytest.mark.parametrize("mesh", all_meshes)
def test_plot_mesh(mesh):
    for tdim in range(mesh.topology.dim + 1):
        mesh.topology.create_entities(tdim)
        febug.plot_mesh(mesh, tdim=tdim)


@pytest.mark.parametrize("mesh", all_meshes)
@pytest.mark.parametrize("p", [1, 2])
def test_plot_dofmap(p, mesh):
    V = dolfinx.fem.functionspace(mesh, ("CG", p))
    febug.plot_dofmap(V)


_plot_elements = (("CG", 1), ("CG", 2), ("DG", 0), ("DG", 1))

@pytest.mark.parametrize("mesh", all_meshes)
@pytest.mark.parametrize("e", _plot_elements)
def test_plot_function_dofs(e, mesh):
    V = dolfinx.fem.functionspace(mesh, e)
    u = dolfinx.fem.Function(V)
    febug.plot_function_dofs(u, fmt=".3e")


@pytest.mark.parametrize("vec_dim", [2, 3, 4])
@pytest.mark.parametrize("mesh", all_meshes)
@pytest.mark.parametrize("e", _plot_elements)
def test_plot_vector_function_dofs(e, mesh, vec_dim):
    V = dolfinx.fem.functionspace(mesh, (*e, (vec_dim,)))
    u = dolfinx.fem.Function(V)
    febug.plot_function_dofs(u, fmt=".3e")


@pytest.mark.parametrize("mesh", all_meshes)
def test_plot_meshtags(mesh):
    for t in range(mesh.topology.dim + 1):
        mesh.topology.create_entities(t)
        indices = np.arange(mesh.topology.index_map(t).size_local)
        values = np.ones_like(indices, dtype=np.int32)
        mt = dolfinx.mesh.meshtags(mesh, t, indices, values)
        febug.plot_meshtags(mt, mesh)


@pytest.mark.parametrize("mesh", all_meshes)
def test_plot_mesh_quality(mesh):
    for d in range(1, mesh.topology.dim+1):
        mesh.topology.create_entities(d)
        febug.plot_mesh_quality(
            mesh, d,
            entities=np.arange(mesh.topology.index_map(d).size_local))


@pytest.mark.parametrize("mesh", all_meshes)
def test_plot_entity_indices(mesh):
    for d in range(mesh.topology.dim+1):
        mesh.topology.create_entities(d)
        febug.plot_entity_indices(mesh, d)


@pytest.mark.parametrize("mesh", all_meshes)
def test_plot_entity_indices_global(mesh):
    for d in range(mesh.topology.dim+1):
        mesh.topology.create_entities(d)
        febug.plot_entity_indices_global(mesh, d)


@pytest.mark.parametrize(
    "mesh,tdim", [pytest.param(msh, tdim, marks=pytest.mark.xfail(
        tdim not in (0, msh.topology.dim),
        reason="Original indices only available for vertices and cells",
        strict=True))
    for msh in all_meshes for tdim in range(msh.topology.dim + 1)])
def test_plot_entity_indices_original(mesh, tdim):
    mesh.topology.create_entities(tdim)
    febug.plot_entity_indices_original(mesh, tdim)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_plot_point_cloud(dim):
    np.random.seed(1)
    pc = np.random.rand(10, 3)
    if dim < 3:
        pc[:,(3-dim):] = 0.0
    febug.plot_point_cloud(pc)
