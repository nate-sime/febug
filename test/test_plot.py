import numpy as np
import pytest
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


@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
@pytest.mark.parametrize("p", [1, 2])
def test_plot_function_cg(p, mesh):
    u = dolfinx.fem.Function(dolfinx.fem.FunctionSpace(mesh, ("CG", p)))
    febug.plot_function(u)


@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
@pytest.mark.parametrize("p", [0, 1, 2])
def test_plot_function_dg(p, mesh):
    u = dolfinx.fem.Function(dolfinx.fem.FunctionSpace(mesh, ("DG", p)))
    febug.plot_function(u)


@pytest.mark.parametrize(
    "mesh",
    [pytest.param(msh, marks=pytest.mark.xfail(
        msh.topology.dim == 1, reason="No warping 1D meshes"))
     for msh in meshes1D + meshes2D + meshes3D])
@pytest.mark.parametrize("p", [1, 2])
def test_plot_warp(p, mesh):
    u = dolfinx.fem.Function(dolfinx.fem.VectorFunctionSpace(mesh, ("CG", p)))
    febug.plot_warp(u)


@pytest.mark.parametrize(
    "mesh",
    [pytest.param(msh, marks=pytest.mark.xfail(
        msh.topology.dim == 1, reason="No warping 1D meshes"))
     for msh in meshes1D + meshes2D + meshes3D])
@pytest.mark.parametrize("p", [1, 2])
def test_plot_quiver(p, mesh):
    u = dolfinx.fem.Function(dolfinx.fem.VectorFunctionSpace(mesh, ("CG", p)))
    febug.plot_quiver(u)


@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
def test_plot_mesh(mesh):
    for tdim in range(mesh.topology.dim + 1):
        mesh.topology.create_entities(tdim)
        febug.plot_mesh(mesh, tdim=tdim)


@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
@pytest.mark.parametrize("p", [1, 2])
def test_plot_dofmap(p, mesh):
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", p))
    febug.plot_dofmap(V)


_plot_elements = [("CG", 1), ("CG", 2), ("DG", 0), ("DG", 1)]

@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
@pytest.mark.parametrize("e", _plot_elements)
def test_plot_function_dofs(e, mesh):
    V = dolfinx.fem.FunctionSpace(mesh, e)
    u = dolfinx.fem.Function(V)
    febug.plot_function_dofs(u, fmt=".3e")


@pytest.mark.parametrize("vec_dim", [2, 3, 4])
@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
@pytest.mark.parametrize("e", _plot_elements)
def test_plot_vector_function_dofs(e, mesh, vec_dim):
    V = dolfinx.fem.VectorFunctionSpace(mesh, e, dim=vec_dim)
    u = dolfinx.fem.Function(V)
    febug.plot_function_dofs(u, fmt=".3e")


@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
def test_plot_meshtags(mesh):
    for t in range(mesh.topology.dim + 1):
        mesh.topology.create_entities(t)
        indices = np.arange(mesh.topology.index_map(t).size_local)
        values = np.ones_like(indices, dtype=np.int32)
        mt = dolfinx.mesh.meshtags(mesh, t, indices, values)
        febug.plot_meshtags(mt, mesh)


@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
def test_plot_mesh_quality(mesh):
    for d in range(1, mesh.topology.dim+1):
        febug.plot_mesh_quality(mesh, d)


@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
def test_plot_entity_indices(mesh):
    for d in range(mesh.topology.dim+1):
        mesh.topology.create_entities(d)
        febug.plot_entity_indices(mesh, d)
