import numpy as np
import pytest
from mpi4py import MPI
import dolfinx
import dolfinx.mesh
import dolfinx.fem
import febug


meshes1D = [dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 1)]
meshes2D = [dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)]
meshes3D = [dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)]


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
    febug.plot_mesh(mesh)


@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
@pytest.mark.parametrize("p", [1, 2])
def test_plot_dofmap(p, mesh):
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", p))
    febug.plot_dofmap(V)


@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
def test_plot_meshtags(mesh):
    for t in range(0, mesh.topology.dim):
        mesh.topology.create_entities(t)
        indices = np.arange(mesh.topology.index_map(t).size_local)
        values = np.ones_like(indices, dtype=np.int32)
        mt = dolfinx.mesh.meshtags(mesh, t, indices, values)
        febug.plot_meshtags(mt)


@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
def test_plot_mesh_quality(mesh):
    D = mesh.topology.dim
    for d in range(1, D+1):
        febug.plot_mesh_quality(mesh, d)
