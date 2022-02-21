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
def test_plot_function(p, mesh):
    u = dolfinx.fem.Function(dolfinx.fem.FunctionSpace(mesh, ("CG", p)))
    febug.plot(u)


@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
def test_plot_mesh(mesh):
    febug.plot(mesh)


@pytest.mark.parametrize("mesh", meshes1D + meshes2D + meshes3D)
@pytest.mark.parametrize("p", [1, 2])
def test_plot_dofmap(p, mesh):
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", p))
    febug.plot_dofmap(V)


@pytest.mark.parametrize("mesh", meshes2D)
def test_plot_meshtags(mesh):
    for t in range(1, mesh.topology.dim):
        mesh.topology.create_entities(t)
        indices = np.arange(mesh.topology.index_map(t).size_local)
        values = np.ones_like(indices, dtype=np.int32)
        mt = dolfinx.mesh.MeshTags(mesh, t, indices, values)
        febug.plot(mt)
