from mpi4py import MPI
import numpy as np

import python.febug.meshquality

import dolfinx.io

# with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
#         "/home/nsime/Projects/flat-slab/examples/lib_impls/meshes_created/flat_slab_cut.xdmf",
#                          "r") as fi:
#     mesh = fi.read_mesh(
#         ghost_mode=dolfinx.cpp.mesh.GhostMode.none,
#         name="subduction_zone")

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
dhangles = python.febug.meshquality.dihedral_angles(mesh)

import matplotlib.pyplot as plt
plt.hist(np.degrees(np.array(dhangles)), bins=100, range=(0, 180.0))
plt.yscale("log")
plt.show()