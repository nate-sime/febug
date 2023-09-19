import numpy as np
import dolfinx
from mpi4py import MPI
import febug.meshquality
import matplotlib.pyplot as plt
import pyvista


def generate_initial_mesh(algorithm):
    import gmsh
    comm = MPI.COMM_WORLD
    gmsh.initialize()

    if comm.rank == 0:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm)
        model = gmsh.model()
        model_name = f"mesh_{algorithm}"
        model.add(model_name)
        model.setCurrent(model_name)

        sphere = model.occ.addSphere(0, 0, 0, 1)
        model.occ.synchronize()
        model.addPhysicalGroup(3, [sphere])
        model.mesh.generate(3)

    partitioner = dolfinx.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.none)
    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3, partitioner=partitioner)
    gmsh.finalize()
    return mesh


if MPI.COMM_WORLD.rank == 0:
    fig, axs = plt.subplots(1, 3, sharey=True)
subplotter = pyvista.Plotter(shape=(1, 3))

for ax_num, algorithm in enumerate((1, 4, 7)):
    subplotter.subplot(0, ax_num)
    subplotter.add_title(f"Algorithm {algorithm}")

    counts = []
    edges = np.linspace(0, 90, 50)
    for j in range(3):
        if j == 0:
            mesh = generate_initial_mesh(algorithm)
            febug.plot.plot_mesh_quality(mesh, mesh.topology.dim,
                                         plotter=subplotter)
        else:
            mesh.topology.create_connectivity(1, 0)
            mesh = dolfinx.mesh.refine(mesh, redistribute=True)

        count, _ = febug.meshquality.histogram_gather(
            febug.meshquality.pyvista_entity_quality(
                mesh, mesh.topology.dim, quality_measure="min_angle"),
            bins=edges)

        if mesh.comm.rank == 0:
            axs[ax_num].bar(edges[:-1], count, width=edges[1:] - edges[:-1],
                      zorder=-j)

    if mesh.comm.rank == 0:
        axs[ax_num].set_yscale("log")
        axs[ax_num].grid()
        axs[ax_num].set_xlim((edges.min(), edges.max()))
        axs[ax_num].set_xlabel("minimum dihedral angle")
        if ax_num == 0:
            axs[ax_num].set_ylabel("frequency")
        axs[ax_num].set_title(f"Algorithm {algorithm}")

if mesh.comm.rank == 0:
    plt.show()
subplotter.show()
