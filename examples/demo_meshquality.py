import numpy as np
import dolfinx
from mpi4py import MPI
import febug.meshquality
import matplotlib.pyplot as plt
import pyvista


def generate_initial_mesh(algorithm):
    import gmsh
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm)
        model = gmsh.model()
        model_name = "mesh"
        model.add(model_name)
        model.setCurrent(model_name)

        model.occ.addSphere(0, 0, 0, 1)

        model.occ.synchronize()
        model.mesh.generate(3)

        x = dolfinx.io.extract_gmsh_geometry(model, model_name=model_name)
        element_types, element_tags, node_tags = model.mesh.getElements(dim=3)
        assert len(element_types) == 1
        etype = comm.bcast(element_types[0], root=0)
        name, dim, order, num_nodes, local_coords, num_first_order_nodes = \
            model.mesh.getElementProperties(etype)
        cells = node_tags[0].reshape(-1, num_nodes) - 1
        num_nodes = comm.bcast(cells.shape[1], root=0)
    else:
        num_nodes = comm.bcast(None, root=0)
        etype = comm.bcast(None, root=0)
        cells, x = np.empty([0, num_nodes]), np.empty([0, 3])

    mesh = dolfinx.mesh.create_mesh(
        comm, cells, x,
        dolfinx.io.ufl_mesh_from_gmsh(etype, x.shape[1]))
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
plt.show()
subplotter.show()
