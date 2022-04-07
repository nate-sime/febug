import dolfinx.mesh
import numba
import numpy as np
import tqdm


def dihedral_angles(mesh: dolfinx.mesh.Mesh):
    mesh.topology.create_connectivity(3, 1)
    mesh.topology.create_connectivity(1, 0)
    c2e = mesh.topology.connectivity(3, 1)
    e2v = mesh.topology.connectivity(1, 0)

    x = mesh.geometry.x

    dhangles = []
    for c in tqdm.trange(mesh.topology.index_map(3).size_local):
        edges = c2e.links(c)

        for i in range(6):
            i0 = e2v.links(edges[i])[0]
            i1 = e2v.links(edges[i])[1]
            i2 = e2v.links(edges[5-i])[0]
            i3 = e2v.links(edges[5-i])[1]

            p0 = x[i0]
            v1 = x[i1] - p0
            v2 = x[i2] - p0
            v3 = x[i3] - p0

            print(i0, i1, i2, i3)

            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            v3 /= np.linalg.norm(v3)

            cphi = (v2.dot(v3) - v1.dot(v2)*v1.dot(v3)) / \
                   (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v1, v3)))

            dhangle = np.arccos(cphi)
            dhangles.append(dhangle)
    return dhangles


def dihedral_angles2(mesh: dolfinx.mesh.Mesh):
    mesh.topology.create_connectivity(3, 1)
    mesh.topology.create_connectivity(1, 0)
    c2e = mesh.topology.connectivity(3, 1)
    e2v = mesh.topology.connectivity(1, 0)

    x = mesh.geometry.x

    ncells = mesh.topology.index_map(3).size_local
    nedges = mesh.topology.index_map(1).size_local

    dhangles = np.zeros(nedges, dtype=np.double)
    half_planes = []
    for c in tqdm.trange(ncells):
        edges = c2e.links(c)

        for i in range(6):
            i0 = e2v.links(edges[i])[0]
            i1 = e2v.links(edges[i])[1]
            i2 = e2v.links(edges[5-i])[0]
            i3 = e2v.links(edges[5-i])[1]

            p0 = x[i0]
            v1 = x[i1] - p0
            v2 = x[i2] - p0
            v3 = x[i3] - p0

            print(i0, i1, i2, i3)

            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            v3 /= np.linalg.norm(v3)

            cphi = (v2.dot(v3) - v1.dot(v2)*v1.dot(v3)) / \
                   (np.linalg.norm(np.cross(v1, v2)) * np.linalg.norm(np.cross(v1, v3)))

            dhangle = np.arccos(cphi)
            dhangles.append(dhangle)
    return dhangles