import math
import dolfinx.mesh
import numpy as np
from mpi4py import MPI
import febug.plot


def pyvista_entity_quality(
        mesh: dolfinx.mesh.Mesh, tdim, entities=None,
        quality_measure: str="scaled_jacobian",
        progress_bar=False):
    if mesh.topology.index_map(tdim) is None:
        mesh.topology.create_entities(tdim)
    pvmesh = febug.plot._to_pyvista_grid(mesh, tdim, entities=entities)
    qual = pvmesh.compute_cell_quality(quality_measure=quality_measure,
                                       progress_bar=progress_bar)
    return qual["CellQuality"]


def histogram_gather(data, bins, weights=None,
                     comm=MPI.COMM_WORLD, rank=0):
    counts, edges = np.histogram(
        data, bins=bins, weights=weights)

    if comm.rank == rank:
        counts_global = np.zeros_like(counts)
    else:
        counts_global = None

    comm.Reduce([counts, MPI.INT], [counts_global, MPI.INT], op=MPI.SUM,
                root=rank)

    return counts_global, edges


def hist_unicode(data, bins, weights=None,
                 cmd_width=80, title=None, comm=MPI.COMM_WORLD):
    counts, edges = histogram_gather(
        data, bins=bins, weights=weights, comm=comm, rank=0)

    if comm.rank != 0:
        return

    if title is not None:
        print(title)

    barv = "▏▎▍▌▋▊▉█"
    barh = "█▇▆▅▄▃▂▁"

    xaxis_tick_buffer = 8

    x_intervals = np.logspace(
        0, math.ceil(math.log10(counts.max())), cmd_width + 1)
    num_full_blks = np.searchsorted(x_intervals, counts)

    # Draw the main bar plots
    print(f"{edges[0]:>{xaxis_tick_buffer}.2f} _", flush=True)
    for j in range(1, len(edges)):
        print(f"{edges[j]:>{xaxis_tick_buffer}.2f} _", end="", flush=True)
        print(barv[-1] * num_full_blks[j-1])

    # Figure out x ticks and their spacing
    major_ticks = np.arange(0, math.ceil(math.log10(counts.max()))+1)
    tick_pos = np.searchsorted(x_intervals, 10**major_ticks)

    pos = xaxis_tick_buffer + 2
    tick_str = "".join(
        "┊" if idx in tick_pos else " " for idx in range(cmd_width+1))
    print(" "*pos + tick_str)

    num2ss = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    xticks_str = list(" " * (pos + cmd_width))

    for tick in major_ticks:
        tick_label = f"10{str(tick).translate(num2ss)}"
        offset = -1
        local_pos_left = tick_pos[tick] + pos + offset
        local_pos_right = tick_pos[tick] + pos + len(tick_label) + offset
        xticks_str[local_pos_left:local_pos_right] = tick_label
    print("".join(xticks_str))