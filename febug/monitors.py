from mpi4py import MPI
import shutil
import math


def monitor_text_petsc(comm=MPI.COMM_WORLD):
    """
    A standard text monitor which employs PETSc's system print function

    :param comm: MPI communicator on which rank 0 will host the plot
    :return: monitor function compatible with petsc4py.PETSc.KSP.setMonitor
    """
    from petsc4py import PETSc
    def monitor(ctx, it, rnorm):
        PETSc.Sys.Print(f"Iteration: {it:>4d}, |r| = {rnorm:.3e}")
    return monitor


class MonitorMPL:

    def __init__(self, comm):
        import matplotlib.pyplot as plt

        self.rnorms = []
        self.its = []
        self.count = -1
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Iterations")
        self.ax.set_ylabel(r"$\Vert \vec{r} \Vert$")
        self.ax.grid("on")

        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.comm = comm

    def monitor(self, ctx, it, rnorm):
        if self.comm.rank == 0:
            if it == 0:
                self.count += 1
                self.its.append([])
                self.rnorms.append([])
            self.its[self.count].append(it)
            self.rnorms[self.count].append(rnorm)
            for count, (its, rnorms) in enumerate(zip(self.its, self.rnorms)):
                self.ax.semilogy(its, rnorms, "-o",
                                 color=self.colors[count % len(self.colors)])
            self.fig.canvas.draw()
            self.fig.canvas.manager.show()
            self.fig.canvas.start_event_loop(1e-9)


def monitor_mpl(comm=MPI.COMM_WORLD):
    """
    Monitor a PETSc KSP by drawing a graph using matplotlib.

    :param comm: MPI communicator on which rank 0 will host the plot
    :return: monitor function compatible with petsc4py.PETSc.KSP.setMonitor
    """
    return MonitorMPL(comm).monitor


def monitor_unicode_graph(comm=MPI.COMM_WORLD):
    """
    Monitor a PETSc KSP by drawing a crude graph in the terminal stdout. We
    deliberately do not use carriage returns in this function so not to pollute
    output in text logging as required by, for example, HPCs.

    Since carriage returns are not used, the symbol "D" is employed to signify
    that the residual has diverged.

    :param comm: MPI communicator on which rank 0 will print
    :return: monitor function compatible with petsc4py.PETSc.KSP.setMonitor
    """
    term_sz = shutil.get_terminal_size()
    start_idx = 4  # Buffer carriage idx from y axis labels

    # Given an exponent value, format the yaxis label
    def y_axis_exp_str(exp_val):
        exp_str = str(exp_val).translate(num2ss)
        return f"10{exp_str}" + " " * (start_idx - len(exp_str))

    # Characters used to draw lines and superscript exponents
    diverged_char = "D"
    # chars = "_,⎵.-'¯`⎴"
    chars = "⎺⎻—⎽_"[::-1]
    num2ss = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

    rnorms = []         # Store residuals
    intervals = [0]     # Store exponents
    carriage_idx = [0]  # Store cursor position
    def monitor(ctx, it, rnorm):
        if comm.rank != 0:
            return

        # Cannot take log10(0.0)
        if rnorm == 0.0:
            print("\nrnorm = 0.0")
            return

        rnorms.append(rnorm)
        carriage_idx[0] += 1

        exp = math.log10(rnorm)
        i_exp = math.floor(exp)

        # On the first iteration set up a y axis label at the exponent
        if it == 0:
            intervals[0] = i_exp
            print(f"\nStarting iteration {it}", flush=True)
            print(y_axis_exp_str(i_exp), end="", flush=True)

        # The y axis label needs to be redrawn on the new line if we're in a
        # new range
        if i_exp < intervals[0]:
            # Fill any missing yaxis labels
            for j in range(i_exp, intervals[0])[::-1]:
                print("\n" + y_axis_exp_str(j), end="", flush=True)
            print(" "*(carriage_idx[0]-1), end="", flush=True)
            intervals[0] = i_exp

        if i_exp > intervals[0]:
            print(diverged_char, end="", flush=True)
        else:
            # Compute the character within the exponent interval to be used
            char = chars[math.floor((exp - math.floor(exp)) * len(chars))]
            print(char, end="", flush=True)

        # If we've exceeded the terminal column limit, generate a new frame
        # below
        if carriage_idx[0] + start_idx >= term_sz.columns:
            print(f"\nStarting iteration {it}", flush=True)
            print(y_axis_exp_str(i_exp), end="", flush=True)
            carriage_idx[0] = 0

    return monitor
