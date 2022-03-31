from mpi4py import MPI
import math


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
    return MonitorMPL(comm).monitor


def monitor_unicode(comm=MPI.COMM_WORLD):
    import shutil
    term_sz = shutil.get_terminal_size()
    start_idx = 4

    def y_axis_exp_str(ival):
        exponent = str(ival).translate(num2ss)
        return f"10{exponent}" + " " * (start_idx - len(exponent))

    chars = "_,⎵.-'¯`⎴"
    num2ss = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁴⁵⁶⁷⁹")
    rnorms = []
    intervals = []
    carriage_idx = [0]
    def monitor(ctx, it, rnorm):
        if comm.rank != 0:
            return

        if rnorm == 0.0:
            print("\nrnorm = 0.0")
            return

        rnorms.append(rnorm)
        carriage_idx[0] += 1

        exp = math.log10(rnorm)
        c_exp = math.ceil(exp)
        if it == 0:
            intervals.append(c_exp)
            print(f"\nStarting iteration {it}", flush=True)
            print(y_axis_exp_str(c_exp), end="", flush=True)

        if c_exp < intervals[-1]:
            intervals.append(c_exp)
            print("\n", end="")
            print(y_axis_exp_str(c_exp) + " "*(carriage_idx[0]-1), end="", flush=True)

        char = chars[math.floor((exp - math.floor(exp)) * len(chars))]
        print(char, end="", flush=True)

        if carriage_idx[0] + start_idx >= term_sz.columns:
            print(f"\nStarting iteration {it}", flush=True)
            print(y_axis_exp_str(c_exp), end="", flush=True)
            carriage_idx[0] = 0

    return monitor
