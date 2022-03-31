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
    term_sz = shutil.get_terminal_size((80, 20))
    start_idx = 3

    def interval_str(ival):
        exponent = str(ival).translate(num2ss)
        return f"10{exponent}" + " " * (start_idx - len(exponent))

    rnorms = []
    max_inteval = 0
    chars = [".", "⋅", "˙"]
    num2ss = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁴⁵⁶⁷⁹")
    intervals = []
    its = []
    def monitor(ctx, it, rnorm):
        if comm.rank != 0:
            return
        rnorms.append(rnorm)
        its.append(its)

        exp = math.log10(rnorm)
        interval = math.floor(exp), math.ceil(exp)
        if it == 0:
            max_interval = interval[1]
            intervals.append(interval[1])
            print(interval_str(max_interval), end="", flush=True)

        if interval[1] < intervals[-1]:
            print("\n", end="")
            print(interval_str(interval[1]) + " "*(len(its)-1), end="", flush=True)
            intervals.append(interval[1])

        char = chars[math.floor((exp - interval[0]) * len(chars))]
        print(char, end="", flush=True)

    return monitor
