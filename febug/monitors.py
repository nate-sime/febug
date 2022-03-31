from mpi4py import MPI


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