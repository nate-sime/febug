from mpi4py import MPI
import dolfinx.nls.petsc

import febug

from dolfinx.nls.petsc import *


class NewtonSolver(dolfinx.nls.petsc.NewtonSolver):

    def __init__(self, comm: MPI.Intracomm,
                 problem: dolfinx.fem.petsc.NonlinearProblem):
        febug.dolfinx.nls.search_for_potential_singularity(problem.L)
        febug.dolfinx.nls.search_for_potential_singularity(problem.a)
        super().__init__(comm, problem)