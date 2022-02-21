import typing
import ufl
import dolfinx.fem.petsc

import febug

from dolfinx.fem.petsc import *


class NonlinearProblem(dolfinx.fem.petsc.NonlinearProblem):

    def __init__(self, F: ufl.form.Form, u: dolfinx.fem.Function,
                 bcs: typing.List[dolfinx.fem.DirichletBCMetaClass] = [],
                 J: ufl.form.Form = None, form_compiler_params={},
                 jit_params={}):
        super().__init__(F, u, bcs=bcs, J=J,
                         form_compiler_params=form_compiler_params,
                         jit_params=jit_params)
        febug.dolfinx.nls.search_for_potential_singularity(self.L)
        febug.dolfinx.nls.search_for_potential_singularity(self.a)
