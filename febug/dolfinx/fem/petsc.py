import typing
import ufl
import dolfinx.fem.petsc

import febug

from dolfinx.fem.petsc import *


def _object_count_check(obj_str, count, limit):
    if count >= limit:
        febug.report_issue(
            f"{obj_str} creation count = {count}. Reuse data structures "
            f"whenever possible.")


class LinearProblem(dolfinx.fem.petsc.LinearProblem):

    _count = 0
    _limit = 10

    def __init__(self, a: ufl.form.Form, L: ufl.form.Form,
                 bcs: typing.List[dolfinx.fem.DirichletBCMetaClass] = [],
                 u: dolfinx.fem.Function = None, petsc_options={},
                 form_compiler_options={}, jit_options={}):
        super().__init__(a, L, bcs=bcs, u=u, petsc_options=petsc_options,
                         form_compiler_options=form_compiler_options,
                         jit_options=jit_options)
        LinearProblem._count += 1
        _object_count_check("LinearProblem", LinearProblem._count,
                            LinearProblem._limit)


class NonlinearProblem(dolfinx.fem.petsc.NonlinearProblem):

    _count = 0
    _limit = 10

    def __init__(self, F: ufl.form.Form, u: dolfinx.fem.Function,
                 bcs: typing.List[dolfinx.fem.DirichletBCMetaClass] = [],
                 J: ufl.form.Form = None, form_compiler_options={},
                 jit_options={}):
        super().__init__(F, u, bcs=bcs, J=J,
                         form_compiler_options=form_compiler_options,
                         jit_options=jit_options)
        febug.dolfinx.nls.search_for_potential_singularity(self.L)
        febug.dolfinx.nls.search_for_potential_singularity(self.a)
        NonlinearProblem._count += 1
        _object_count_check("NonlinearLinearProblem", NonlinearProblem._count,
                            NonlinearProblem._limit)


# -- Matrix instantiation ----------------------------------------------------
class _MatrixCount:
    _count = 0
    _limit = 10


def create_matrix(a: dolfinx.fem.FormMetaClass, mat_type=None) -> PETSc.Mat:
    _MatrixCount._count += 1
    _object_count_check("Matrix", _MatrixCount._count, _MatrixCount._limit)
    return dolfinx.fem.petsc.create_matrix(a, mat_type)


def create_matrix_block(
        a: typing.List[typing.List[dolfinx.fem.FormMetaClass]]) -> PETSc.Mat:
    _MatrixCount._count += 1
    _object_count_check("Matrix", _MatrixCount._count, _MatrixCount._limit)
    return dolfinx.fem.petsc.create_matrix_block(a)


def create_matrix_nest(
        a: typing.List[typing.List[dolfinx.fem.FormMetaClass]]) -> PETSc.Mat:
    _MatrixCount._count += 1
    _object_count_check("Matrix", _MatrixCount._count, _MatrixCount._limit)
    return dolfinx.fem.petsc.create_matrix_nest(a)
