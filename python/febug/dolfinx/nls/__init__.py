import typing
import dolfinx.fem
import ufl
import numpy as np

from python import febug


def search_for_potential_singularity(
        form: typing.Union[dolfinx.fem.FormMetaClass, ufl.Form],
        rtol: float=1.e-5, atol: float=1.e-8):
    coeffs = form.coefficients() if isinstance(form, ufl.Form) \
        else form.coefficients
    zero_idxs = [np.argwhere(np.isclose(coeff.x.array, 0.0, rtol, atol)
                             ).flatten()
                 for coeff in coeffs]
    zeroes_found = tuple(zip(coeffs, zero_idxs))

    febug.report_issue(f"Potential singularities found:\n" + "\n".join(
        (f"{coeff.name} at DoFs {zero_idxs}") for coeff, zero_idxs in
        zeroes_found))

    return zeroes_found
