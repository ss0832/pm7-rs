# SPDX-License-Identifier: GPL-3.0-or-later
"""PM7-family semiempirical quantum chemistry for Python and ASE."""

from . import native

__all__ = [
    "native",
    "single_point",
    "gradient",
    "forces",
    "optimize",
    "frequencies",
    "hessian",
]

single_point = native.single_point
gradient = native.gradient
forces = native.forces
optimize = native.optimize
frequencies = native.frequencies
hessian = native.hessian

__version__ = "0.1.2"
