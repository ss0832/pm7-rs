# SPDX-License-Identifier: GPL-3.0-or-later
"""PM7-family semiempirical quantum chemistry for Python and ASE.

Molecules and periodic systems (1-D polymers, 2-D layers, 3-D crystals) go through the same
functions: pass a ``cell`` and the calculation becomes periodic. See :mod:`pm7_rs.native` for
the periodic keywords and :class:`pm7_rs.ase.PM7` for the ASE calculator.
"""

from importlib.metadata import PackageNotFoundError, version as _version

from . import constants, native

#: Everything :mod:`pm7_rs.native` exports, re-exported here.
#:
#: Kept in step with ``native.__all__`` by a test rather than by hand. Through v0.2.1 this list
#: was six names short — ``phonons``, ``band_structure``, ``divide_and_conquer``, ``dfpt``,
#: ``born_charges`` and ``molden`` all worked through ``pm7_rs.native`` while ``pm7_rs.dfpt``
#: raised ``AttributeError``. Since the documentation said in the same breath that perturbation
#: theory "reaches no binding", the natural reading was that the feature did not exist outside
#: Rust. It did; only the alias was missing.
__all__ = [
    "native",
    "constants",
    "single_point",
    "gradient",
    "forces",
    "stress",
    "optimize",
    "frequencies",
    "hessian",
    "orbitals",
    "scf_stability",
    "vibrations",
    "Vibrations",
    "phonons",
    "band_structure",
    "divide_and_conquer",
    "dfpt",
    "born_charges",
    "dielectric_with_extent",
    "finite_field",
    "berry_polarization",
    "static_dielectric",
    "dielectric_origin_sensitivity",
    "polarizability",
    "molden",
]

single_point = native.single_point
gradient = native.gradient
forces = native.forces
stress = native.stress
optimize = native.optimize
frequencies = native.frequencies
hessian = native.hessian
orbitals = native.orbitals
scf_stability = native.scf_stability
vibrations = native.vibrations
Vibrations = native.Vibrations
phonons = native.phonons
band_structure = native.band_structure
divide_and_conquer = native.divide_and_conquer
dfpt = native.dfpt
born_charges = native.born_charges
dielectric_with_extent = native.dielectric_with_extent
finite_field = native.finite_field
berry_polarization = native.berry_polarization
static_dielectric = native.static_dielectric
dielectric_origin_sensitivity = native.dielectric_origin_sensitivity
polarizability = native.polarizability
molden = native.molden

try:
    # One source of truth: the version comes from the installed distribution metadata, which
    # maturin fills from Cargo.toml. Hard-coding it here would be a third place to forget.
    __version__ = _version("pm7-rs-python")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"
