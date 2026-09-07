# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit conversions used at the Python boundary.

Derived from the two CODATA-2018 values the Rust core uses (``src/constants.rs``) rather than
transcribed, so the two layers cannot drift apart. MOPAC v23.2.5 uses the same CODATA set, which
is what makes the heats of formation comparable to the printed digit.

Everything here is a plain float; nothing in this module has any non-ASCII content, so it stays
importable on a machine whose preferred encoding is not UTF-8.
"""

from __future__ import annotations

#: eV per Hartree (CODATA 2018; MOPAC ``fpcref(1,4)``).
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

#: Bohr radius in Angstrom (CODATA 2018; MOPAC ``fpcref(1,3)``).
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

#: One eV in kcal/mol (CODATA 2018; MOPAC ``fpcref(1,9)``).
EV_TO_KCAL = 23.06054783061903
KCAL_TO_EV = 1.0 / EV_TO_KCAL

#: An atomic-unit dipole (e*a0) in Debye.
AU_DIPOLE_TO_DEBYE = 2.541746473

#: One Debye in e*Angstrom -- the conversion the ASE calculator needs for ``dipole``.
#:
#: Derived, not tabulated: one e*a0 is ``AU_DIPOLE_TO_DEBYE`` Debye and ``a0`` is the Bohr radius
#: in Angstrom, so one Debye is ``a0 / AU_DIPOLE_TO_DEBYE`` e*Angstrom. Equals 0.2081943..., which
#: used to sit in ``ase.py`` as a bare literal with nothing tying it to the constants it comes
#: from.
DEBYE_TO_E_ANGSTROM = BOHR_TO_ANGSTROM / AU_DIPOLE_TO_DEBYE

#: The elementary charge as Debye per Angstrom: a dipole derivative in atomic units (e) times
#: this is the D/A that spectroscopy -- and MOPAC's ``DIPT`` -- reports. Equals 4.803204...
E_IN_DEBYE_PER_ANGSTROM = 1.0 / DEBYE_TO_E_ANGSTROM

__all__ = [
    "HARTREE_TO_EV",
    "EV_TO_HARTREE",
    "BOHR_TO_ANGSTROM",
    "ANGSTROM_TO_BOHR",
    "EV_TO_KCAL",
    "KCAL_TO_EV",
    "AU_DIPOLE_TO_DEBYE",
    "DEBYE_TO_E_ANGSTROM",
    "E_IN_DEBYE_PER_ANGSTROM",
]
