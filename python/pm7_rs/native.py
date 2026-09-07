# SPDX-License-Identifier: GPL-3.0-or-later
"""Native PM7-family API (atomic units at the public boundary).

Every entry point accepts:

* ``charge``       — formal charge (e) of the molecule, or **of the unit cell** for a periodic
  system. Charged periodic cells are supported; see ``periodic`` below.
* ``multiplicity`` — spin multiplicity 2S+1 (1 = singlet, 2 = doublet, …).
* ``method``       — ``"pm7"``, ``"pm7-ts"``, ``"pm7-hh"``, ``"pm7-"`` (PM7-minus)…
* ``reference``    — spin treatment, independent of ``multiplicity``:
  ``"auto"`` (RHF closed shell / UHF open shell), ``"rhf"``/``"r"`` (force RHF),
  ``"uhf"``/``"u"`` (force UHF, e.g. a UHF singlet).

.. _periodic:

Periodic systems
----------------

Passing ``cell`` turns the same call into a 1-D polymer, 2-D layer, or 3-D crystal:

* ``cell``         — 3×3 lattice vectors in Ångström, one row per vector.
* ``pbc``          — which directions are periodic, e.g. ``(True, True, False)`` for a slab.
  Any pattern works, including ``(True, False, True)``: the lattice vectors are cyclically
  reordered so the periodic ones lead, and ``kpoints``, ``kpoint_shift``, ``supercell`` and
  fractional ``qpoints`` are reordered with them, so everything stays in your axis order.
  Through 0.2.2 a non-leading pattern was refused. Defaults to all three.
* ``kpoints``      — Monkhorst–Pack divisions, e.g. ``(4, 4, 4)``. Defaults to the Γ point.
* ``kpoint_shift`` — fractional offset of the mesh.
* ``smearing``     — ``(kind, width_ev, order)`` with kind ``"none"``, ``"fermi"``, ``"gauss"``,
  or ``"mp"``; needed for metals, ignored otherwise.
* ``pbc_mode``     — ``"ewald"`` (default, physically rigorous) or ``"mopac"``.

Periodic results carry extra keys: ``stress`` (3×3, eV/Å³), ``stress_voigt``
(``[xx, yy, zz, yz, xz, xy]``), ``pressure_gpa``, ``fermi_ev``, ``entropy_ev``, ``n_kpoints``,
``volume_angstrom3``, and the charged-cell diagnostics ``ewald_ev``, ``background_ev``,
``makov_payne_ev``.

Two energies, and which is which
--------------------------------
``energy_ev`` is the internal electronic energy at the converged occupations. ``free_energy_ev``
is the **Mermin electronic** free energy ``E − TS``: the same number plus ``entropy_ev``, which is
the smearing's occupation entropy already carrying its sign. Without ``smearing`` the entropy is
exactly zero and the two are identical, bit for bit. With it they differ, and the *free* one is
what the forces differentiate — so an ASE optimizer or MD run asking for the force-consistent
energy gets that one.

Neither is a **thermochemical** free energy. No vibrational partition function enters either, so
neither is the Gibbs free energy a normal-mode analysis would give; ``frequencies`` and
``vibrations`` return harmonic frequencies and nothing is derived from them. ``free_energy_ev`` is
electronic, at the smearing width, and is about how a metal's bands are occupied rather than about
molecular vibrations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from . import _native

if TYPE_CHECKING:  # pragma: no cover - the stub is only read by a type checker
    from ._native import SinglePointResult

#: Every public function this module defines.
#:
#: ``dfpt``, ``born_charges`` and ``molden`` were missing from this list through v0.2.1 although
#: all three were defined and worked. `from pm7_rs.native import *` therefore did not bring them
#: in, and any tooling that reads ``__all__`` reported them as non-existent. A test now compares
#: this list against the module's own contents, because that is the only thing that keeps the two
#: in step.
__all__ = [
    "single_point",
    "gradient",
    "forces",
    "stress",
    "optimize",
    "frequencies",
    "hessian",
    "phonons",
    "band_structure",
    "divide_and_conquer",
    "orbitals",
    "scf_stability",
    "vibrations",
    "Vibrations",
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


def _input(numbers, positions):
    numbers = [int(z) for z in np.asarray(numbers).reshape(-1)]
    positions = np.asarray(positions, dtype=float).reshape(len(numbers), 3).tolist()
    return numbers, positions


def _field(field):
    """A uniform external electric field as three floats in volts/Angstrom, or ``None``.

    The sign is MOPAC's ``FIELD=``, which is the *potential gradient* rather than the physical
    field, so the interaction energy is ``+F.mu``. See ``docs/theory.md`` convention C-1.
    """
    if field is None:
        return None
    values = np.asarray(field, dtype=float).reshape(-1)
    if values.size != 3:
        raise ValueError("field must be three components [x, y, z] in volts/Angstrom")
    return values.tolist()


def _direction(direction):
    """A Cartesian direction as three floats, or ``None``.

    Used for the LO–TO limit, where only the direction matters and the length does not, so it is
    not normalized here — the Rust side does that, and doing it twice would be two places to get
    a zero vector wrong.
    """
    if direction is None:
        return None
    values = np.asarray(direction, dtype=float).reshape(-1)
    if values.size != 3:
        raise ValueError("lo_to_direction must be three Cartesian components [x, y, z]")
    return values.tolist()


def _periodic(cell, pbc, kpoints, kpoint_shift, smearing):
    """Normalize the periodic keywords into what the native layer expects.

    Accepting ASE-shaped inputs here rather than in the Rust layer keeps the boundary a plain
    list-of-lists, and lets an ``ase.cell.Cell``, a NumPy array, or a nested list all work.
    """
    out = {}
    if cell is not None:
        arr = np.asarray(cell, dtype=float)
        if arr.shape == (3,):
            # A length-3 vector is the orthorhombic shorthand ASE also accepts.
            arr = np.diag(arr)
        if arr.shape != (3, 3):
            raise ValueError(f"cell must be 3x3 (or a length-3 vector), got shape {arr.shape}")
        out["cell"] = arr.tolist()
    if pbc is not None:
        flags = np.asarray(pbc).reshape(-1)
        if flags.size == 1:
            flags = np.repeat(flags, 3)
        if flags.size != 3:
            raise ValueError("pbc must be a bool or three bools")
        out["pbc"] = [bool(x) for x in flags]
    if kpoints is not None:
        n = np.asarray(kpoints).reshape(-1)
        if n.size != 3:
            raise ValueError("kpoints must be three Monkhorst-Pack divisions, e.g. (4, 4, 4)")
        out["kpoints"] = [int(x) for x in n]
    if kpoint_shift is not None:
        s = np.asarray(kpoint_shift, dtype=float).reshape(-1)
        if s.size != 3:
            raise ValueError("kpoint_shift must have three entries")
        out["kpoint_shift"] = s.tolist()
    if smearing is not None:
        out["smearing"] = _smearing_tuple(smearing)
    return out


def _smearing_tuple(smearing):
    """Accept ``"fermi"``, ``("gauss", 0.1)``, ``{"kind": "mp", "width_ev": 0.2, "order": 1}``."""
    if isinstance(smearing, str):
        return (smearing, 0.1, 1)
    if isinstance(smearing, dict):
        kind = smearing.get("kind", "fermi")
        return (str(kind), float(smearing.get("width_ev", 0.1)), int(smearing.get("order", 1)))
    items = list(smearing)
    kind = str(items[0])
    width = float(items[1]) if len(items) > 1 else 0.1
    order = int(items[2]) if len(items) > 2 else 1
    return (kind, width, order)


def single_point(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    cell=None,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    field=None,
    dipole_origin: str | None = None,
) -> "SinglePointResult":
    """Single point; positions in Angstrom. Returns atomic-unit energetics plus the
    accepted ``charge``/``multiplicity``/``reference`` and whether UHF was used.

    An unrestricted run also returns ``spin_squared`` — ``⟨S²⟩``, MOPAC's ``(S**2)``. It is
    ``None`` for a restricted run, where the value is ``S(S+1)`` by construction, and for a k-mesh
    run, where the density available is the ``T = 0`` block rather than the whole solution. A UHF
    determinant is not a spin eigenfunction, so the gap between ``⟨S²⟩`` and ``S(S+1)`` is the
    measure of how contaminated the answer is: a clean doublet radical is near 0.75, and a
    broken-symmetry singlet — which is what ``stability="follow"`` deliberately produces — runs
    up to 1.0 at dissociation.

    With a ``cell`` the result also carries the stress tensor and the periodic diagnostics
    described in the module docstring."""
    numbers, positions = _input(numbers, positions)
    return _native.single_point(
        numbers,
        positions,
        float(charge),
        int(multiplicity),
        method,
        reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        field=_field(field),
        dipole_origin=dipole_origin,
        **_periodic(cell, pbc, kpoints, kpoint_shift, smearing),
    )


def gradient(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    cell=None,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    field=None,
    dipole_origin: str | None = None,
) -> dict:
    """Energy and analytic **gradient** ∂E/∂x (``gradient_hartree_per_bohr`` and
    ``gradient_ev_per_angstrom``); positions in Angstrom."""
    numbers, positions = _input(numbers, positions)
    return _native.gradient(
        numbers,
        positions,
        float(charge),
        int(multiplicity),
        method,
        reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        field=_field(field),
        dipole_origin=dipole_origin,
        **_periodic(cell, pbc, kpoints, kpoint_shift, smearing),
    )


def forces(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    cell=None,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    field=None,
    dipole_origin: str | None = None,
) -> dict:
    """Energy and analytic **forces** −∂E/∂x (``forces_hartree_per_bohr`` and
    ``forces_ev_per_angstrom``); positions in Angstrom."""
    numbers, positions = _input(numbers, positions)
    return _native.forces(
        numbers,
        positions,
        float(charge),
        int(multiplicity),
        method,
        reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        field=_field(field),
        dipole_origin=dipole_origin,
        **_periodic(cell, pbc, kpoints, kpoint_shift, smearing),
    )


def stress(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    cell,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    field=None,
    dipole_origin: str | None = None,
) -> dict:
    """Energy, forces, and the analytic **stress** of a periodic system, from one SCF.

    ``cell`` is required. Returns ``stress`` (3×3) and ``stress_voigt``
    (``[xx, yy, zz, yz, xz, xy]``) in eV/Å³, positive under tension — the convention ASE uses —
    alongside ``forces_ev_per_angstrom`` and the energy."""
    numbers, positions = _input(numbers, positions)
    return _native.stress(
        numbers,
        positions,
        float(charge),
        int(multiplicity),
        method,
        reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        field=_field(field),
        dipole_origin=dipole_origin,
        **_periodic(cell, pbc, kpoints, kpoint_shift, smearing),
    )


def optimize(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    cell=None,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    field=None,
    dipole_origin: str | None = None,
    relax_cell: bool = False,
    gtol: float | None = None,
    stress_tol: float | None = None,
    max_iter: int | None = None,
    stability_every: int | None = None,
) -> dict:
    """Optimize a structure on the requested PM7-family surface.

    Returns ``positions_angstrom``, ``energy_hartree``, ``heat_of_formation_kcal``, ``converged``,
    ``iterations`` and ``trajectory`` — one entry per step with the energy, the largest gradient
    component, and the geometry. A periodic run also returns ``cell_angstrom``, ``pbc`` and
    ``periodicity``; without them a relaxed periodic structure cannot be read back, since the
    coordinates alone do not say what they repeat in.

    ``relax_cell=True`` makes the **lattice** a degree of freedom as well, through the strain
    conjugate to the analytic stress. It is opt-in: the default relaxes the atoms in a fixed cell,
    which can leave an arbitrarily large stress standing — 41.6 GPa in one measurement, reported
    beside ``converged: True``, because the atoms genuinely were at their minimum for that cell.
    Two convergence tests then apply and both must pass, ``gtol`` on the force and ``stress_tol``
    on the largest free stress component; a single mixed norm would let a converged force hide an
    unconverged stress.

    ASE's ``FrechetCellFilter`` with :class:`pm7_rs.ase.PM7` remains a perfectly good route to the
    same thing and is what ASE users already had. ``relax_cell`` is for everyone else: the Rust
    library, both command lines, and Python callers who are not going through ASE.

    ``stability_every=n`` re-runs the SCF stability analysis every ``n``-th step. A geometry step
    changes the orbitals, so a solution that was a minimum at the starting geometry can stop being
    one on the way — and the optimizer would then converge on a stationary point of the wrong
    surface without saying so. Once an instability is found the run switches to ``"follow"`` for
    good, so the remaining steps stay on the lower solution rather than hopping between two.
    ``0`` (the default) never checks, which is what every published number was produced with.
    """
    numbers, positions = _input(numbers, positions)
    return _native.optimize(
        numbers,
        positions,
        float(charge),
        int(multiplicity),
        method,
        reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        field=_field(field),
        dipole_origin=dipole_origin,
        relax_cell=bool(relax_cell),
        gtol=gtol,
        stress_tol=stress_tol,
        max_iter=max_iter,
        stability_every=stability_every,
        **_periodic(cell, pbc, kpoints, kpoint_shift, smearing),
    )


def frequencies(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    cell=None,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    field=None,
    dipole_origin: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    projection: str | None = None,
) -> dict:
    """Harmonic frequencies (cm⁻¹) at the supplied geometry.

    Pass a ``cell`` for the **zone-centre** frequencies of a periodic system; :func:`phonons` and
    :func:`dfpt` give other wavevectors.

    Through v0.2.1 this wrapper took no periodic keywords and its docstring said "molecules only" —
    but the extension has accepted them since 0.2.0, so a caller who passed a periodic `Atoms`
    through :class:`pm7_rs.ase.PM7` got the frequencies of those atoms **as an isolated molecule**,
    with no error and no warning. On a two-atom diamond cell that is a C₂ diatomic: `−772, 654, 654`
    where the crystal's zone centre is `0, 0, 0, 1248, 1248, 1248`. A plausible spectrum, unrelated
    to the system asked about.

    ``field`` applies a uniform external electric field, which changes the frequencies through
    the Hessian. It is a keyword here rather than being dropped: a caller who asks for a field and
    silently gets the field-free spectrum has no way to notice."""
    numbers, positions = _input(numbers, positions)
    return _native.frequencies(
        numbers,
        positions,
        float(charge),
        int(multiplicity),
        method,
        reference,
        field=_field(field),
        dipole_origin=dipole_origin,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        projection=projection,
        **_periodic(cell, pbc, kpoints, kpoint_shift, smearing),
    )


def phonons(
    numbers: Sequence[int],
    positions,
    cell,
    qpoints,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    lo_to_direction=None,
    supercell=(1, 1, 1),
    pbc=None,
    acoustic_sum_rule: bool = True,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    field=None,
    dipole_origin: str | None = None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
) -> dict:
    """Phonon frequencies (cm⁻¹) at each **fractional** q point.

    Force constants come from the analytic Hessian of the ``supercell`` repeat, so the result is
    exact at every q commensurate with it and Fourier-interpolated between. ``supercell=(1,1,1)``
    gives only the zone centre — a dispersion needs at least ``(2,2,2)``, and the force constants
    have to have decayed inside whatever you pick.

    Returns ``frequencies_cm`` (one list per q point, ascending; negative means imaginary),
    ``modes`` and ``cartesian_modes`` (the polarization vectors, below),
    ``acoustic_residual_ev_per_bohr2`` as the honesty check on the sum rule, and the translations
    the force constants were resolved over.

    **The polarization vectors**, new in 0.2.3 — they were computed on every call and discarded,
    which left the frequency the only thing this route could tell you. ``modes[i]`` is a
    ``(real, imag)`` pair for q point ``i``, each a ``3N × 3N`` nested list with **one mode per
    column**, in the same order as ``frequencies_cm[i]``; they are the eigenvectors of the
    mass-weighted dynamical matrix, so each column has unit norm. ``cartesian_modes`` is the same
    set as Cartesian displacements ``m⁻¹ᐟ² e``, renormalized per column — that is the one to
    displace a structure along, and it is the convention the molecular
    :func:`vibrations` uses. Away from the zone centre the vectors are genuinely complex: a phonon
    at ``q`` is ``u_A ∝ e_A e^{iq·R_A}``, and the phase is what separates branches. As numpy::

        m = np.asarray(out["modes"][0][0]) + 1j * np.asarray(out["modes"][0][1])
        softest = m[:, np.argmin(out["frequencies_cm"][0])].reshape(-1, 3)

    ``acoustic_sum_rule`` is **on by default since 0.2.3**. A uniform translation of the whole
    crystal costs no energy, so a non-zero residual is numerical noise, and leaving it in reports a
    non-zero acoustic frequency at the zone centre — on diamond, the difference between ``0.0001``
    and ``-0.0000 cm⁻¹``. It is a genuine two-sided projector, not a shift. The residual is
    reported either way, because it is the honest measure of how well the force constants respect
    translational invariance and projecting hides it; pass ``acoustic_sum_rule=False`` to see it in
    the spectrum as well.

    Pass ``lo_to_direction`` (a Cartesian direction, need not be normalized) to also get
    ``frequencies_cm_lo_to``: the same frequencies with the non-analytic LO–TO term added along
    that direction. It costs a field response on top of the force constants, because the term is
    built from the Born charges and ``eps_inf``. **3-D only**, and there is no default direction —
    the ``q → 0`` limit of the macroscopic field is direction dependent, so a silently chosen one
    would be a wrong answer rather than an approximate one. Entries for a q away from the zone
    centre are ``None``: the term is a zone-centre limit and does not belong there.
    """
    numbers, positions = _input(numbers, positions)
    cell = np.asarray(cell, dtype=float)
    if cell.shape == (3,):
        cell = np.diag(cell)
    q = np.asarray(qpoints, dtype=float).reshape(-1, 3).tolist()
    n = np.asarray(supercell).reshape(-1)
    if n.size != 3:
        raise ValueError("supercell must have three entries, e.g. (2, 2, 2)")
    # The k mesh, smearing and electrostatics mode reach the SCF the force constants
    # differentiate, so they belong here too; they used to be dropped silently.
    periodic = _periodic(None, pbc, kpoints, kpoint_shift, smearing)
    periodic.pop("cell", None)
    return _native.phonons(
        numbers,
        positions,
        cell.tolist(),
        q,
        _direction(lo_to_direction),
        float(charge),
        int(multiplicity),
        method,
        reference,
        supercell=[int(x) for x in n],
        acoustic_sum_rule=bool(acoustic_sum_rule),
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        field=_field(field),
        dipole_origin=dipole_origin,
        pbc_mode=pbc_mode,
        **periodic,
    )


def dfpt(
    numbers: Sequence[int],
    positions,
    cell,
    qpoints,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    lo_to_direction=None,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    field=None,
    dipole_origin: str | None = None,
    dfpt_tolerance: float | None = None,
    dfpt_max_iterations: int | None = None,
    dfpt_mixing: float | None = None,
    long_range: str | None = None,
    keep_response: bool = False,
) -> dict:
    """Phonons at arbitrary **fractional** q by density-functional perturbation theory.

    The alternative to :func:`phonons`, which builds a supercell. DFPT solves the linear response
    directly at each q, so there is no commensurability condition — any q is reachable, not only
    those a supercell happens to fold onto — and the cost is one response solve per q rather than
    one SCF over ``n1*n2*n3`` cells.

    Returns ``frequencies_cm`` (one list per q, ascending; negative means imaginary), ``modes``
    and ``cartesian_modes`` (the polarization vectors, in the same ``(real, imag)`` /
    one-mode-per-column layout :func:`phonons` uses — the two routes compute the same object and
    report it the same way), ``force_constants_ev_per_bohr2`` as a ``(real, imag)`` pair of
    matrices per q, and the solver's ``iterations``/``converged``/``residual``.

    A response that fails to converge raises rather than returning a number: the response is a
    *linear* fixed point, so it does not land near the answer, it diverges.

    ``lo_to_direction`` adds ``frequencies_cm_lo_to`` the way :func:`phonons` does, but here the
    term belongs **only at the zone centre**: away from it the macroscopic field is already inside
    ``Phi(q)`` through the phased Ewald sum, so adding it again would count it twice. Entries for a
    non-zero q are ``None`` for that reason, not because the calculation failed. **3-D only.**
    """
    numbers, positions = _input(numbers, positions)
    cell = np.asarray(cell, dtype=float)
    if cell.shape == (3,):
        cell = np.diag(cell)
    q = np.asarray(qpoints, dtype=float).reshape(-1, 3).tolist()
    periodic = _periodic(None, pbc, kpoints, kpoint_shift, smearing)
    periodic.pop("cell", None)
    return _native.dfpt(
        numbers,
        positions,
        cell.tolist(),
        q,
        _direction(lo_to_direction),
        float(charge),
        int(multiplicity),
        method,
        reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        field=_field(field),
        dipole_origin=dipole_origin,
        dfpt_tolerance=dfpt_tolerance,
        dfpt_max_iterations=dfpt_max_iterations,
        dfpt_mixing=dfpt_mixing,
        long_range=long_range,
        keep_response=keep_response,
        **periodic,
    )


def dielectric_with_extent(
    numbers,
    positions,
    cell,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    slab_thickness: float | None = None,
    wire_cross_section: float | None = None,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    dfpt_tolerance: float | None = None,
    dfpt_max_iterations: int | None = None,
    dfpt_mixing: float | None = None,
) -> dict:
    """``eps_inf`` for a chain or a slab, where the cell has no volume of its own.

    :func:`born_charges` leaves ``dielectric`` as the identity below three dimensions, because
    ``eps`` needs a volume and a slab's cell has an area. Supply the missing extent —
    ``slab_thickness`` in Bohr for a 2-D cell, ``wire_cross_section`` in Bohr² for a 1-D one —
    and exactly one of them. There is no default: a supercell says where the atoms are, not where
    the material stops, so doubling the vacuum must not change ``eps``, and it would if the code
    took the cell height.

    The conversion is **not** a division. ``alpha`` here is the response to the *external* field,
    so for a slab polarized along its normal the depolarizing field is already inside it; the law
    is ``eps = 1 + 4*pi*chi / (1 - 4*pi*N*chi)`` with ``N`` the depolarization factor of the
    assumed body — 0 in plane and along a wire, 1 across a slab, ½ across a wire. The
    three-dimensional case is the ``N = 0`` row of the same table, which is what makes this a
    generalization rather than a second rule.

    Returns ``dielectric``, the raw ``polarizability`` it came from, the ``axis`` used, and:

    * ``sheet_parallel_bohr`` and ``sheet_perpendicular_bohr`` — ``(eps_par - 1) d`` and
      ``(1 - 1/eps_perp) d``, both **free of the extent**. These are what a slab can quote without
      choosing a convention at all; half the first is the Rytova–Keldysh screening length.
    * ``axis_mixing`` — how much of ``alpha`` couples the distinguished axis to its complement,
      relative to the largest diagonal entry. A depolarization factor is a per-principal-axis
      quantity, so this reports how much the conversion assumed. Zero means nothing was lost.
    """
    numbers, positions = _input(numbers, positions)
    cell = np.asarray(cell, dtype=float)
    if cell.shape == (3,):
        cell = np.diag(cell)
    periodic = _periodic(None, pbc, kpoints, kpoint_shift, smearing)
    periodic.pop("cell", None)
    return _native.dielectric_with_extent(
        numbers,
        positions,
        cell.tolist(),
        None if slab_thickness is None else float(slab_thickness),
        None if wire_cross_section is None else float(wire_cross_section),
        float(charge),
        int(multiplicity),
        method,
        reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        dfpt_tolerance=dfpt_tolerance,
        dfpt_max_iterations=dfpt_max_iterations,
        dfpt_mixing=dfpt_mixing,
        **periodic,
    )

def born_charges(
    numbers: Sequence[int],
    positions,
    cell,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    dfpt_tolerance: float | None = None,
    dfpt_max_iterations: int | None = None,
    dfpt_mixing: float | None = None,
    lo_to_direction=None,
    long_range: str | None = None,
    keep_response: bool = False,
) -> dict:
    """Born effective charges, the electronic dielectric tensor, and the LO–TO term.

    ``born_charges[A][a][b]`` is ``d²E / df_a dR_{A,b}`` in units of the elementary charge, with
    **a the field index and b the displacement index** (they are not symmetric in general).

    ``dielectric`` is the clamped-ion ``eps_inf``, and is meaningful only for a 3-D cell — in one
    and two dimensions there is no volume to divide by and it is left as the identity. Use
    ``polarizability``, the raw ``d(mu_a)/d(f_b)`` per cell, which is defined in any dimension.

    **Unrestricted cells are supported** since 0.2.2, on the same terms as the phonons. The field
    reaches the response through the commutator ``[H, r]`` and an unrestricted cell has two
    different ``H``, so it carries a commutator, a band basis and a response density per spin and
    sums each channel's contribution. v0.2.1 refused rather than approximating, and the refusal
    named exactly these three things.

    Note that in PM7 ``born_charges`` is quantitative (LiF gives +1.03 against a measured ~1.04)
    while ``dielectric`` is only qualitative — the minimal valence basis has no polarization
    functions, so it lands two to five times low. See ``docs/fidelity.md``.

    Pass ``lo_to_direction`` (a Cartesian direction, need not be normalized) to also get
    ``lo_to_force_constants_ev_per_bohr2``. It is **3-D only** and there is no default: the q → 0
    limit is direction dependent, so a silently chosen direction would be a wrong answer.
    """
    numbers, positions = _input(numbers, positions)
    cell = np.asarray(cell, dtype=float)
    if cell.shape == (3,):
        cell = np.diag(cell)
    periodic = _periodic(None, pbc, kpoints, kpoint_shift, smearing)
    periodic.pop("cell", None)
    direction = _direction(lo_to_direction)
    return _native.born_charges(
        numbers,
        positions,
        cell.tolist(),
        float(charge),
        int(multiplicity),
        method,
        reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        dfpt_tolerance=dfpt_tolerance,
        dfpt_max_iterations=dfpt_max_iterations,
        dfpt_mixing=dfpt_mixing,
        lo_to_direction=direction,
        **periodic,
    )


def molden(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    basis: str = "sto-6g",
    comment: str | None = None,
    field=None,
    dipole_origin: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
) -> str:
    """The converged wavefunction as a Molden-format **string**.

    Returns the text rather than writing a file, so the caller decides where it goes.

    ``basis`` is ``"sto-6g"`` (default) for a Gaussian rendering basis every viewer reads, or
    ``"sto"`` for the exact single-zeta Slater basis PM7 actually uses — which is faithful but is
    refused for a molecule carrying d functions, because Molden's ``[STO]`` primitive is a single
    Cartesian monomial and two of the five d functions are not monomials.

    The file carries its own caveat: NDDO assumes an orthonormal AO basis, so the coefficients are
    in an implicitly orthogonalized basis while the listed functions are the raw non-orthogonal
    ones. Shapes, nodes and symmetry are faithful; bonding-region amplitudes are approximate.
    """
    numbers, positions = _input(numbers, positions)
    return _native.molden(
        numbers,
        positions,
        float(charge),
        int(multiplicity),
        method,
        reference,
        basis=basis,
        comment=comment,
        field=_field(field),
        dipole_origin=dipole_origin,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
    )


def band_structure(
    numbers: Sequence[int],
    positions,
    cell,
    kpath,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    field=None,
    dipole_origin: str | None = None,
) -> dict:
    """Band energies (eV) at each **fractional** k point along ``kpath``.

    This is not an SCF along the path. A band path runs down high-symmetry lines, which is the
    wrong set of points to build a density from — it would weight those lines as though they were
    the whole Brillouin zone. The density and the Fermi level come from the ``kpoints`` sampling
    mesh, and the path only asks the converged Hamiltonian what its eigenvalues are elsewhere.

    Returns ``energies_ev`` (one ascending list per k point), ``fermi_ev``, and for an
    unrestricted run ``energies_beta_ev``.
    """
    numbers, positions = _input(numbers, positions)
    cell = np.asarray(cell, dtype=float)
    if cell.shape == (3,):
        cell = np.diag(cell)
    path = np.asarray(kpath, dtype=float).reshape(-1, 3).tolist()
    periodic = _periodic(cell, pbc, kpoints, kpoint_shift, smearing)
    return _native.band_structure(
        numbers,
        positions,
        cell.tolist(),
        path,
        float(charge),
        int(multiplicity),
        method,
        reference,
        pbc=periodic.get("pbc"),
        kpoints=periodic.get("kpoints"),
        kpoint_shift=periodic.get("kpoint_shift"),
        smearing=periodic.get("smearing"),
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        field=_field(field),
        dipole_origin=dipole_origin,
    )


def divide_and_conquer(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    buffer: float = 8.0,
    core_size: int = 12,
    cell=None,
    pbc=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    field=None,
    dipole_origin: str | None = None,
) -> dict:
    """Linear-scaling single point: energy, forces, and (with a ``cell``) stress.

    ``buffer`` is the accuracy knob, in **Ångström**: everything outside a subsystem's buffer
    reaches it only as a monopole, so widening it walks the answer onto the exact SCF at a cost
    that grows with the buffer cubed.

    This is the wrong tool below roughly 300 atoms — the ordinary :func:`single_point` is both
    exact and faster there. See ``docs/divide_and_conquer.md`` for the measured crossover, the
    scaling, and the buffer/accuracy table.
    """
    numbers, positions = _input(numbers, positions)
    periodic = _periodic(cell, pbc, None, None, None)
    return _native.divide_and_conquer(
        numbers,
        positions,
        float(charge),
        int(multiplicity),
        method,
        reference,
        buffer=float(buffer),
        core_size=int(core_size),
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        field=_field(field),
        dipole_origin=dipole_origin,
        **periodic,
    )


def hessian(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    cell=None,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    field=None,
    dipole_origin: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
) -> dict:
    """Analytic Cartesian **Hessian** (3N×3N) as ``hessian_hartree_per_bohr2`` (atomic
    units) and ``hessian_ev_per_angstrom2``; positions in Angstrom.

    Molecules **and** periodic cells. Pass a ``cell`` and the zone-centre force constants come
    back: at the Γ point from the periodic analytic Hessian, and on a k mesh from the perturbation
    solver at ``q = 0``, which is the same calculation reached the way the coupling `k ↔ k + q`
    requires. For force constants at other wavevectors use :func:`dfpt` or :func:`phonons`.

    Through v0.2.1 this wrapper accepted none of the periodic keywords and its docstring said a
    periodic system raises — both stale: the extension had dispatched to the Γ periodic path since
    0.2.0, and only a k mesh was refused.
    """
    numbers, positions = _input(numbers, positions)
    periodic = _periodic(cell, pbc, kpoints, kpoint_shift, smearing)
    return _native.hessian(
        numbers,
        positions,
        float(charge),
        int(multiplicity),
        method,
        reference,
        field=_field(field),
        dipole_origin=dipole_origin,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        **periodic,
    )


def scf_stability(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    cell=None,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    field=None,
    dipole_origin: str | None = None,
) -> dict:
    """Is the converged SCF solution a **minimum**, or only a stationary point?

    An SCF iteration converges to a stationary point of the energy, and a saddle converges as
    cleanly as a minimum — same residual, same ``converged``. This measures the curvature it
    stopped at.

    Returns ``lowest_ev`` for the **singlet** (spin-preserving) channel and ``lowest_triplet_ev``
    for the **spin-breaking** one, both in eV; negative means a saddle, and ``unstable`` is true
    when either is. The two are different questions and a closed shell can pass the first and fail
    the second — stretched H₂ is a minimum among closed-shell solutions and 121 kcal/mol above the
    right answer, which only the triplet channel sees. ``lowest_triplet_ev`` is ``None`` when the
    solution is already unrestricted, where no spin symmetry is left to break.

    ``analysed`` is ``False`` when the question does not apply: a periodic cell, an unconverged
    solve, or nothing to rotate into. That is reported rather than raised, because asking should
    always be allowed.

    It measures **the solution the options it was given produce**, which by default is the plain
    SCF solution. Passing ``stability="follow"`` here is neither an error nor a no-op: the SCF
    escapes first, so what comes back describes the *escaped* solution and correctly reports
    ``unstable: False``. To act on an instability, that argument belongs on the entry point whose
    answer you actually want — :func:`single_point`, :func:`optimize` and the rest all take it.

    ``spin_squared`` comes back too, so a followed run says what the escape cost in spin purity.
    """
    numbers, positions = _input(numbers, positions)
    return _native.scf_stability(
        numbers,
        positions,
        float(charge),
        int(multiplicity),
        method,
        reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        field=_field(field),
        dipole_origin=dipole_origin,
        **_periodic(cell, pbc, kpoints, kpoint_shift, smearing),
    )


def orbitals(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    cell=None,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    field=None,
    dipole_origin: str | None = None,
) -> dict:
    """Orbital energies and coefficients for both spin channels; positions in Angstrom.

    Returns ``mo_energies_ev`` / ``mo_energies_hartree``, ``mo_coefficients`` (``nao x nmo``,
    column ``k`` is MO ``k``), ``occupations``, ``homo_ev`` / ``lumo_ev`` / ``gap_ev``, and the
    ``ao_labels`` / ``ao_atom_index`` / ``ao_shell`` needed to read the coefficient matrix without
    rebuilding the basis. Unrestricted runs add the ``_beta`` twins.

    ``orbital_source`` says which Hamiltonian the orbitals belong to: ``"molecular"``,
    ``"gamma"``, or ``"kmesh-gamma"``. In the last case they are the **Gamma** eigenpair of the
    converged k-mesh Hamiltonian, so the implied gap is a Gamma gap; ``fermi_ev`` and
    :func:`band_structure` are the zone-wide answers.

    A periodic run also returns ``fermi_ev``, ``entropy_ev`` and ``n_kpoints``. That matters for
    reading ``occupations``, which is an aufbau count over the Gamma states and so is the mesh's
    filling only when a gap straddles the Fermi level. A non-zero ``entropy_ev`` says it does not.
    Through 0.2.2 this function accepted ``smearing`` and a k mesh and returned neither key, so a
    smeared metal came back with occupations nothing in the result contradicted.

    Separate from :func:`single_point` because the coefficient matrix is ``nao x nao``: nobody
    running molecular dynamics wants to marshal one on every step.
    """
    numbers, positions = _input(numbers, positions)
    return _native.orbitals(
        numbers,
        positions,
        float(charge),
        int(multiplicity),
        method,
        reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        field=_field(field),
        dipole_origin=dipole_origin,
        **_periodic(cell, pbc, kpoints, kpoint_shift, smearing),
    )


def vibrations(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    cell=None,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    field=None,
    dipole_origin: str | None = None,
    hessian: bool = True,
    frequencies: bool = True,
    modes: bool = False,
    ir: bool = False,
    orbital_response: bool = False,
    step: float = 1.0e-3,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    projection: str | None = None,
) -> dict:
    """Everything one Hessian produces: frequencies, modes, IR intensities, orbital response.

    Molecules **and** periodic cells: with a `cell` the Hessian, frequencies and modes are the
    zone-centre ones. `ir=True` still needs a molecule -- the dipole operator a periodic cell
    admits is not the one IR intensities are built from -- and says so.

    Through v0.2.1 this took no periodic keywords at all, so a periodic system silently got the
    frequencies of its atoms **as an isolated molecule**. On a two-atom diamond cell that is a C2
    diatomic: `-772, 654, 654` where the crystal's zone centre is `0, 0, 0, 1248, 1248, 1248`.

    One entry point rather than several because every one of these comes from the
    **same** CPHF solve -- asking for frequencies and then a spectrum through separate calls pays
    for that solve twice. :func:`frequencies` and :func:`hessian` are unchanged and remain the
    cheapest route when they are all you want.

    ``ir=True`` adds ``ir_intensities_km_per_mol``, the dense ``dipole_derivatives_e``
    (``3 x 3N``, atomic units) and its ``_debye_per_angstrom`` twin, the mode-projected
    ``mode_dipole_derivatives``, and MOPAC's ``mopac_dipt`` / ``mopac_trdip`` for comparison
    against a ``FORCE LARGE`` run.

    ``orbital_response=True`` returns the CPHF response as a **flat** list plus
    ``orbital_response_shape`` (``[3N, n_vir, n_occ]``); use :class:`Vibrations` to get it
    reshaped. It is off by default because it is large to hand back, not because it is slow --
    the Hessian solves for it either way.
    """
    numbers, positions = _input(numbers, positions)
    return _native.vibrations(
        numbers,
        positions,
        float(charge),
        int(multiplicity),
        method,
        reference,
        field=_field(field),
        dipole_origin=dipole_origin,
        pbc_mode=pbc_mode,
        hessian=bool(hessian),
        frequencies=bool(frequencies),
        modes=bool(modes),
        ir=bool(ir),
        orbital_response=bool(orbital_response),
        step=float(step),
        **_periodic(cell, pbc, kpoints, kpoint_shift, smearing),
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        projection=projection,
    )


class Vibrations:
    """One Hessian, many derived quantities, computed once and cached.

    Constructing this is **free**: nothing runs until the first attribute access. That access
    performs exactly one SCF and one CPHF solve, producing everything the constructor asked for;
    every later access is a cache read.

    ``ir`` and ``modes`` default to on because, once the Hessian exists, they cost one
    dipole-operator build and 3N traces. ``orbital_response`` defaults to off because its payload
    is ``3N x n_vir x n_occ`` floats -- about 98 MB for a hundred-atom molecule. Asking for
    something the constructor did not request raises, naming the flag to set, so the cost of a
    call can never quietly double.

        v = Vibrations(numbers, positions)
        v.frequencies_cm          # runs the calculation
        v.ir_intensities_km_per_mol   # cache read
    """

    def __init__(
        self,
        numbers: Sequence[int],
        positions,
        charge: float = 0.0,
        multiplicity: int = 1,
        method: str = "pm7",
        reference: str = "auto",
        *,
        ir: bool = True,
        modes: bool = True,
        orbital_response: bool = False,
        **kwargs,
    ) -> None:
        self._call = dict(
            numbers=numbers,
            positions=positions,
            charge=charge,
            multiplicity=multiplicity,
            method=method,
            reference=reference,
            ir=ir,
            modes=modes,
            orbital_response=orbital_response,
            **kwargs,
        )
        self._requested = {"ir": ir, "modes": modes, "orbital_response": orbital_response}
        self._result: dict | None = None

    @property
    def result(self) -> dict:
        """The raw dictionary, computing it on first access."""
        if self._result is None:
            self._result = vibrations(**self._call)
        return self._result

    def _get(self, key: str, flag: str | None = None):
        value = self.result.get(key)
        if value is None and flag is not None and not self._requested.get(flag, False):
            raise AttributeError(
                f"{key!r} was not computed: construct Vibrations(..., {flag}=True) to ask for it"
            )
        return value

    @property
    def hessian_ev_per_angstrom2(self):
        return np.asarray(self._get("hessian_ev_per_angstrom2"), dtype=float)

    @property
    def frequencies_cm(self):
        return np.asarray(self._get("frequencies_cm"), dtype=float)

    @property
    def modes(self):
        return np.asarray(self._get("modes", "modes"), dtype=float)

    @property
    def cartesian_modes(self):
        return np.asarray(self._get("cartesian_modes", "modes"), dtype=float)

    @property
    def ir_intensities_km_per_mol(self):
        return np.asarray(self._get("ir_intensities_km_per_mol", "ir"), dtype=float)

    @property
    def dipole_derivatives(self):
        """The dense ``3 x 3N`` tensor in atomic units (e)."""
        return np.asarray(self._get("dipole_derivatives_e", "ir"), dtype=float)

    @property
    def mode_dipole_derivatives(self):
        return np.asarray(self._get("mode_dipole_derivatives", "ir"), dtype=float)

    @property
    def orbital_response(self):
        """The CPHF response, reshaped to ``(3N, n_vir, n_occ)``."""
        flat = self._get("orbital_response", "orbital_response")
        shape = self.result.get("orbital_response_shape")
        if flat is None or shape is None:
            raise AttributeError(
                "the orbital response was not computed: construct "
                "Vibrations(..., orbital_response=True) to ask for it"
            )
        return np.asarray(flat, dtype=float).reshape(tuple(shape))

    @property
    def orbital_response_beta(self):
        flat = self.result.get("orbital_response_beta")
        shape = self.result.get("orbital_response_beta_shape")
        if flat is None or shape is None:
            return None
        return np.asarray(flat, dtype=float).reshape(tuple(shape))


def polarizability(
    numbers,
    positions,
    cell,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    dfpt_tolerance: float | None = None,
    dfpt_max_iterations: int | None = None,
    dfpt_mixing: float | None = None,
    long_range: str | None = None,
) -> dict:
    """The cell's electronic polarizability ``alpha_ab = d(mu_a)/d(f_b)``, in Bohr^3.

    The same tensor :func:`born_charges` reports, without computing the Born charges to get it.
    Separate because ``alpha`` is defined in **every** dimensionality where ``eps_inf`` is not: a
    chain and a slab have a polarizability, and neither has a dielectric constant until someone
    says where the material stops (:func:`dielectric_with_extent`).

    In the MOPAC ``FIELD=`` sign convention, so it carries the opposite sign to the physical
    polarizability.

    ``long_range`` is ``"auto"`` (default), ``"require"`` or ``"off"``. ``"off"`` drops the
    long-range monopole term from all three places it enters, which is how its effect is measured
    rather than argued.
    """
    periodic = _periodic(cell, pbc, kpoints, kpoint_shift, smearing)
    return _native.polarizability(
        numbers,
        positions,
        periodic.pop("cell"),
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        reference=reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        dfpt_tolerance=dfpt_tolerance,
        dfpt_max_iterations=dfpt_max_iterations,
        dfpt_mixing=dfpt_mixing,
        long_range=long_range,
        **periodic,
    )


def dielectric_origin_sensitivity(
    numbers,
    positions,
    cell,
    offset,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    dfpt_tolerance: float | None = None,
    dfpt_max_iterations: int | None = None,
    dfpt_mixing: float | None = None,
) -> float:
    """How much ``alpha`` moves when every atom is displaced by ``offset`` (Bohr).

    The position operator a field perturbation is built on is not a well-defined periodic operator.
    That the *response* is nevertheless well defined is an argument; this measures it. A value near
    machine precision says the argument holds for this system; a large one says the polarizability
    being reported is a statement about where the origin was put.
    """
    periodic = _periodic(cell, pbc, kpoints, kpoint_shift, smearing)
    return _native.dielectric_origin_sensitivity(
        numbers,
        positions,
        periodic.pop("cell"),
        offset,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        reference=reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        dfpt_tolerance=dfpt_tolerance,
        dfpt_max_iterations=dfpt_max_iterations,
        dfpt_mixing=dfpt_mixing,
        **periodic,
    )


def static_dielectric(
    numbers,
    positions,
    cell,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    dfpt_tolerance: float | None = None,
    dfpt_max_iterations: int | None = None,
    dfpt_mixing: float | None = None,
) -> dict:
    """``eps^0``: the clamped-ion ``eps^inf`` plus what the ions contribute.

    ``eps^0 = eps^inf + (4 pi / Omega) sum_m (Z* e_m)(Z* e_m) / omega_m^2`` over the optical modes.
    This is the quantity a measured dielectric constant is usually compared against; ``eps^inf``
    alone is the high-frequency limit, and for an ionic crystal the two differ by a lot.

    **Check ``soft_optical_modes``.** Non-zero means the geometry is not a minimum, and the ionic
    term is then missing whatever those modes would have added -- which for a soft mode is most of
    it, since ``1/omega^2`` weights them hardest.

    ``skipped_modes`` is always 3, and is the acoustic branch. Since 0.2.3 those are identified by
    their overlap with the mass-weighted uniform translations rather than by falling below a
    frequency floor, so the count is a property of the subspace and not of the spectrum. The old
    ``soft_mode_floor`` key is gone with the constant behind it.

    3-D only: both halves carry a ``4 pi / Omega`` that needs a volume.
    """
    periodic = _periodic(cell, pbc, kpoints, kpoint_shift, smearing)
    return _native.static_dielectric(
        numbers,
        positions,
        periodic.pop("cell"),
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        reference=reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        dfpt_tolerance=dfpt_tolerance,
        dfpt_max_iterations=dfpt_max_iterations,
        dfpt_mixing=dfpt_mixing,
        **periodic,
    )


def berry_polarization(
    numbers,
    positions,
    cell,
    strings: int = 16,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    pbc=None,
    kpoints=None,
    kpoint_shift=None,
    smearing=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
) -> dict:
    """Berry-phase electronic polarization (King-Smith and Vanderbilt), in ``e/Bohr^2``.

    The dipole per cell of a crystal is not a function of the density -- moving the cell boundary
    moves charge across it -- so ``sum q r`` is not the polarization and this is. What it gives is
    defined **modulo** the returned ``quantum``: a different branch of the logarithm puts the
    electrons in a different cell, which is an equally valid choice.

    Compare two polarizations by reducing their difference onto the nearest branch, never by
    subtracting ``polarization`` directly. A finite displacement commonly crosses a branch and the
    raw difference is then wrong by exactly one quantum -- a number that looks like a catastrophic
    error rather than a bookkeeping choice.

    ``strings`` is the convergence parameter and the answer has to stop moving with it.
    """
    periodic = _periodic(cell, pbc, kpoints, kpoint_shift, smearing)
    return _native.berry_polarization(
        numbers,
        positions,
        periodic.pop("cell"),
        strings=strings,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        reference=reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        **periodic,
    )


def finite_field(
    numbers,
    positions,
    cell,
    divisions,
    field,
    charge: float = 0.0,
    multiplicity: int = 1,
    method: str = "pm7",
    reference: str = "auto",
    *,
    pbc=None,
    pbc_mode: str | None = None,
    scf_tolerance: float | None = None,
    max_scf: int | None = None,
    cphf_max_iterations: int | None = None,
    stability: str | None = None,
    exchange_cutoff: tuple[float, float] | None = None,
    use_diis: bool | None = None,
    field_tolerance: float | None = None,
    field_max_iterations: int | None = None,
    field_mixing: float | None = None,
) -> dict:
    """A finite electric field applied **along** a periodic direction.

    A field orthogonal to every lattice vector is an ordinary calculation: pass ``field=`` to the
    other entry points. Along a periodic direction the ``E.R`` potential is unbounded and the
    ground state does not exist, so this minimizes the Nunes-Gonze electric enthalpy
    ``E - Omega E.P`` with ``P`` the Berry-phase polarization instead.

    ``divisions`` is the k mesh **and** the string length, so it is the convergence parameter for
    the polarization as well as for the Brillouin-zone integral; an axis the field touches needs at
    least three points. ``resolved`` says which axes had enough, because an axis that cannot carry
    a phase is reported unresolved rather than as zero.

    Restricted closed shells only, matching :func:`berry_polarization`.
    """
    periodic = _periodic(cell, pbc, None, None, None)
    return _native.finite_field(
        numbers,
        positions,
        periodic.pop("cell"),
        divisions,
        field,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        reference=reference,
        pbc_mode=pbc_mode,
        scf_tolerance=scf_tolerance,
        max_scf=max_scf,
        cphf_max_iterations=cphf_max_iterations,
        stability=stability,
        exchange_cutoff=exchange_cutoff,
        use_diis=use_diis,
        field_tolerance=field_tolerance,
        field_max_iterations=field_max_iterations,
        field_mixing=field_mixing,
        **periodic,
    )
