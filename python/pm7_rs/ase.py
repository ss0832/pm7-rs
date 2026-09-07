# SPDX-License-Identifier: GPL-3.0-or-later
"""ASE calculator for PM7-family methods (eV / Angstrom conventions).

Handles molecules and periodic systems through the same class: the calculator reads
``atoms.get_cell()`` and ``atoms.get_pbc()`` and switches automatically, so a
``FrechetCellFilter`` relaxation or an ``NPT`` run works with no extra setup beyond giving the
``Atoms`` a cell.
"""

from __future__ import annotations

import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install `pm7-rs-python[ase]` to use pm7_rs.ase.PM7.") from exc

from . import native
from .constants import DEBYE_TO_E_ANGSTROM


def _hashable(value):
    """A cache-key form of an argument that may be a list, an array, or ``None``.

    Floats go in by **bit pattern**, so a q path that differs in the last bit misses rather than
    silently reusing a neighbour's answer. `None` stays `None` so "no LO–TO direction" is its own
    key rather than colliding with the zero vector.
    """
    if value is None:
        return None
    array = np.ascontiguousarray(value, dtype=float)
    return (array.shape, array.tobytes())


class PM7(Calculator):
    """PM7-family calculator.

    Parameters
    ----------
    charge : int
        Formal charge (e) of the molecule, or **of the unit cell** for a periodic system.
        Charged periodic cells are supported; the neutralizing background is reported in
        ``calc.results["background_ev"]`` because it makes the absolute energy
        convention-dependent.
    multiplicity : int
        Spin multiplicity 2S+1 (1 = singlet, 2 = doublet, …).
    method : str
        ``"pm7"``, ``"pm7-ts"``, ``"pm7-hh"``, ``"pm7-"`` (PM7-minus), …
    reference : str
        Spin treatment independent of ``multiplicity``: ``"auto"`` (RHF/UHF by shell),
        ``"rhf"`` (force RHF), or ``"uhf"`` (force UHF). Unrestricted is supported periodically
        on the same terms as restricted.
    kpts : tuple[int, int, int] | None
        Monkhorst–Pack divisions for a periodic system. ``None`` (default) is the Γ point.
        For a small primitive cell a k mesh matters: Γ-only sampling cannot represent the
        exchange properly (see ``docs/pbc.md``).
    smearing : str | tuple | dict | None
        Occupation broadening for metals: ``"fermi"``, ``("gauss", 0.1)``, or
        ``{"kind": "mp", "width_ev": 0.2, "order": 1}``.
    pbc_mode : str | None
        ``"ewald"`` (default) or ``"mopac"``.
    scf_tolerance : float | None
        SCF density-convergence threshold. Molecular dynamics needs the SCF to converge at
        *every* step, and a trajectory usually tolerates a looser threshold than a single point
        wants — 1e-6 is normally plenty for forces.
    max_scf : int | None
        Maximum SCF iterations.
    cphf_max_iterations : int | None
        Operator applications one CPHF orbital-response solve may spend (default 100).
        Every analytic Hessian goes through one, and how many it takes is set by the
        frontier gap: a small-gap cell can need several times the default and refuses
        rather than returning an unconverged response.
    stability : str | None
        ``"off"`` (default), ``"check"`` or ``"follow"``. An SCF converges on a **stationary**
        point, not necessarily a minimum, and a saddle reports the same ``converged`` in the same
        words. ``"check"`` fills in :meth:`get_scf_stability`; ``"follow"`` rotates along the
        unstable direction, re-converges, and keeps whichever solution is lower — so it can never
        make an answer worse. Escaping through the triplet channel returns an **unrestricted**
        solution where you asked for a restricted one; it warns on stderr when it does, and
        ``results["spin_squared"]`` reports the spin contamination that came with it.

    Energies are eV, forces eV/Å, stress eV/Å³ in Voigt order, the Hessian eV/Å², charges e,
    dipole e·Å.

    ``results`` also carries ``heat_of_formation_kcal``, ``unrestricted`` and ``spin_squared``
    (``⟨S²⟩``, MOPAC's ``(S**2)``; ``None`` for a restricted or a k-mesh run) after any calculation
    that runs a single point.

    ``energy`` and ``free_energy``
    ------------------------------
    ``energy`` is the internal electronic energy at the occupations the SCF converged to.
    ``free_energy`` — what ASE hands back for ``get_potential_energy(force_consistent=True)`` — is
    the **Mermin electronic** free energy ``E − TS``, with ``T`` the smearing width and ``S`` the
    occupation entropy reported separately as ``results["entropy_ev"]``. The distinction only bites
    with ``smearing`` set: there the forces are ``∂F/∂R``, not ``∂E/∂R``, so an optimizer or an MD
    run using the force-consistent energy needs the free one. Without smearing the entropy is
    exactly zero and the two are the same number, bit for bit.

    Neither is a **thermochemical** free energy. No vibrational partition function enters either,
    so neither is the Gibbs free energy a normal-mode analysis would give: ``get_frequencies``
    returns harmonic frequencies and nothing is derived from them (see ``docs/scope.md``, "No
    thermochemistry"). ``free_energy`` here is electronic, at the smearing width, and is about the
    Brillouin-zone occupation of a metal rather than about molecular vibrations.
    """

    #: Properties that all come out of one Hessian, and are therefore computed together.
    #:
    #: Only ``hessian`` is an ASE-standard name, so the rest are **not** in
    #: ``implemented_properties``: code that iterates that list — and ASE itself does — must keep
    #: seeing only names it understands.
    #:
    #: That exclusion is also why they cannot be reached through ``get_property``, which raises
    #: ``PropertyNotImplementedError`` for any name outside the list. An earlier version routed
    #: them through it anyway, so every one of ``get_frequencies``, ``get_normal_modes``,
    #: ``get_ir_intensities`` and ``get_dipole_derivatives`` raised on the first call — and no test
    #: called them, so the docstrings went on describing a cache that never ran. They use
    #: :meth:`_vibrational` now, which is the same one-solve-per-geometry cache reached directly;
    #: ``hessian`` additionally lands in ``results`` so ASE's own machinery keeps working.
    _VIBRATIONAL = frozenset(
        {"hessian", "frequencies", "normal_modes", "ir_intensities", "dipole_derivatives"}
    )

    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "charges",
        "dipole",
        "hessian",
    ]

    def __init__(
        self,
        charge: int = 0,
        multiplicity: int = 1,
        method: str = "pm7",
        reference: str = "auto",
        kpts=None,
        smearing=None,
        pbc_mode: str | None = None,
        scf_tolerance: float | None = None,
        max_scf: int | None = None,
        cphf_max_iterations: int | None = None,
        stability: str | None = None,
        field=None,
        dipole_origin: str | None = None,
        **kwargs,
    ):
        # ASE's base Calculator absorbs unknown keywords into `self.parameters`, so a stale
        # `variant=` would be silently ignored and quietly downgrade the run to plain PM7.
        # Fail loudly instead: this was renamed in 0.1.2 and the accepted values are unchanged.
        if "variant" in kwargs:
            raise TypeError(
                "PM7(variant=...) was renamed to PM7(method=...) in pm7-rs 0.1.2; "
                f"use PM7(method={kwargs['variant']!r}) instead."
            )
        # The 0.2.0 changelog advertised `dandc=` on this constructor and it did not exist; because
        # `**kwargs` swallows unknown keywords it did not fail either, so asking for the linear
        # scaling solver quietly got the exact one. v0.2.1 turned that into an error, and 0.2.2
        # implements it: `dandc=True` for the defaults, or a dict of the solver's own knobs.
        dandc = kwargs.pop("dandc", None)
        if dandc is True:
            dandc = {}
        elif dandc is not None and not isinstance(dandc, dict):
            raise TypeError(
                "PM7(dandc=...) takes True for the defaults or a dict of solver options "
                f"(buffer, core_size), not {type(dandc).__name__}."
            )
        if dandc is not None:
            unknown = set(dandc) - {"buffer", "core_size"}
            if unknown:
                raise TypeError(
                    f"PM7(dandc=...) got unknown options {sorted(unknown)}; it takes "
                    "'buffer' (Angstrom) and 'core_size' (atoms)."
                )
        super().__init__(**kwargs)
        self.dandc = dandc
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.method = method
        self.reference = reference
        self.kpts = None if kpts is None else tuple(int(k) for k in kpts)
        self.smearing = smearing
        self.pbc_mode = pbc_mode
        self.scf_tolerance = scf_tolerance
        self.max_scf = max_scf
        self.cphf_max_iterations = cphf_max_iterations
        self.stability = stability
        self.field = field
        self.dipole_origin = dipole_origin
        # Caches for the results ASE's own `results` dict cannot hold: the vibrational group,
        # whose names are deliberately outside `implemented_properties`, and the periodic methods,
        # which take arguments `get_property` has no way to key on.
        self._vib_cache: tuple | None = None
        self._extra_cache: dict = {}
        # Set by `set_atoms`, which ASE calls on `atoms.calc = calc`.
        self._bound_atoms = None

    # -- internals ---------------------------------------------------------------------

    def set_atoms(self, atoms):
        """ASE calls this from ``atoms.calc = calc``; that is the only reason it exists.

        Kept in its **own** attribute rather than assigned to ``self.atoms``. ASE's
        ``Calculator.calculate`` deliberately stores a *copy* there, because ``check_state``
        compares it against the live object to decide whether the geometry moved — parking a live
        reference in ``self.atoms`` would make every comparison a comparison of an object with
        itself, and the standard properties would never invalidate. The methods below key on the
        geometry's bytes each time they are called, so a live reference is exactly right for them.
        """
        self._bound_atoms = atoms

    def _resolve(self, atoms):
        """The `Atoms` to work on, with a message worth reading when there is none.

        `self.atoms` is only populated once ASE has run a property through `calculate`, and
        `_bound_atoms` only once the calculator has been attached to an `Atoms`. A method called
        on a calculator that has never seen either used to die on
        `'NoneType' object has no attribute 'get_atomic_numbers'`.
        """
        if atoms is None:
            atoms = self.atoms if self.atoms is not None else self._bound_atoms
        if atoms is None:
            raise ValueError(
                "this PM7 calculator has no Atoms yet: pass one explicitly, e.g. "
                "calc.get_orbitals(atoms), or evaluate a property first "
                "(atoms.get_potential_energy()) so the calculator learns its system."
            )
        return atoms

    @staticmethod
    def _geometry_key(atoms):
        """An exact key for one geometry: species, positions, cell and periodicity, as bytes.

        Bytes rather than values so the comparison is bit-for-bit — a geometry that differs in the
        last bit misses, which is the conservative direction for a cache.
        """
        return (
            atoms.get_atomic_numbers().tobytes(),
            np.ascontiguousarray(atoms.get_positions(), dtype=float).tobytes(),
            np.ascontiguousarray(atoms.cell[:], dtype=float).tobytes(),
            tuple(bool(p) for p in atoms.pbc),
        )

    def _cached(self, tag, atoms, build):
        """Memoize `build()` under `tag` for this exact geometry.

        One slot per tag, cleared whenever the geometry changes — the same invalidation rule ASE
        applies to `results`, applied to the things `results` cannot hold.
        """
        key = self._geometry_key(atoms)
        hit = self._extra_cache.get(tag)
        if hit is not None and hit[0] == key:
            return hit[1]
        value = build()
        self._extra_cache[tag] = (key, value)
        return value

    def _vibrational(self, atoms=None):
        """The whole vibrational group from **one** SCF and CPHF solve, cached per geometry.

        Frequencies, normal modes, IR intensities, dipole derivatives and the Hessian all fall out
        of the same solve, so asking for several must not pay for several.

        For a **periodic** system this is the zone centre, and IR is left out: the dipole a
        periodic cell admits is not the operator infrared intensities are built from. Through
        v0.2.1 no periodic keyword reached here at all, so `PM7(...).get_frequencies()` on a
        periodic `Atoms` returned the frequencies of those atoms **as an isolated molecule** — for
        a two-atom diamond cell, a C₂ diatomic at `−772, 654, 654` where the crystal's zone centre
        is `0, 0, 0, 1248, 1248, 1248`. No error, no warning, and a plausible-looking spectrum.
        """
        atoms = self._resolve(atoms)
        key = self._geometry_key(atoms)
        if self._vib_cache is not None and self._vib_cache[0] == key:
            return self._vib_cache[1]
        periodic = self._periodic_kwargs(atoms)
        out = native.vibrations(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            self.charge,
            self.multiplicity,
            self.method,
            self.reference,
            field=self.field,
            dipole_origin=self.dipole_origin,
            hessian=True,
            frequencies=True,
            modes=True,
            ir=not periodic,
            scf_tolerance=self.scf_tolerance,
            max_scf=self.max_scf,
            cphf_max_iterations=self.cphf_max_iterations,
            stability=self.stability,
            **periodic,
        )
        self._vib_cache = (key, out)
        return out

    def _infrared(self, atoms):
        """The vibrational solve, with the infrared half present.

        A periodic cell has no infrared section: the dipole it admits is defined only along
        non-periodic directions and is not the operator intensities are built from. Say so here
        rather than letting a `KeyError` out of the results dictionary.
        """
        if self._periodic_kwargs(self._resolve(atoms)):
            raise NotImplementedError(
                "infrared intensities are molecular: the dipole operator they are built from is "
                "not defined along a periodic direction. Use get_frequencies() or "
                "get_phonons() for a periodic cell, or drop the cell for the molecule."
            )
        return self._vibrational(atoms)

    def _periodic_kwargs(self, atoms):
        """The periodic keywords for this ``Atoms``, or an empty dict for a molecule.

        Any combination of periodic directions works. Through 0.2.2 this raised for anything but
        a leading pattern -- ``pbc=(True, False, True)``, which is what an ``Atoms`` built as a
        slab along *y* has, was rejected rather than run, and reordering the cell was left to the
        user. The native layer now rotates the lattice vectors itself; ``pbc`` is passed straight
        through and any per-lattice-vector keyword (``kpoints``) rotates with it.
        """
        pbc = np.asarray(atoms.get_pbc(), dtype=bool)
        if not pbc.any():
            return {}
        cell = np.asarray(atoms.get_cell(), dtype=float)
        if not np.isfinite(cell).all() or np.allclose(cell[pbc], 0.0):
            raise ValueError("periodic Atoms need a non-degenerate cell")
        kwargs = {"cell": cell, "pbc": pbc}
        if self.kpts is not None:
            kwargs["kpoints"] = self.kpts
        if self.smearing is not None:
            kwargs["smearing"] = self.smearing
        return kwargs

    def _common(self, atoms):
        return dict(
            charge=self.charge,
            multiplicity=self.multiplicity,
            method=self.method,
            reference=self.reference,
            pbc_mode=self.pbc_mode,
            scf_tolerance=self.scf_tolerance,
            max_scf=self.max_scf,
            cphf_max_iterations=self.cphf_max_iterations,
            stability=self.stability,
            field=self.field,
            dipole_origin=self.dipole_origin,
            **self._periodic_kwargs(atoms),
        )

    def _common_without_cell(self, atoms):
        """[`_common`] minus `cell`, for the entry points that take it positionally.

        `_common` already folds in `_periodic_kwargs`, so spreading that a second time alongside
        it passes `pbc` and `kpoints` twice and Python rejects the call outright. Every periodic
        method here did exactly that when it was written, which made `get_phonons`, `get_dfpt` and
        `get_born_charges` raise `TypeError` on any `Atoms` with `pbc` set — the tests exercised
        `native.*` directly and never went through the calculator.
        """
        return {k: v for k, v in self._common(atoms).items() if k != "cell"}

    def _store_periodic(self, out):
        for key in (
            "fermi_ev",
            "entropy_ev",
            "n_kpoints",
            "ewald_ev",
            "background_ev",
            "makov_payne_ev",
            "pressure_gpa",
            "volume_angstrom3",
        ):
            if key in out:
                self.results[key] = out[key]
        if "stress_voigt" in out:
            # ASE wants the Voigt 6-vector [xx, yy, zz, yz, xz, xy] in eV/A^3, positive under
            # tension. Getting the order or the sign wrong here does not fail loudly — it makes
            # a cell relaxation walk the wrong way — so it is pinned by a round-trip test.
            self.results["stress"] = np.asarray(out["stress_voigt"], dtype=float)

    def _calculate_dandc(self, numbers, positions, properties):
        """The linear-scaling route, for a calculator built with ``dandc=``.

        Divide and conquer supplies the energy and the forces and nothing else. The properties it
        cannot supply are **refused by name** rather than served from a quietly-substituted exact
        SCF: a caller who asked for the approximate solver and silently received the exact one
        would have no way to tell, and the whole point of choosing this solver is that the exact
        one is unaffordable at that size.
        """
        wanted = set(properties)
        unsupported = wanted - {"energy", "free_energy", "forces"}
        if unsupported:
            from ase.calculators.calculator import PropertyNotImplementedError

            raise PropertyNotImplementedError(
                f"divide and conquer supplies energy and forces; {sorted(unsupported)} "
                "would need the exact SCF. Drop `dandc=` for those, or ask for energy and "
                "forces only."
            )
        options = dict(
            charge=self.charge,
            multiplicity=self.multiplicity,
            method=self.method,
            reference=self.reference,
            pbc_mode=self.pbc_mode,
            scf_tolerance=self.scf_tolerance,
            max_scf=self.max_scf,
            cphf_max_iterations=self.cphf_max_iterations,
            stability=self.stability,
            field=self.field,
            dipole_origin=self.dipole_origin,
            **self._periodic_kwargs(self.atoms),
            **self.dandc,
        )
        out = native.divide_and_conquer(numbers, positions, **options)
        self.results["energy"] = out["energy_ev"]
        # No smearing entropy is reported by this solver, so the two energies coincide. Setting
        # `free_energy` anyway keeps `get_potential_energy(force_consistent=True)` working rather
        # than raising, which is what ASE optimizers reach for first.
        self.results["free_energy"] = out["energy_ev"]
        self.results["forces"] = np.asarray(out["forces_ev_per_angstrom"], dtype=float)
        for key in ("fermi_ev", "subsystems", "largest_subsystem", "stored_density_elements"):
            if key in out:
                self.results[key] = out[key]

    # -- ASE interface -----------------------------------------------------------------

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        # ASE's `get_property` passes `atoms=None` when it already believes the calculator knows
        # its system, and `Calculator.calculate` only stores a copy when it is given one — so on a
        # calculator that has never run, `self.atoms` stayed `None` and the next line died on it.
        atoms = self._resolve(atoms)
        super().calculate(atoms, properties, system_changes)
        numbers = self.atoms.get_atomic_numbers()
        positions = self.atoms.get_positions()
        periodic = bool(np.asarray(self.atoms.get_pbc()).any())

        if self.dandc is not None:
            self._calculate_dandc(numbers, positions, properties)
            return

        # Stress, forces, and energy all come from one SCF, so a request for any of them that
        # includes stress is served by the single periodic call.
        if periodic and ("stress" in properties or "forces" in properties):
            out = native.stress(numbers, positions, **self._common(self.atoms))
            self.results["forces"] = np.asarray(out["forces_ev_per_angstrom"], dtype=float)
            self.results["energy"] = out["energy_ev"]
            self.results["free_energy"] = out["free_energy_ev"]
            self.results["heat_of_formation_kcal"] = out["heat_of_formation_kcal"]
            self._store_periodic(out)
        elif "forces" in properties:
            # A gradient call already performs the SCF and returns its energy. Reuse it for
            # normal ASE force/optimization steps instead of an extra single-point SCF first.
            force = native.forces(numbers, positions, **self._common(self.atoms))
            self.results["forces"] = np.asarray(force["forces_ev_per_angstrom"], dtype=float)
            self.results["energy"] = force["energy_ev"]
            self.results["free_energy"] = force["free_energy_ev"]
            self.results["heat_of_formation_kcal"] = force["heat_of_formation_kcal"]

        need_point = (
            "charges" in properties
            or "dipole" in properties
            or (
                ("energy" in properties or "free_energy" in properties)
                and "forces" not in properties
                and "stress" not in properties
            )
        )
        if need_point:
            point = native.single_point(numbers, positions, **self._common(self.atoms))
            self.results["energy"] = point["energy_ev"]
            self.results["free_energy"] = point["free_energy_ev"]
            self.results["charges"] = np.asarray(point["charges"], dtype=float)
            self.results["dipole"] = np.asarray(point["dipole_debye"], dtype=float) * DEBYE_TO_E_ANGSTROM
            self.results["heat_of_formation_kcal"] = point["heat_of_formation_kcal"]
            self.results["unrestricted"] = point["unrestricted"]
            # ``<S²>``, MOPAC's ``(S**2)``, or ``None`` for a restricted run. A UHF determinant is
            # not a spin eigenfunction, and how far this sits above ``S(S+1)`` is the only thing in
            # the output that says so — which matters most for ``stability="follow"``, where the
            # answer is a broken-symmetry solution on purpose.
            self.results["spin_squared"] = point["spin_squared"]
            self._store_periodic(point)

        # Everything that comes out of one Hessian is computed together, and only when one of
        # them is explicitly requested.
        #
        # This is the project's single lazy mechanism, not a second one: ASE's own
        # `get_property` / `results` / `check_state` machinery already caches per geometry and
        # invalidates on a move, so grouping here means a caller can ask for frequencies and then
        # IR intensities and pay for exactly one CPHF solve. A normal energy or forces cycle never
        # touches any of it.
        if self._VIBRATIONAL.intersection(properties):
            # Only `hessian` can live in `results` — the other four are not in
            # `implemented_properties` and ASE would refuse to hand them back. They are reached
            # through `_vibrational`, which is the same cache this call fills, so requesting the
            # Hessian here and the frequencies afterwards is still one solve.
            out = self._vibrational(atoms)
            self.results["hessian"] = np.asarray(out["hessian_ev_per_angstrom2"], dtype=float)

    def get_gradient(self, atoms=None) -> np.ndarray:
        """Energy **gradient** ∂E/∂x in eV/Å (= −forces), shape (N, 3). Forces themselves
        come from the standard ASE ``get_forces(atoms)`` on the base ``Calculator``."""
        atoms = self._resolve(atoms)
        grad = native.gradient(
            atoms.get_atomic_numbers(), atoms.get_positions(), **self._common(atoms)
        )
        return np.asarray(grad["gradient_ev_per_angstrom"], dtype=float)

    def get_hessian(self, atoms=None) -> np.ndarray:
        """Analytic Cartesian **Hessian** in eV/Å², shape (3N, 3N). Declared in
        ``implemented_properties`` but computed **lazily** — only this call (or
        ``get_property("hessian")``) triggers it, never a normal energy/forces cycle. Routed
        through ASE's property machinery so the result is cached per geometry.

        Molecules **and** periodic cells: a periodic system gives the zone-centre force
        constants, from the periodic analytic Hessian at Γ and from the perturbation solver at
        ``q = 0`` on a k mesh. For other wavevectors use :meth:`get_dfpt` or :meth:`get_phonons`,
        which is also what ``ase.phonons.Phonons`` would be reaching for.

        (This said "molecules only" through v0.2.1, which was stale in both halves: the extension
        had dispatched to the Γ periodic path since 0.2.0, and the k mesh it did refuse is now the
        `q = 0` response.)"""
        return self.get_property("hessian", atoms)

    def get_stress(self, atoms=None) -> np.ndarray:
        """Analytic **stress** in ASE's Voigt order ``[xx, yy, zz, yz, xz, xy]``, eV/Å³.

        Positive under tension. Periodic systems only."""
        return self.get_property("stress", atoms)

    def get_frequencies(self, atoms=None) -> np.ndarray:
        """Harmonic frequencies in cm⁻¹, ascending; negative means imaginary.

        Part of the vibrational group — see :meth:`get_ir_intensities`."""
        return np.asarray(self._vibrational(atoms)["frequencies_cm"], dtype=float)

    def get_normal_modes(self, atoms=None) -> np.ndarray:
        """Cartesian normal modes as the **columns** of a ``(3N, n)`` array, unit-normalized.

        ``n`` is ``3N − 6`` for a non-linear molecule and ``3N − 5`` for a linear one: the
        rigid-body subspace is projected out of the mass-weighted Hessian rather than filtered out
        of the spectrum, so there are no columns for it to occupy. A periodic cell keeps all ``3N``,
        because a branch index has to mean the same thing at every wavevector.
        """
        return np.asarray(self._vibrational(atoms)["cartesian_modes"], dtype=float)

    def get_ir_intensities(self, atoms=None) -> np.ndarray:
        """Double-harmonic IR intensities per mode, km/mol.

        One of the **vibrational group** — ``hessian``, ``frequencies``, ``normal_modes``,
        ``ir_intensities`` and ``dipole_derivatives``. Requesting any one of them computes the
        whole group from a single SCF and CPHF solve and caches it against the geometry, so
        asking for frequencies and then intensities costs one calculation, not two. None of them
        is touched by an ordinary energy or forces cycle.

        Molecules only, and physically meaningful only at a stationary point.

        ASE has no standard ``ir`` property, so these names are deliberately *not* in
        ``implemented_properties``; ``ase.vibrations.Infrared`` also works against this
        calculator's ``dipole``, by finite differences, if you would rather use ASE's own driver.
        """
        return np.asarray(self._infrared(atoms)["ir_intensities_km_per_mol"], dtype=float)

    def get_ir_spectrum(self, atoms=None) -> dict:
        """Frequencies, IR intensities and normal modes together, from one solve.

        The same numbers :meth:`get_frequencies`, :meth:`get_ir_intensities` and
        :meth:`get_normal_modes` return, handed back in one dictionary for callers who want the
        whole spectrum — plotting it, say — without three attribute lookups. Costs nothing extra:
        they all come out of the same cached solve.
        """
        out = self._vibrational(atoms)
        return {
            "frequencies_cm": np.asarray(out["frequencies_cm"], dtype=float),
            "ir_intensities_km_per_mol": np.asarray(
                out["ir_intensities_km_per_mol"], dtype=float
            ),
            "cartesian_modes": np.asarray(out["cartesian_modes"], dtype=float),
            "dipole_derivatives_debye_per_angstrom": np.asarray(
                out["dipole_derivatives_debye_per_angstrom"], dtype=float
            ),
        }

    def get_dipole_breakdown(self, atoms=None) -> dict:
        """The dipole split into its terms, in Debye, rather than as one vector.

        ASE's standard ``dipole`` property is the total in e·Å. This is the same quantity with its
        **point-charge**, **s–p hybrid** and **p–d hybrid** parts separate, mirroring MOPAC's
        ``POINT-CHG./HYBRID/SUM`` print. The p–d term is what a d-bearing atom contributes and
        what `pm7-rs` ≤ 0.2.0 omitted entirely; seeing it separately is how you tell.
        """
        atoms = self._resolve(atoms)
        point = self._cached(
            "single_point",
            atoms,
            lambda: native.single_point(
                atoms.get_atomic_numbers(), atoms.get_positions(), **self._common(atoms)
            ),
        )
        return {
            key: np.asarray(point[key], dtype=float)
            for key in (
                "dipole_debye",
                "dipole_point_charge_debye",
                "dipole_sp_hybrid_debye",
                "dipole_pd_hybrid_debye",
            )
            if key in point
        }

    def get_dipole_derivatives(self, atoms=None) -> np.ndarray:
        """The dense (3, 3N) dipole-derivative tensor ∂μ/∂x, in Debye/Å.

        Taken about the **input coordinate origin** regardless of ``dipole_origin``: a moving
        origin would make the tensor convention-dependent for an ion. See ``docs/theory.md``
        convention C-8."""
        return np.asarray(
            self._infrared(atoms)["dipole_derivatives_debye_per_angstrom"], dtype=float
        )

    def get_orbitals(self, atoms=None) -> dict:
        """Orbital energies, coefficients and occupations for both spin channels.

        Its own group: nothing in the vibrational set needs it and it needs nothing from them, so
        it is one plain SCF. Returns the dictionary :func:`pm7_rs.native.orbitals` produces,
        including ``ao_labels`` so the coefficient matrix can be read directly."""
        atoms = self._resolve(atoms)
        return self._cached(
            "orbitals",
            atoms,
            lambda: native.orbitals(
                atoms.get_atomic_numbers(), atoms.get_positions(), **self._common(atoms)
            ),
        )

    # -- periodic properties -----------------------------------------------------------
    #
    # These take an explicit argument (a q path, a supercell, a LO–TO direction), so they are not
    # ASE "properties": there is nothing for `get_property` to key on. They cache on **the
    # geometry and the argument together**, which is the part `results` could not express —
    # asking for the same q path twice must not solve it twice, and asking for a different one
    # must not be handed the first one's answer.

    def _require_cell(self, atoms, what):
        if not atoms.cell.rank:
            raise ValueError(f"{what} need a periodic cell; this Atoms object has none")
        return atoms.cell[:]

    def get_phonons(
        self,
        qpoints,
        atoms=None,
        *,
        supercell=(1, 1, 1),
        acoustic_sum_rule=True,
        lo_to_direction=None,
    ):
        """Phonon frequencies in cm⁻¹ at each **fractional** q, via a force-constant supercell.

        Exact at every q commensurate with ``supercell`` and Fourier-interpolated between, so a
        dispersion needs at least ``(2, 2, 2)`` and the force constants must have decayed inside
        whatever you pick. For arbitrary q with no supercell at all, use :meth:`get_dfpt`.

        **The calculator's ``kpts`` does not apply here**, and is not forwarded. Force constants
        come from the supercell's Γ point, and Γ of an ``n₁×n₂×n₃`` supercell *is* the
        ``n₁×n₂×n₃`` mesh of the cell — so ``supercell`` is the Brillouin-zone sampling knob for
        this route and ``kpts`` would be a second, contradictory one. It still governs the
        energies and forces this same calculator produces, and it does apply to :meth:`get_dfpt`,
        which samples k and solves the response at each q directly.

        ``lo_to_direction`` (Cartesian, not normalized) adds ``frequencies_cm_lo_to`` — the
        zone-centre frequencies with the non-analytic LO–TO term along that direction. It is
        opt-in and the direction is required, because the ``q → 0`` limit is direction dependent.
        **3-D only**, and it costs a field response on top of the force constants.

        The result also carries ``modes`` and ``cartesian_modes`` from 0.2.3 — the polarization
        vectors, which this route computed and discarded through 0.2.2. See
        :meth:`get_phonon_modes` for them as complex arrays rather than ``(real, imag)`` pairs.
        """
        atoms = self._resolve(atoms)
        cell = self._require_cell(atoms, "phonons")
        tag = (
            "phonons",
            _hashable(qpoints),
            tuple(int(n) for n in supercell),
            acoustic_sum_rule,
            _hashable(lo_to_direction),
        )
        common = {
            k: v
            for k, v in self._common_without_cell(atoms).items()
            if k not in ("kpoints", "kpoint_shift")
        }
        return self._cached(
            tag,
            atoms,
            lambda: native.phonons(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                cell,
                qpoints,
                supercell=supercell,
                acoustic_sum_rule=acoustic_sum_rule,
                lo_to_direction=lo_to_direction,
                **common,
            ),
        )

    def get_scf_stability(self, atoms=None):
        """Is the converged SCF solution a minimum, or only a stationary point?

        Returns the dict :func:`pm7_rs.native.scf_stability` produces: ``lowest_ev`` for the
        spin-preserving channel, ``lowest_triplet_ev`` for the spin-breaking one (``None`` when the
        solution is already unrestricted), and ``unstable``. Negative curvature means a saddle.

        Worth asking whenever a bond is being stretched — a dissociation curve, a transition state,
        a scan. A closed-shell solution that is perfectly good at equilibrium can be 100 kcal/mol
        above the right answer once pulled apart, with every step converged, and only the triplet
        channel sees it. Pass ``stability="follow"`` to ``PM7(...)`` to act on it rather than only
        measure it.
        """
        atoms = self._resolve(atoms)
        return native.scf_stability(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            **self._common(atoms),
        )

    def get_phonon_modes(self, qpoints, atoms=None, *, route="supercell", **kwargs):
        """Phonon frequencies **and polarization vectors**, as complex arrays.

        ``(frequencies, modes, cartesian_modes)``: ``frequencies[i]`` is ascending cm⁻¹ at q point
        ``i`` (negative means imaginary), and ``modes[i]`` is a complex ``(3N, 3N)`` array with
        **one mode per column**, in that same order. ``modes`` are eigenvectors of the mass-weighted
        dynamical matrix, unit norm per column; ``cartesian_modes`` are the Cartesian displacements
        ``m⁻¹ᐟ² e``, renormalized — the ones to move atoms along.

        This is :meth:`get_phonons` (or :meth:`get_dfpt`, with ``route="dfpt"``) with the
        ``(real, imag)`` pairs assembled into complex arrays, which is what a caller wants nine
        times out of ten. Everything else goes through as keywords, so ``supercell=``,
        ``acoustic_sum_rule=`` and the DFPT solver controls all apply as usual::

            freqs, modes, cart = calc.get_phonon_modes([(0, 0, 0)], supercell=(2, 2, 2))
            softest = cart[0][:, 0].real.reshape(-1, 3)     # displacement pattern, one row per atom
            atoms.positions += 0.1 * softest                # push the structure along it

        A note about the phase: away from the zone centre these are genuinely complex, because a
        phonon at ``q`` is ``u_A ∝ e_A e^{iq·R_A}``. Taking ``.real`` is only meaningful at ``q``
        where every Bloch phase is ``±1`` — the zone centre, and the zone-boundary points.
        """
        if route not in ("supercell", "dfpt"):
            raise ValueError(f"route must be 'supercell' or 'dfpt', not {route!r}")
        out = (
            self.get_dfpt(qpoints, atoms, **kwargs)
            if route == "dfpt"
            else self.get_phonons(qpoints, atoms, **kwargs)
        )
        complex_of = lambda pair: np.asarray(pair[0]) + 1j * np.asarray(pair[1])  # noqa: E731
        return (
            [np.asarray(f) for f in out["frequencies_cm"]],
            [complex_of(m) for m in out["modes"]],
            [complex_of(m) for m in out["cartesian_modes"]],
        )

    def get_band_structure(self, path=None, atoms=None, *, npoints=50):
        """Band energies along a k path.

        ``path`` may be an ASE :class:`~ase.dft.kpoints.BandPath` (from
        ``atoms.cell.bandpath(...)``), a path string such as ``"GXWL"``, or an explicit sequence of
        **fractional** k points. Given a ``BandPath`` or a string this returns an ASE
        :class:`~ase.spectrum.band_structure.BandStructure`, so ``.plot()`` works; given bare k
        points it returns the underlying dict, because a ``BandStructure`` cannot be built without
        a path to label the axis with.

        **Not** ASE's inherited ``Calculator.band_structure()``. That one reconstructs a band
        structure from a completed SCF through ``get_eigenvalues`` / ``get_ibz_k_points``, so it
        can only ever return the mesh the SCF ran on. This runs a fresh non-self-consistent
        diagonalization at whatever k points you ask for, which is what a dispersion needs — the
        two have the same name in the namespace and are not the same calculation.
        """
        atoms = self._resolve(atoms)
        cell = self._require_cell(atoms, "band structures")

        band_path = None
        if path is None or isinstance(path, str):
            band_path = atoms.cell.bandpath(path, npoints=npoints)
        elif hasattr(path, "kpts") and hasattr(path, "cell"):
            band_path = path
        kpath = band_path.kpts if band_path is not None else path

        tag = ("bands", _hashable(np.asarray(kpath, dtype=float)))
        out = self._cached(
            tag,
            atoms,
            lambda: native.band_structure(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                cell,
                kpath,
                **self._common_without_cell(atoms),
            ),
        )
        if band_path is None:
            return out

        from ase.spectrum.band_structure import BandStructure

        # One spin channel in the array ASE wants, `(spins, kpoints, bands)`. An unrestricted cell
        # is not handled here rather than being averaged into something that looks like a
        # restricted band structure.
        energies = np.asarray(out["energies_ev"], dtype=float)[None, :, :]
        return BandStructure(band_path, energies, reference=out["fermi_ev"])

    def get_dfpt(
        self,
        qpoints,
        atoms=None,
        *,
        lo_to_direction=None,
        dfpt_tolerance=None,
        dfpt_max_iterations=None,
        dfpt_mixing=None,
    ):
        """Phonons at arbitrary **fractional** q by perturbation theory — no supercell.

        Raises if the linear response fails to converge, rather than returning a number: the
        response is a linear fixed point, so a failure is a divergence, not a near miss. The three
        solver controls are the ones to reach for when it does: a looser ``dfpt_tolerance``, a
        larger ``dfpt_max_iterations``, or a smaller ``dfpt_mixing``. They stopped at this boundary
        until v0.2.2 — ``native.dfpt`` took them and the calculator did not, so the only way to
        touch them from ASE was to leave ASE.

        The cache key carries them, so changing one re-solves rather than handing back the previous
        settings' answer.
        """
        atoms = self._resolve(atoms)
        cell = self._require_cell(atoms, "DFPT phonons")
        settings = {
            "lo_to_direction": lo_to_direction,
            "dfpt_tolerance": dfpt_tolerance,
            "dfpt_max_iterations": dfpt_max_iterations,
            "dfpt_mixing": dfpt_mixing,
        }
        return self._cached(
            (
                "dfpt",
                _hashable(qpoints),
                _hashable(lo_to_direction),
                tuple(sorted((k, v) for k, v in settings.items() if k != "lo_to_direction")),
            ),
            atoms,
            lambda: native.dfpt(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                cell,
                qpoints,
                **settings,
                **self._common_without_cell(atoms),
            ),
        )

    def get_born_charges(
        self,
        atoms=None,
        *,
        lo_to_direction=None,
        dfpt_tolerance=None,
        dfpt_max_iterations=None,
        dfpt_mixing=None,
    ):
        """Born effective charges, ``eps_inf``, and optionally the LO–TO term.

        ``born_charges[A][a][b]`` is ``d²E/df_a dR_{A,b}`` in units of the elementary charge, with
        **a the field index and b the displacement index**. See :func:`pm7_rs.native.born_charges`
        for what PM7 does and does not get right here — ``born_charges`` is quantitative,
        ``dielectric`` is not.
        """
        atoms = self._resolve(atoms)
        cell = self._require_cell(atoms, "Born charges")
        # `cell` goes positionally, and `born_charges` takes no field or dipole origin: the field
        # it applies is its own perturbation, through the commutator.
        common = {
            k: v
            for k, v in self._common_without_cell(atoms).items()
            if k not in ("field", "dipole_origin")
        }
        settings = {
            "dfpt_tolerance": dfpt_tolerance,
            "dfpt_max_iterations": dfpt_max_iterations,
            "dfpt_mixing": dfpt_mixing,
        }
        return self._cached(
            ("born", _hashable(lo_to_direction), tuple(sorted(settings.items()))),
            atoms,
            lambda: native.born_charges(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                cell,
                lo_to_direction=lo_to_direction,
                **settings,
                **common,
            ),
        )

    def get_polarizability(self, atoms=None, *, long_range=None, **solver):
        """The cell's electronic polarizability, in Bohr³.

        Defined in every dimensionality, where ``eps_inf`` is not: use this for a chain or a slab
        and :meth:`get_dielectric_with_extent` when you are ready to say where the material stops.
        """
        atoms = self._resolve(atoms)
        cell = self._require_cell(atoms, "a polarizability")
        tag = ("polarizability", long_range, _hashable(sorted(solver.items())))
        return self._cached(
            tag,
            atoms,
            lambda: native.polarizability(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                cell,
                long_range=long_range,
                **self._common_without_cell(atoms),
                **solver,
            ),
        )

    def get_static_dielectric(self, atoms=None, **solver):
        """``eps^0``: the clamped-ion tensor plus the ionic term, for a 3-D cell.

        **Read ``soft_optical_modes``.** Non-zero means the geometry is not a minimum and the
        ionic term is missing what those modes would have contributed. ``skipped_modes`` is the
        acoustic branch and is always three.
        """
        atoms = self._resolve(atoms)
        cell = self._require_cell(atoms, "a static dielectric tensor")
        return self._cached(
            ("static_dielectric", _hashable(sorted(solver.items()))),
            atoms,
            lambda: native.static_dielectric(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                cell,
                **self._common_without_cell(atoms),
                **solver,
            ),
        )

    def get_dielectric_origin_sensitivity(self, offset, atoms=None, **solver):
        """How much ``alpha`` moves when the cell is translated by ``offset`` (**Bohr**).

        Near machine precision means the position operator's periodicity argument holds for this
        system. A large value means the polarizability is a statement about the origin.
        """
        atoms = self._resolve(atoms)
        cell = self._require_cell(atoms, "an origin sensitivity")
        return self._cached(
            ("origin_sensitivity", _hashable(offset), _hashable(sorted(solver.items()))),
            atoms,
            lambda: native.dielectric_origin_sensitivity(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                cell,
                offset,
                **self._common_without_cell(atoms),
                **solver,
            ),
        )

    def get_berry_polarization(self, atoms=None, *, strings=16, **solver):
        """Berry-phase electronic polarization, defined modulo the returned ``quantum``.

        Compare two of these with their difference reduced onto the nearest branch, never by
        subtracting ``polarization`` — a displacement that crosses a branch makes the raw
        difference wrong by exactly one quantum.
        """
        atoms = self._resolve(atoms)
        cell = self._require_cell(atoms, "a Berry phase")
        return self._cached(
            ("berry", strings, _hashable(sorted(solver.items()))),
            atoms,
            lambda: native.berry_polarization(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                cell,
                strings=strings,
                **self._common_without_cell(atoms),
                **solver,
            ),
        )

    def get_finite_field(self, field, divisions, atoms=None, **solver):
        """Converge the cell in a finite field applied **along** a periodic direction.

        For a field orthogonal to every lattice vector, use ``PM7(field=...)`` instead — that is an
        ordinary calculation and this machinery is not needed. Along a periodic direction the
        ``E·R`` potential is unbounded, so this minimizes the electric enthalpy instead.

        ``divisions`` is the k mesh and the string length at once; an axis the field touches needs
        at least three points.
        """
        atoms = self._resolve(atoms)
        cell = self._require_cell(atoms, "a finite field")
        common = {
            k: v
            for k, v in self._common_without_cell(atoms).items()
            if k not in ("kpoints", "kpoint_shift", "smearing", "field", "dipole_origin")
        }
        return self._cached(
            ("finite_field", _hashable(field), _hashable(divisions),
             _hashable(sorted(solver.items()))),
            atoms,
            lambda: native.finite_field(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                cell,
                divisions,
                field,
                **common,
                **solver,
            ),
        )

    def get_dielectric_with_extent(
        self,
        atoms=None,
        *,
        slab_thickness=None,
        wire_cross_section=None,
        dfpt_tolerance=None,
        dfpt_max_iterations=None,
        dfpt_mixing=None,
    ):
        """``eps_inf`` for a chain or a slab, which needs an assigned extent.

        ``get_born_charges`` leaves ``dielectric`` as the identity below three dimensions, because
        ``eps`` needs a volume and a slab's cell has an area. Give a ``slab_thickness`` (Bohr) or a
        ``wire_cross_section`` (Bohr²) — exactly one, and required, because a supercell says where
        the atoms are and not where the material stops.

        The result also carries ``sheet_parallel_bohr`` and ``sheet_perpendicular_bohr``, which are
        **free of the extent** and are what a slab can quote without choosing one, and
        ``axis_mixing``, which says how much the per-principal-axis depolarization assumed.
        """
        atoms = self._resolve(atoms)
        cell = self._require_cell(atoms, "a dielectric tensor with an assigned extent")
        common = {
            k: v
            for k, v in self._common_without_cell(atoms).items()
            if k not in ("field", "dipole_origin")
        }
        settings = {
            "slab_thickness": slab_thickness,
            "wire_cross_section": wire_cross_section,
            "dfpt_tolerance": dfpt_tolerance,
            "dfpt_max_iterations": dfpt_max_iterations,
            "dfpt_mixing": dfpt_mixing,
        }
        return self._cached(
            ("extent", tuple(sorted(settings.items(), key=lambda kv: kv[0]))),
            atoms,
            lambda: native.dielectric_with_extent(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                cell,
                **settings,
                **common,
            ),
        )
    def write_molden(self, path=None, atoms=None, *, basis="sto-6g", comment=None):
        """The converged wavefunction in Molden format.

        Writes to ``path`` if given and returns the text either way. Molecules only — Molden's
        ``[MO]`` section is a list of molecular orbitals, not Bloch states.
        """
        atoms = self._resolve(atoms)
        text = native.molden(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            basis=basis,
            comment=comment,
            **{
                k: v
                for k, v in self._common(atoms).items()
                if k
                in ("charge", "multiplicity", "method", "reference", "field", "dipole_origin")
            },
        )
        if path is not None:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        return text


__all__ = ["PM7"]
