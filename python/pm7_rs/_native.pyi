# SPDX-License-Identifier: GPL-3.0-or-later
"""Type stubs for the compiled `pm7_rs._native` extension (PEP 561).

A native extension carries no signatures a type checker can read, so without this file every call
into it is `Any` and nothing downstream is checked either. The shapes here mirror the `#[pyo3(
signature = ...)]` attributes in `src/python.rs`; `python/tests/test_native.py` exercises the same
keywords, so a signature that drifts out of step here shows up there as a `TypeError` rather than
as silence.

Units at this boundary are the ones the docstrings state: Ångström for positions and cells, eV for
energies, eV/Å for forces, eV/Å³ for stress.
"""

from typing import Any, Literal, Sequence, TypedDict

Vector = Sequence[float]
Matrix3 = Sequence[Sequence[float]]
Smearing = tuple[str, float, int]
# The accepted values, exported for callers who want to constrain their own code. The function
# signatures below take a plain `str` rather than these: the extension validates at run time and
# raises `ValueError` naming the accepted set, and typing the parameters as `Literal` made every
# call that passes a `str` variable — which is what `native.py` does, and what a caller reading a
# method name from a config file does — a type error for no benefit.
Method = Literal["pm7", "pm7-ts", "pm7-", "pm7-minus", "pm7-hh", "pm7-sparkle"]
Reference = Literal["auto", "rhf", "r", "uhf", "u"]
PbcMode = Literal["ewald", "mopac"]

class SinglePointResult(TypedDict, total=False):
    """Keys marked *periodic* appear only when `cell` was given."""

    energy_ev: float
    #: Mermin **electronic** free energy `E - TS` at the smearing width; equals `energy_ev`
    #: exactly when there is no smearing. Not a thermochemical Gibbs free energy.
    free_energy_ev: float
    heat_of_formation_kcal: float
    charges: list[float]
    dipole_debye: list[float]
    dipole_point_charge_debye: list[float]
    dipole_sp_hybrid_debye: list[float]
    dipole_pd_hybrid_debye: list[float]
    dipole_origin_bohr: list[float]
    homo_ev: float
    lumo_ev: float
    homo_ev_beta: float | None
    lumo_ev_beta: float | None
    gap_ev: float | None
    orbital_source: str
    mo_energies_ev: list[float]
    n_occ: int
    field_ev: float | None
    electronic_ev: float
    core_ev: float
    energy_hartree: float
    unrestricted: bool
    #: `<S^2>`, MOPAC's `(S**2)`. `None` for a restricted run — where it is `S(S+1)` by
    #: construction — and for a k-mesh run. The gap between this and `S(S+1)` is the spin
    #: contamination, which `stability="follow"` introduces on purpose.
    spin_squared: float | None
    iterations: int
    converged: bool
    method: str
    charge: float
    multiplicity: int
    reference: str
    # Periodic only.
    stress: list[list[float]]
    stress_voigt: list[float]
    pressure_gpa: float
    fermi_ev: float
    entropy_ev: float
    n_kpoints: int
    volume_angstrom3: float
    ewald_ev: float
    background_ev: float
    makov_payne_ev: float

def single_point(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    cell: Matrix3 | None = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: Smearing | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
) -> SinglePointResult: ...
def gradient(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    cell: Matrix3 | None = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: Smearing | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
) -> dict[str, Any]: ...
def forces(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    cell: Matrix3 | None = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: Smearing | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
) -> dict[str, Any]: ...
def stress(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    cell: Matrix3 | None = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: Smearing | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
) -> dict[str, Any]: ...
def optimize(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    cell: Matrix3 | None = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: Smearing | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
    relax_cell: bool = ...,
    gtol: float | None = ...,
    stress_tol: float | None = ...,
    max_iter: int | None = ...,
    stability_every: int | None = ...,
) -> dict[str, Any]: ...
def frequencies(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    cell: Matrix3 | None = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: Smearing | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    projection: str | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
) -> dict[str, Any]: ...
def hessian(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    cell: Matrix3 | None = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: Smearing | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
) -> dict[str, Any]: ...
def phonons(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    cell: Matrix3,
    qpoints: Sequence[Vector],
    lo_to_direction: Vector | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    supercell: Sequence[int] | None = ...,
    pbc: Sequence[bool] | None = ...,
    acoustic_sum_rule: bool = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: tuple[str, float, int] | None = ...,
    pbc_mode: str | None = ...,
) -> dict[str, Any]: ...
def dfpt(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    cell: Matrix3,
    qpoints: Sequence[Vector],
    lo_to_direction: Vector | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: tuple[str, float, int] | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
    dfpt_tolerance: float | None = ...,
    dfpt_max_iterations: int | None = ...,
    dfpt_mixing: float | None = ...,
    long_range: str | None = ...,
    keep_response: bool = ...,
) -> dict[str, Any]: ...
def dielectric_with_extent(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    cell: Matrix3,
    slab_thickness: float | None = ...,
    wire_cross_section: float | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: tuple[str, float, int] | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    dfpt_tolerance: float | None = ...,
    dfpt_max_iterations: int | None = ...,
    dfpt_mixing: float | None = ...,
) -> dict[str, Any]: ...
def born_charges(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    cell: Matrix3,
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: tuple[str, float, int] | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    dfpt_tolerance: float | None = ...,
    dfpt_max_iterations: int | None = ...,
    dfpt_mixing: float | None = ...,
    lo_to_direction: Vector | None = ...,
) -> dict[str, Any]: ...
def molden(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    basis: str = ...,
    comment: str | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
) -> str: ...
def band_structure(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    cell: Matrix3,
    kpath: Sequence[Vector],
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: Smearing | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
) -> dict[str, Any]: ...
def divide_and_conquer(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    buffer: float = ...,
    core_size: int = ...,
    cell: Matrix3 | None = ...,
    pbc: Sequence[bool] | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
) -> dict[str, Any]: ...

def orbitals(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    cell: Matrix3 | None = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: Smearing | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
) -> dict[str, Any]: ...
def scf_stability(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    cell: Matrix3 | None = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: Smearing | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
) -> dict[str, Any]: ...
def vibrations(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    cell: Matrix3 | None = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Vector | None = ...,
    smearing: tuple[str, float, int] | None = ...,
    pbc_mode: str | None = ...,
    field: Vector | None = ...,
    dipole_origin: str | None = ...,
    hessian: bool = ...,
    frequencies: bool = ...,
    modes: bool = ...,
    ir: bool = ...,
    orbital_response: bool = ...,
    step: float = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    projection: str | None = ...,
) -> dict[str, Any]: ...

def polarizability(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    cell: Matrix3,
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Sequence[float] | None = ...,
    smearing: tuple[str, float, int] | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    dfpt_tolerance: float | None = ...,
    dfpt_max_iterations: int | None = ...,
    dfpt_mixing: float | None = ...,
    long_range: str | None = ...,
) -> dict[str, Any]: ...
def dielectric_origin_sensitivity(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    cell: Matrix3,
    offset: Vector,
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Sequence[float] | None = ...,
    smearing: tuple[str, float, int] | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    dfpt_tolerance: float | None = ...,
    dfpt_max_iterations: int | None = ...,
    dfpt_mixing: float | None = ...,
) -> float: ...
def static_dielectric(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    cell: Matrix3,
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Sequence[float] | None = ...,
    smearing: tuple[str, float, int] | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    dfpt_tolerance: float | None = ...,
    dfpt_max_iterations: int | None = ...,
    dfpt_mixing: float | None = ...,
) -> dict[str, Any]: ...
def berry_polarization(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    cell: Matrix3,
    strings: int = ...,
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    pbc: Sequence[bool] | None = ...,
    kpoints: Sequence[int] | None = ...,
    kpoint_shift: Sequence[float] | None = ...,
    smearing: tuple[str, float, int] | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
) -> dict[str, Any]: ...
def finite_field(
    numbers: Sequence[int],
    positions: Sequence[Vector],
    cell: Matrix3,
    divisions: Sequence[int],
    field: Vector,
    charge: float = ...,
    multiplicity: int = ...,
    method: str = ...,
    reference: str = ...,
    pbc: Sequence[bool] | None = ...,
    pbc_mode: str | None = ...,
    scf_tolerance: float | None = ...,
    max_scf: int | None = ...,
    cphf_max_iterations: int | None = ...,
    stability: str | None = ...,
    exchange_cutoff: tuple[float, float] | None = ...,
    use_diis: bool | None = ...,
    field_tolerance: float | None = ...,
    field_max_iterations: int | None = ...,
    field_mixing: float | None = ...,
) -> dict[str, Any]: ...
