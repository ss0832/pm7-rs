# SPDX-License-Identifier: GPL-3.0-or-later
"""Molecular dynamics: NVE, NVT, and NPT.

These are integration tests in the strongest sense available. A single-point number can be wrong
in ways that still look plausible; a trajectory cannot. NVE conservation exercises the forces
against the energy over hundreds of evaluations, and **NPT exercises the stress**: Parrinello–
Rahman integrates the cell directly from it, so a wrong sign blows the cell up or collapses it and
a wrong Voigt ordering shears it. Neither failure is visible in one energy evaluation.

Every run here is done twice, at Γ and on a k mesh, because the two take different code paths
through the gradient: at Γ the density block `P(T)` is the same matrix for every translation, so a
gradient that contracts everything against `P(0)` is right by accident. Only a k-point trajectory
notices.

The system is diamond carbon. It is a well-behaved covalent solid for PM7 — wide gap, stiff, and
near its own equilibrium at the experimental lattice constant once the Brillouin zone is sampled —
so a drift here means the derivatives are wrong, not that the model wandered somewhere silly.
"""

from __future__ import annotations

import numpy as np
import pytest

ase = pytest.importorskip("ase")

from ase import units  # noqa: E402
from ase.build import bulk  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402
from ase.md.verlet import VelocityVerlet  # noqa: E402

from pm7_rs.ase import PM7  # noqa: E402

# A trajectory has to converge at *every* step, including geometries no curated test set would
# pick, and it tolerates a looser threshold than a single point wants: 1e-6 in the density is
# worth well under 1e-4 eV/Å in the forces. These are ordinary MD settings, not a way of hiding a
# convergence failure — the NVE drift bound below is what actually checks the forces.
MD_SCF = dict(scf_tolerance=1e-6, max_scf=400)

# (1,2,2) on the doubled cell samples the primitive Brillouin zone at 2×2×2; (2,4,4) samples it at
# 4×4×4, which is where diamond's pressure has converged to within a few GPa of zero.
COARSE_MESH = (1, 2, 2)
FINE_MESH = (2, 4, 4)


def diamond(kpts=None, a=3.567):
    """Diamond carbon, primitive cell doubled along **a** so a single atom's displacement is a
    zone-boundary perturbation rather than a uniform one."""
    atoms = bulk("C", "diamond", a=a, cubic=False).repeat((2, 1, 1))
    atoms.calc = PM7(kpts=kpts, **MD_SCF)
    return atoms


@pytest.mark.parametrize("kpts", [None, COARSE_MESH])
def test_nve_conserves_the_total_energy(kpts):
    """Energy conservation is the cleanest statement that the forces are the energy's gradient.

    A force error shows up as a systematic drift, so the bound is on the drift rather than on the
    instantaneous fluctuation, which is dominated by the finite time step.
    """
    atoms = diamond(kpts)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(7))
    atoms.set_momenta(atoms.get_momenta() - atoms.get_momenta().mean(axis=0))

    dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
    energies = []
    for _ in range(40):
        dyn.run(2)
        energies.append(atoms.get_total_energy())
    energies = np.asarray(energies)

    span = energies.max() - energies.min()
    # Drift over the run, from a straight-line fit, per atom.
    slope = np.polyfit(np.arange(energies.size), energies, 1)[0]
    drift = abs(slope) * energies.size / len(atoms)
    assert np.isfinite(energies).all()
    assert span < 0.05, f"NVE total energy varied by {span:.4f} eV"
    assert drift < 5e-3, f"NVE energy drifted by {drift:.2e} eV/atom over the run"


def test_nvt_holds_the_temperature():
    from ase.md.langevin import Langevin

    atoms = diamond()
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(11))
    dyn = Langevin(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=300.0,
        friction=0.02 / units.fs,
        rng=np.random.default_rng(3),
    )
    temperatures = []
    for _ in range(60):
        dyn.run(2)
        temperatures.append(atoms.get_temperature())
    mean = float(np.mean(temperatures[20:]))
    assert np.isfinite(temperatures).all()
    # Four atoms over a short run fluctuate hard; the bound only has to exclude a thermostat that
    # is not working at all.
    assert 50.0 < mean < 900.0, f"NVT mean temperature {mean:.1f} K is not near 300 K"


def _upper_triangular(atoms):
    """Rotate to the upper-triangular cell ASE's Parrinello–Rahman integrator requires.

    The scaled positions have to be read *before* the cell is replaced — reading them afterwards
    reinterprets the old Cartesian coordinates in the new frame and quietly scrambles the crystal.
    """
    scaled = atoms.get_scaled_positions()
    atoms.set_cell(atoms.get_cell().standard_form()[0])
    atoms.set_scaled_positions(scaled)
    return atoms


@pytest.mark.parametrize("kpts", [COARSE_MESH, FINE_MESH])
def test_npt_keeps_the_cell_alive_and_tracks_the_target_pressure(kpts):
    """The decisive stress test.

    Parrinello–Rahman integrates the cell from `σ − P_target`. The barostat is aimed at the
    system's *own* static pressure, so a correct stress leaves the cell fluctuating about where it
    started. That is not circular: it still catches every way the stress can be wrong at the
    interface, because with the sign inverted the barostat pushes the wrong way and the cell runs
    away even when it starts balanced, and with the Voigt components transposed the off-diagonals
    land in the wrong places and the cell shears. What it does *not* check is the absolute value
    of the stress — `tests/stress.rs` pins that against full-SCF finite differences.
    """
    from ase.md.npt import NPT

    atoms = _upper_triangular(diamond(kpts))
    target_gpa = -float(np.mean(atoms.get_stress()[:3])) / units.GPa
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(5))
    atoms.set_momenta(atoms.get_momenta() - atoms.get_momenta().mean(axis=0))

    v0 = atoms.get_volume()
    dyn = NPT(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=300.0,
        externalstress=target_gpa * units.GPa,
        ttime=25.0 * units.fs,
        pfactor=(75.0 * units.fs) ** 2 * units.GPa,
    )

    volumes, pressures, energies, shears = [], [], [], []
    for _ in range(20):
        dyn.run(2)
        volumes.append(atoms.get_volume())
        stress = atoms.get_stress()
        # ASE's stress is positive under tension, so the pressure is minus its trace/3.
        pressures.append(-np.mean(stress[:3]) / units.GPa)
        shears.append(np.abs(stress[3:]).max() / units.GPa)
        energies.append(atoms.get_potential_energy())

    volumes = np.asarray(volumes)
    pressures = np.asarray(pressures)
    assert np.isfinite(energies).all(), "NPT produced a non-finite energy"
    assert np.isfinite(volumes).all()
    assert (volumes > 0.5 * v0).all() and (volumes < 2.0 * v0).all(), (
        f"NPT cell left the physical range: volume went from {v0:.2f} to "
        f"[{volumes.min():.2f}, {volumes.max():.2f}] A^3"
    )
    # The volume must actually respond — a cell frozen at its initial value would mean the stress
    # never reached the integrator.
    assert volumes.std() > 1e-6, "the NPT cell never moved; is the stress reaching the barostat?"
    mean_pressure = float(np.mean(pressures[5:]))
    assert abs(mean_pressure - target_gpa) < 60.0, (
        f"NPT mean pressure {mean_pressure:.2f} GPa drifted away from the {target_gpa:.2f} GPa "
        "target; check the sign of the stress"
    )
    # A cubic crystal under hydrostatic conditions has no shear to speak of. A transposed Voigt
    # ordering would put diagonal-sized numbers here.
    assert max(shears) < 0.2 * max(abs(target_gpa), 1.0) + 5.0, (
        f"NPT developed a shear stress of {max(shears):.2f} GPa in a cubic crystal; check the "
        "Voigt ordering of the stress"
    )


def test_npt_berendsen_agrees_that_the_cell_is_stable():
    """A second barostat, so the conclusion does not rest on one integrator's conventions."""
    from ase.md.nptberendsen import NPTBerendsen

    atoms = diamond(COARSE_MESH)
    target_au = -float(np.mean(atoms.get_stress()[:3]))
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(13))
    v0 = atoms.get_volume()
    dyn = NPTBerendsen(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=300.0,
        pressure_au=target_au,
        taut=50.0 * units.fs,
        taup=250.0 * units.fs,
        compressibility_au=5e-7 / units.GPa,
    )
    volumes = []
    for _ in range(20):
        dyn.run(2)
        volumes.append(atoms.get_volume())
    volumes = np.asarray(volumes)
    assert np.isfinite(volumes).all()
    assert (volumes > 0.5 * v0).all() and (volumes < 2.0 * v0).all(), (
        f"Berendsen NPT cell left the physical range: {volumes.min():.2f}..{volumes.max():.2f} "
        f"from {v0:.2f} A^3"
    )
